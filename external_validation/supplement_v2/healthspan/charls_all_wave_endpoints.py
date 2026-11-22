"""Map the original CHARLS 2020 question routing and rebuild 2011/2015 landmarks."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from prepare_healthspan import OUT,DATA,EXT,META,ORGANS,ADL,IADL,date,read
from repeat_biomarkers import charls,scores,verify
DX=dict(hibpe=1,dyslipe=2,diabe=3,cancre=4,lunge=5,livere=6,hearte=7,stroke=8,kidneye=9,digeste=10,psyche=11,memrye=12,arthre=14,asthmae=15)
def any_item(d):
    return d.max(axis=1).where(d.eq(1).any(axis=1)|d.notna().all(axis=1))
def recall(d,p):
    words=d[[p+str(i) for i in range(1,11)]]
    none=d[p+'11'].gt(0); refused=d[p+'12'].gt(0)
    if p+'13' in d: refused|=d[p+'13'].gt(0)
    n=words.gt(0).sum(axis=1)
    # All ten item checkboxes and explicit assessment of none/rejection are required.
    return n.where(words.notna().all(axis=1)&d[p+'11'].notna()&d[p+'12'].notna()&~refused&(n.gt(0)^none))
def charls2020():
    path=next(DATA.rglob('2020charls/Health_Status_and_Functioning.dta'))
    avars=['db001','db003','db005','db007','db009']; ivars=['db022','db020','db016','db014']
    cols=avars+ivars+[f'dc{i:03}' for i in range(16,26)]
    for diag in list(DX.values()): cols += [f'da002_{diag}_',f'da003_{diag}_']
    for pre,maxi in [('dc012_s',13),('dc028_s',12)]:
        cols += [pre+str(i) for i in range(1,maxi+1)]
    raw=read(path,['ID',*cols]).set_index('ID',verify_integrity=True)
    sample=read(next(DATA.rglob('2020charls/Sample_Infor.dta')),['ID','iyear','imonth','died']).set_index('ID',verify_integrity=True)
    dates=date(pd.to_numeric(sample.iyear),pd.to_numeric(sample.imonth))
    d=pd.DataFrame({'visitdate':dates.where(sample.died.eq(0))},index=raw.index)
    audit={}
    for k,i in DX.items():
        old=raw[f'da002_{i}_']; direct=raw[f'da003_{i}_'].map({1:1.,2:0.})
        history=old.isin([1,2,3])
        conflict=history&direct.eq(0)
        v=direct.where(~(history&direct.isna()),1.).mask(conflict)
        d[k]=v
        audit[k]={'supported_only_by_reconfirmed_prior':int((history&direct.isna()).sum()),
                  'specific_no_diagnosis_or_confirmed_yes':int(direct.notna().sum()),
                  'conflict_excluded':int(conflict.sum()),'unascertained':int(v.isna().sum())}
    d['adl']=any_item(raw[avars].apply(lambda v:v.map({1:0,2:1,3:1,4:1})))
    d['iadl']=any_item(raw[ivars].apply(lambda v:v.map({1:0,2:1,3:1,4:1})))
    # Both positive-affect CES-D items (hopeful and happy) are reversed.
    mood=pd.DataFrame({i:raw[f'dc{i:03}'].where(lambda x:x.between(1,4)) for i in range(16,26)})
    for j in (20,23):mood[j]=5-mood[j]
    d['cesd10']=(mood-1).sum(axis=1,min_count=10)
    d['recall']=recall(raw,'dc012_s')+recall(raw,'dc028_s')
    return d,audit

def timed_sample(year):
    raw=next(DATA.rglob(f'{year}charls/Sample_Infor.dta'))
    d=read(raw,['ID','died','iyear','imonth']).set_index('ID',verify_integrity=True)
    date_=date(pd.to_numeric(d.iyear),pd.to_numeric(d.imonth))
    alive=read(next(DATA.rglob(f'{year}charls/Demographic_Background.dta')),['ID']).ID
    assert not d.index[d.died.eq(1)].isin(alive).any()
    return date_.where(d.died.eq(0)&d.index.isin(alive)),date_.where(d.died.eq(1))

def cohort2015(blood,h):
    blood=blood.dropna(subset=['age','origin']).loc[lambda x:x.age.ge(20)]
    feats=verify()['features']
    lab_cols='wbc hgb mcv platelets triglycerides creatinine bun hdl cholesterol glucose uric_acid hba1c crp'.split()
    blood=blood.loc[blood[lab_cols].notna().any(axis=1)]
    h=h.reindex(blood.index)
    raw=scores(blood,feats)
    reference=json.loads((EXT/'nhanes3_organ'/'organ_score_reference.json').read_text(encoding='utf8'))['panels']
    b=pd.DataFrame({'age':blood.age,'sex':h.ragender.map({1:1,2:0}),'education':h.raeducl,'origin':blood.origin,'full_z':raw.full_accel/reference['full']['accel_sd_years']})
    for o in ORGANS:b[o]=raw[o+'_accel']/reference[o]['accel_sd_years']
    a4,d4=timed_sample(2018);a5,d5=timed_sample(2020)
    follow=pd.concat([a4.rename('a4'),d4.rename('d4'),a5.rename('a5'),d5.rename('d5')],axis=1).reindex(b.index)
    # Death interval only. Survivors need a personal living follow-up record.
    alive=pd.concat([b.origin,follow.a4,follow.a5],axis=1).max(axis=1)
    hi=follow[['d4','d5']].min(axis=1)
    b['death']=hi.notna().astype(int)
    mid=alive+(hi-alive)/2
    conflict=hi.notna()&(hi<=alive)
    final=mid.where(b.death.eq(1),alive)
    b['death_or_censor']=(final-b.origin)/pd.Timedelta(days=365.25)
    b['death_upper']=((hi-b.origin)/pd.Timedelta(days=365.25)).where(b.death.eq(1),b.death_or_censor)
    b['horizon']=(max(a5.max(),d5.max())+pd.offsets.MonthEnd(0)-b.origin)/pd.Timedelta(days=365.25)
    b=b[~conflict&b.death_or_censor.gt(0)]
    return b

def main():
    v5,aud=charls2020()
    harm=next(DATA.rglob('H_CHARLS_D_Data.dta'))
    stubs=['iwstat','iwy','iwm','tr20','imrc','dlrc','cesd10','gripsum','wspeed',*ADL,*IADL,*DX]
    h=read(harm,['ID','ragender','raeducl']+[f'r{w}{s}' for w in (1,2,3,4) for s in stubs]).set_index('ID',verify_integrity=True)
    testpath=next(DATA.rglob('2018charls/Cognition.dta'))
    testcols=[f'{stem}{i}' for stem in ['dc028_w4_s','dc047_w4_s'] for i in range(1,13)]
    test=read(testpath,['ID',*testcols]).set_index('ID',verify_integrity=True)
    checks={}
    for field,stem in [('imrc','dc028_w4_s'),('dlrc','dc047_w4_s')]:
        diff=(recall(test,stem)-h['r4'+field]).dropna()
        assert (diff==0).all() and len(diff)>10000
        checks[field]=len(diff)
    waves={5:v5}
    for w in (1,2,3,4):
        st=f'r{w}';d=pd.DataFrame({'visitdate':date(h[st+'iwy'],h[st+'iwm']).where(h[st+'iwstat'].eq(1))})
        for name in DX:d[name]=h[st+name].where(lambda v:v.isin([0,1]))
        for name,items in [('adl',ADL),('iadl',IADL)]:d[name]=any_item(h[[st+s for s in items]].where(lambda x:x.isin([0,1])))
        d['recall']=h[st+'tr20'].where(lambda x:x.between(0,20))
        d['cesd10']=h[st+'cesd10'].where(lambda x:x.between(0,30))
        d['grip']=h.get(st+'gripsum',pd.Series(np.nan,index=h.index)).where(lambda x:x.gt(0))
        d['walk_seconds']=h.get(st+'wspeed',pd.Series(np.nan,index=h.index)).where(lambda x:x.gt(0))
        waves[w]=d
    base1=pd.read_csv(OUT/'charls_w5_baseline.csv.gz',dtype={'ID':str}).set_index('ID')
    base1['origin']=waves[1].visitdate
    _,raw3=charls()
    base3=cohort2015(raw3,h)
    for w0,b in [(1,base1),(3,base3)]:
        first=waves[w0].reindex(b.index)
        prior=first[list(DX)]
        table=[]
        for w in range(w0,6):
            d=waves[w].reindex(b.index).copy()
            elapsed=(d.visitdate-b.origin)/pd.Timedelta(days=365.25)
            d['time']=elapsed
            d['wave']=w
            d['dxcount']=d[list(DX)].sum(axis=1,min_count=len(DX))
            d['multimorbidity']=d.dxcount.ge(2).astype(float).where(d.dxcount.notna())
            has=(d[list(DX)].eq(1)&prior.eq(0)).any(axis=1)
            complete=(prior.eq(1)|d[list(DX)].notna()).all(axis=1)
            d['new_chronic']=has.astype(float).where(prior.notna().all(axis=1)&(has|complete))
            d.loc[d.time.isna(),d.columns.difference(['wave','time'])]=np.nan
            table.append(d.drop(columns='visitdate').reset_index())
        b.drop(columns='origin').to_csv(OUT/f'charls_extended_w{w0}_base.csv.gz',index_label='ID',compression={'method':'gzip','mtime':0})
        pd.concat(table).to_csv(OUT/f'charls_extended_w{w0}_visits.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    notes={'mapping':DX,'2020_disease_routing':aud,'n_matched_previous_wave_recall_coding':checks,
           'n_original_baseline':len(base1),'n_2015_blood_dated_and_positive_follow':len(base3),
           'raw_2020_no_objective_biomarker_file':True,'sources':META}
    (OUT/'charls_extended_audit.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False),encoding='utf8')
if __name__=='__main__':
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    main()

