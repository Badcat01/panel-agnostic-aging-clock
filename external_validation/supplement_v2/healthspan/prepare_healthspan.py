"""Build healthspan rosters using frozen patient-level predictions and dated interviews."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import pyreadstat as pr

OUT=Path(__file__).resolve().parent
EXT=OUT.parents[1]
ROOT=EXT.parent
sys.path.insert(0,str(EXT/"scripts"))
from study_paths import COMMUNITY_ROOT as DATA
from build_hrs_elsa import month_start


COHORTS={
    "hrs":("randhrs1992_2020v2.dta","hhidpn","raeduc",13,15),
    "elsa":("h_elsa_g3.dta","idauniq","raeducl",2,6),
    "charls":("H_CHARLS_D_Data.dta","ID","raeducl",1,4),
}
ADL="dressa batha eata beda toilta".split()
IADL="moneya medsa shopa mealsa".split()
DISEASES="hibpe diabe cancre lunge hearte stroke arthre psyche".split()
ORGANS=["liver","kidney","metabolic","immune_heme","cardiovascular"]
META={}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(2**22),b''): h.update(b)
    return h.hexdigest()

def read(p,cols):
    d,m=pr.read_dta(p,usecols=list(dict.fromkeys(cols)))
    META[str(p.relative_to(DATA))]={"sha256":sha(p),"columns":{n:m.column_names_to_labels.get(n) for n in cols}}
    return d

def date(year,month):
    return month_start(year,month)+pd.Timedelta(days=14)

def index_date(c,wide):
    if c=="charls": return date(wide.r1iwy,wide.r1iwm)
    if c=="elsa":
        p=next(DATA.rglob("wave_2_nurse_data_v2.dta"))
        d=read(p,["idauniq","visyear","vismon"])
        d.index=d.idauniq.astype('int64').astype(str)
        return date(d.visyear,d.vismon).reindex(wide.index)
    p=next(DATA.rglob('hrs2016vbs.sav'))
    v,m=pr.read_sav(p,usecols=['HHID','PN','PVBS_N_DAYS'])
    META[str(p.relative_to(DATA))]={'sha256':sha(p),'columns':m.column_names_to_labels}
    d=read(next(DATA.rglob("trk2020tr_r.dta")),["HHID","PN","PIWMONTH","PIWYEAR"])
    v=v.merge(d,how='left',on=['HHID','PN'],validate='one_to_one')
    v.index=v.HHID.str.strip()+"-"+v.PN.str.strip()
    return (date(v.PIWYEAR,v.PIWMONTH)+pd.to_timedelta(v.PVBS_N_DAYS,unit='D')).reindex(wide.index)

def make(c):
    name,idcol,educ,w0,end=COHORTS[c]
    p=next(DATA.rglob(name))
    available=set(pr.read_dta(p,metadataonly=True)[1].column_names)
    move=["walk1a" if c=="hrs" else "walk100a","chaira","climsa","stoopa","lifta","armsa","dimea"]
    mood=["depres","effort","sleepr","flone","going"]
    if c=="charls": mood=[m+"l" for m in mood]
    fi_names=ADL+IADL+move+DISEASES+["shlt"]+mood # 30 items
    assert len(fi_names)==30
    dx=DISEASES+(["dyslipe","livere","kidneye","digeste","asthmae","memrye"] if c=="charls" else [])
    if c=="hrs": datefields=["iwendy","iwendm"]
    elif c=="elsa": datefields=["iwindy","iwindm"]
    else: datefields=["iwy","iwm"]
    stubs=set(fi_names+dx+datefields+"iwstat tr20 tr20p proxy demene alzhe cog27 grp gripsum timwlk wspeed cesd10 verbf".split())
    cols=[idcol,educ]+[f"r{w}{v}" for w in range(w0,end+1) for v in sorted(stubs) if f"r{w}{v}" in available]
    wide=read(p,cols)
    if c=="hrs":
        ids=wide[idcol].astype('int64').astype(str).str.zfill(9)
        wide.index=ids.str[:-3]+'-'+ids.str[-3:]
    elif c=="elsa": wide.index=wide[idcol].astype('int64').astype(str)
    else: wide.index=wide[idcol]
    assert wide.index.is_unique
    scores_path=EXT/"results"/c/"predictions.csv.gz"
    META[str(scores_path.relative_to(ROOT))]={'sha256':sha(scores_path)}
    scores=pd.read_csv(scores_path,dtype={'ID':str}).set_index('ID',verify_integrity=True)
    assert scores.index.isin(wide.index).all()
    wide=wide.loc[scores.index]
    base_date=index_date(c,wide)
    assert base_date.notna().all()
    ref=json.loads((EXT/'nhanes3_organ/organ_score_reference.json').read_text(encoding='utf8'))['panels']
    common=scores[['age','sex','ridge_lp','event','time_yr','time_upper_yr']].rename(columns={'event':'death','time_yr':'death_or_censor','time_upper_yr':'death_upper'}).copy()
    common['full_z']=scores.full_accel/ref['full']['accel_sd_years']
    for o in ORGANS: common[o]=scores[o+'_accel']/ref[o]['accel_sd_years']
    common['education']=wide[educ]
    if c=='hrs': common['education']=common.education.map({1:1,2:2,3:2,4:3,5:3})
    common['origin']=base_date
    waves=[]
    for w in range(w0,end+1):
        def v(s):
            k=f'r{w}{s}'; return wide[k] if k in wide else pd.Series(np.nan,index=wide.index)
        def item(s):
            a=v(s)
            if s in ADL+IADL+move: return a.map({0:0,1:1,2:1} if c=='hrs' else {0:0,1:1})
            if s=='shlt': return (a-1).div(4).where(a.between(1,5))
            if s in mood and c=='charls': return (a-1).div(3).where(a.between(1,4))
            return a.where(a.isin([0,1]))
        def any_response(vals):
            return vals.max(axis=1).where(vals.eq(1).any(axis=1)|vals.notna().all(axis=1))
        fields=pd.DataFrame({n:item(n) for n in fi_names},index=wide.index)
        datew=date(v(datefields[0]),v(datefields[1]))
        time=(datew-base_date)/pd.Timedelta(days=365.25)
        source_ok=v('iwstat').isin([1,2] if c=='hrs' else [1]) & datew.notna()
        if w==w0: time[:]=0
        d=pd.DataFrame({'wave':w,'time':time,'adl':any_response(fields[ADL]),'iadl':any_response(fields[IADL])})
        d['fi']=fields.mean(axis=1).where(fields.notna().sum(axis=1)>=24)
        non_adl_iadl=fields.drop(columns=ADL+IADL)
        d['fi_without_adl_iadl']=non_adl_iadl.mean(axis=1).where(non_adl_iadl.notna().sum(axis=1)>=17)
        d['frailty']=(d.fi>=.25).astype(float).where(d.fi.notna())
        for s in dx: d[s]=item(s)
        d['dxcount']=d[dx].sum(axis=1,min_count=len(dx))
        d['recall']=v('tr20p' if c=='hrs' and w>w0 else 'tr20')
        d['recall']=d.recall.where(d.recall.between(0,20))
        if c=='hrs':
            d['lw_self_dementia']=(v('cog27')<=6).astype(float).where(v('proxy').eq(0)&v('cog27').between(0,27))
        if c!='charls':
            d['reported_dementia']=any_response(pd.DataFrame({'a':v('alzhe'),'b':v('demene')}))
        d['grip']=v('grp' if c=='hrs' else 'gripsum').where(lambda x:x.gt(0))
        d['walk_seconds']=v('timwlk' if c=='hrs' else 'wspeed').where(lambda x:x.gt(0))
        if c=='elsa': d['fluency']=v('verbf').where(lambda x:x.ge(0))
        if c=='charls': d['cesd10']=v('cesd10').where(lambda x:x.between(0,30))
        d.loc[~source_ok,d.columns.difference(['wave'])]=np.nan
        if w>w0:
            after=d.time.gt(0)
            d.loc[~after,d.columns.difference(['wave'])]=np.nan
        waves.append(d)
    base=waves[0]
    final_year = wide[f'r{end}{datefields[0]}']
    final_month = wide[f'r{end}{datefields[1]}']
    cutoff = date(final_year, final_month).where(wide[f'r{end}iwstat'].isin([1,2] if c=='hrs' else [1])).max() + pd.offsets.MonthEnd(0)
    common['horizon']=(cutoff-base_date)/pd.Timedelta(days=365.25)
    common['fi0']=base.fi
    common['fi_without_adl_iadl0']=base.fi_without_adl_iadl
    common['base_disease_count']=base.dxcount
    for d in waves:
        d['multimorbidity']=d.dxcount.ge(2).astype(float).where(d.dxcount.notna())
        d['new_chronic']=((d[dx].eq(1)&base[dx].eq(0)).any(axis=1)).astype(float)
        # A zero requires complete ascertainment of every condition still at risk.
        valid=(base[dx].eq(1)|d[dx].isin([0,1])).all(axis=1) & base[dx].notna().all(axis=1)
        d['new_chronic']=d.new_chronic.where(base[dx].notna().all(axis=1)&(d.new_chronic.eq(1)|valid))
    common.drop(columns='origin').to_csv(OUT/f'{c}_baseline.csv.gz',compression={'method':'gzip','mtime':0})
    visits=pd.concat(waves).rename_axis('ID')
    visits.to_csv(OUT/f'{c}_visits.csv.gz',compression={'method':'gzip','mtime':0})
    return {'cohort':c,'n':len(common),'waves':list(range(w0,end+1)),'fields_fi':fi_names,'fields_dx':dx,'n_baseline_fi':int(base.fi.notna().sum()),'median_baseline_fi':base.fi.median(),'visit_n':{int(w):int(b.time.notna().sum()) for w,b in visits.groupby('wave')}}

if __name__=='__main__':
    res=[make(k) for k in COHORTS]
    (OUT/'source_audit.json').write_text(json.dumps(dict(sources=META,cohorts=res),ensure_ascii=False,indent=2),encoding='utf8')
    print(res)

