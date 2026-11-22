"""All observed ELSA health interviews up to wave 9 at three blood-draw landmarks."""
import json
import numpy as np
import pandas as pd
import pyreadstat as pr
import torch
from prepare_healthspan import OUT,DATA,EXT,META,ADL,IADL,DISEASES,read,date
from repeat_biomarkers import elsa,scores,verify

def main():
    p=next(DATA.rglob('h_elsa_g3.dta'))
    have=pr.read_dta(p,metadataonly=True)[1].column_names
    fields=['iwstat','iwindy','iwindm','tr20','verbf','demene','alzhe',*ADL,*IADL,*DISEASES]
    d=read(p,['idauniq','ragender','raeducl','radyear']+[f'r{w}{k}' for w in range(2,10) for k in fields if f'r{w}{k}' in have])
    d.index=d.idauniq.astype('int64').astype(str)
    assert d.index.is_unique
    vdata={}
    for w in range(2,10):
        st=f'r{w}'
        def item(k):
            a=d.get(st+k,pd.Series(np.nan,index=d.index))
            return a.where(a.isin([0,1]))
        def any_group(group):
            v=pd.concat([item(k) for k in group],axis=1)
            return v.max(axis=1).where(v.eq(1).any(axis=1)|v.notna().all(axis=1))
        v=pd.DataFrame(dict(adl=any_group(ADL),iadl=any_group(IADL),reported_dementia=any_group(['demene','alzhe'])))
        for k in DISEASES:v[k]=item(k)
        v['recall']=d[st+'tr20'].where(lambda x:x.between(0,20))
        v['fluency']=d.get(st+'verbf',pd.Series(np.nan,index=d.index)).where(lambda x:x.ge(0))
        v['visitdate']=date(d[st+'iwindy'],d[st+'iwindm']).where(d[st+'iwstat'].eq(1))
        vdata[w]=v
    bundle=verify()
    ref=json.loads((EXT/'nhanes3_organ'/'organ_score_reference.json').read_text(encoding='utf8'))['panels']['full']['accel_sd_years']
    audit=[]
    for w0 in (2,4,6):
        a=elsa(w0)
        if w0==2:
            c=pd.read_csv(OUT/'elsa_baseline.csv.gz',dtype={'ID':str}).set_index('ID')[['age','sex','education','full_z']]
            a=a.loc[c.index]
        else:
            labs='cholesterol hdl triglycerides glucose hba1c hgb crp wbc mch'.split()
            a=a[a[labs].notna().any(axis=1)&a.age.ge(20)&a.origin.notna()]
            pred=scores(a,bundle['features'])
            c=pd.DataFrame({'age':a.age,'full_z':pred.full_accel/ref,'sex':d.ragender.map({1:1,2:0}).reindex(a.index),'education':d.raeducl.reindex(a.index)})
        c=c[vdata[w0].visitdate.reindex(c.index).le(a.origin)]
        base=vdata[w0].loc[c.index,DISEASES]
        pieces=[]
        for w in range(w0,10):
            v=vdata[w].reindex(c.index).copy()
            t=(v.visitdate-a.origin)/pd.Timedelta(days=365.25)
            if w==w0: t=pd.Series(0.,index=c.index)
            ok=v.visitdate.notna()&(w==w0 or t.gt(0))
            v=v.loc[ok]
            t=t.reindex(v.index)
            y=v[DISEASES]
            has=(y.eq(1)&base.loc[y.index].eq(0)).any(axis=1)
            v['new_chronic']=has.astype(float).where(base.loc[y.index].notna().all(axis=1)&(has|(base.loc[y.index].eq(1)|y.notna()).all(axis=1)))
            v['multimorbidity']=y.sum(axis=1,min_count=len(DISEASES)).ge(2).astype(float).where(y.notna().all(axis=1))
            z=v.drop(columns='visitdate').join(c).assign(wave=w,time=t)
            pieces.append(z.rename_axis('ID').reset_index())
        output=pd.concat(pieces,ignore_index=True)
        assert not output.duplicated(['ID','wave']).any()
        output.to_csv(OUT/f'elsa_visits_w{w0}_to_w9.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        audit.append(dict(wave=w0,n_baseline=len(c),n_observed_again=output.loc[output.wave>w0,'ID'].nunique()))
    tracking={str(w):int(d[f'r{w}iwstat'].eq(5).sum()) for w in (6,7,8,9)}
    (OUT/'elsa_followup_audit.json').write_text(json.dumps(dict(rosters=audit,deaths_flagged_this_wave=tracking,last_radyear=int(d.radyear.max()),sources=META),indent=2,ensure_ascii=False),encoding='utf8')
if __name__=='__main__':
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    main()
