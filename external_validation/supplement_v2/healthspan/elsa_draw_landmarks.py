"""Independent W4 and W6 blood-draw landmarks in returning ELSA participants."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from statsmodels.stats.multitest import multipletests
from prepare_healthspan import OUT,DATA,META,ADL,IADL,date,read
from repeat_biomarkers import elsa,scores,verify
from prepare_healthspan import EXT
def main():
    bundle=verify()
    ref=json.loads((EXT/'nhanes3_organ'/'organ_score_reference.json').read_text(encoding='utf8'))
    fields=['iwstat','iwindm','iwindy','tr20','verbf','alzhe','demene',*ADL,*IADL]
    path=next(DATA.rglob('h_elsa_g3.dta'))
    wide=read(path,['idauniq','ragender','raeducl']+[f'r{w}{i}' for w in (4,5,6,7) for i in fields])
    wide.index=wide.idauniq.astype('int64').astype(str)
    assert wide.index.is_unique
    rows=[]; audit=[]
    for w in (4,6):
        nurse=elsa(w)
        labs=['cholesterol','hdl','triglycerides','glucose','hba1c','hgb','crp','wbc','mch']
        nurse=nurse[nurse[labs].notna().any(axis=1)&nurse.origin.notna()&nurse.age.ge(20)]
        raw=scores(nurse,bundle['features'])
        wide1=wide.reindex(nurse.index)
        age=(nurse.age-60)/10
        stat=pd.DataFrame({'age':age,'age2':age**2,'sex':wide1.ragender.map({1:1,2:0}),'edu':wide1.raeducl,
                           'full_z':raw.full_accel/ref['panels']['full']['accel_sd_years']})
        assert wide1.idauniq.notna().all()
        bdate=date(wide1[f'r{w}iwindy'],wide1[f'r{w}iwindm'])
        fdate=date(wide1[f'r{w+1}iwindy'],wide1[f'r{w+1}iwindm'])
        # Month resolution; baseline interview must be at or before nurse sampling.
        atbaseline=wide1[f'r{w}iwstat'].eq(1)&bdate.le(nurse.origin)
        returns=atbaseline & wide1[f'r{w+1}iwstat'].eq(1)&fdate.gt(nurse.origin)
        dt=(fdate-nurse.origin)/pd.Timedelta(days=365.25)
        stat['years']=dt
        audit.append(dict(baseline_wave=w,n_blood=len(nurse),time_consistent_baseline=int(atbaseline.sum()),
                          observed_next=int(returns.sum()),median_time_years=dt[returns].median()))
        for ep in ['adl','iadl','reported_dementia','recall','fluency']:
            continuous=ep in ('recall','fluency')
            if continuous:
                field='tr20' if ep=='recall' else 'verbf'
                if any(f'r{k}{field}' not in wide1 for k in (w,w+1)):
                    audit.append(dict(baseline_wave=w,endpoint=ep,reason='required harmonized test absent in this visit pair'))
                    continue
                y0=wide1[f'r{w}{field}'].where(lambda v:v.between(0,20) if ep=='recall' else v.ge(0))
                y=wide1[f'r{w+1}{field}'].where(lambda v:v.between(0,20) if ep=='recall' else v.ge(0))
                model=stat.assign(prior=y0,y=y).loc[returns].dropna()
            else:
                def measured(wave):
                    names={'adl':ADL,'iadl':IADL,'reported_dementia':['alzhe','demene']}[ep]
                    v=wide1[[f'r{wave}{s}' for s in names]]
                    return v.max(axis=1).where(v.eq(1).any(axis=1)|v.isin([0,1]).all(axis=1))
                prior,y=measured(w),measured(w+1)
                model=stat.assign(y=y).loc[prior.eq(0)&returns].dropna()
            X=sm.add_constant(pd.get_dummies(model.drop(columns='y'),columns=['edu'],drop_first=True,dtype=float)).astype(float)
            model['y']=pd.to_numeric(model.y).astype(float)
            mod=(sm.OLS(model.y,X).fit(cov_type='HC3') if continuous else sm.GLM(model.y,X,family=sm.families.Binomial()).fit(cov_type='HC0'))
            b=mod.params.full_z
            lo,hi=mod.conf_int().loc['full_z']
            trans=(lambda z:z) if continuous else np.exp
            rows.append(dict(baseline_wave=w,next_wave=w+1,endpoint=ep,n=len(model),events=int(model.y.sum()) if not continuous else np.nan,
                             effect_type='next score points per NHANES SD, conditional on baseline' if continuous else 'OR per NHANES SD',
                             effect=trans(b),lo=trans(lo),hi=trans(hi),p=mod.pvalues.full_z))
    res=pd.DataFrame(rows);res['q']=multipletests(res.p,method='fdr_bh')[1]
    res.to_csv(OUT/'elsa_additional_draws.csv',index=False)
    (OUT/'elsa_additional_draws_audit.json').write_text(json.dumps({'rosters':audit,'sources':META},indent=2,ensure_ascii=False),encoding='utf8')
if __name__=='__main__':
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    main()

