"""Actual W8/9 combined nurse measurements linked to the W10 core return interview."""
import json,sys
import numpy as np
import pandas as pd
import pyreadstat as pr
import statsmodels.api as sm
import torch
from statsmodels.stats.multitest import multipletests
from prepare_healthspan import OUT,DATA,EXT,META,read,date
from repeat_biomarkers import elsa,scores,verify

def core(w):
    name=f'wave_{w}_elsa_data_eul_v'+('2' if w==8 else '1')+'.dta'
    p=next(DATA.rglob(name))
    vars_=pr.read_dta(p,metadataonly=True)[1].column_names
    names={k.lower():k for k in vars_}
    cols='idauniq iintdatm iintdaty headldr headlba headlea headlbe headlwc headlpr headlsh headlme headlmo cflisen cflisd cfani'.split()
    d=read(p,[names[k] for k in cols if k in names]);d.columns=d.columns.str.lower()
    d.index=d.idauniq.astype('int64').astype(str)
    assert d.index.is_unique
    z=pd.DataFrame({'date':date(d.iintdaty,d.iintdatm)})
    for measure,items in [('adl','dr ba ea be wc'),('iadl','pr sh me mo')]:
        it=d[['headl'+x for x in items.split()]]
        v=it.where(it.isin([0,1]))
        z[measure]=v.max(axis=1).where(v.eq(1).any(axis=1)|v.notna().all(axis=1))
    if {'cflisen','cflisd'}<=set(d):
        z['recall']=d.cflisen.where(d.cflisen.between(0,10))+d.cflisd.where(d.cflisd.between(0,10))
    else:z['recall']=np.nan
    z['fluency']=d['cfani'].where(d['cfani'].between(0,100)) if 'cfani' in d else np.nan
    return z

def main():
    bundle=verify()
    ref=json.loads((EXT/'nhanes3_organ'/'organ_score_reference.json').read_text(encoding='utf8'))['panels']['full']['accel_sd_years']
    a=pd.concat([elsa(w).assign(wave=w) for w in (8,9)]).sort_values('origin',kind='stable')
    assert a.index.duplicated().sum()==2
    a=a[~a.index.duplicated(keep='first')]
    nurse=next(DATA.rglob('elsa_nurse_w8w9_data_eul.dta'))
    markers=['hba1c','rtin','cfib']
    d=read(nurse,['idauniq','wave','w89bldwt',*markers])
    prior_w8=read(next(DATA.rglob('wave_8_elsa_nurse_data_eul_v1.dta')),['idauniq',*markers]).set_index('idauniq')
    merge_w8=d[d.wave.eq(8)].set_index('idauniq')
    assert prior_w8.index.isin(merge_w8.index).all()
    pd.testing.assert_frame_equal(prior_w8[markers],merge_w8.loc[prior_w8.index,markers],check_dtype=False)
    late_audit={'n_records':len(d),'unique_people':d.idauniq.nunique(),'n_by_wave':d.wave.value_counts().to_dict(),
                'common_with_standalone_W8':len(prior_w8),'common_with_standalone_exact_assays':markers,
                'HbA1c_raw_positive_median_by_wave':d[d.hba1c.gt(0)].groupby('wave').hba1c.median().to_dict(),
                'n_positive_blood_weights_by_wave':d[d.w89bldwt.gt(0)].wave.value_counts().to_dict()}
    d.index=d.idauniq.astype('int64').astype(str)
    weights=d.reset_index(drop=True).assign(ID=lambda v:v.idauniq.astype('int64').astype(str)).set_index(['ID','wave']).w89bldwt
    a['weight']=[weights.get((i,w)) for i,w in zip(a.index,a.wave)]
    labs='cholesterol hdl triglycerides glucose hba1c hgb crp wbc mch'.split()
    a=a[a[labs].notna().any(axis=1)&a.age.ge(20)&a.origin.notna()]
    raw=scores(a,bundle['features'])
    h=read(next(DATA.rglob('h_elsa_g3.dta')),['idauniq','ragender','raeducl','r9tr20','r9verbf','r9dressa','r9batha','r9eata','r9beda','r9toilta','r9medsa','r9moneya','r9shopa','r9mealsa'])
    h.index=h.idauniq.astype('int64').astype(str)
    z=pd.DataFrame({'age':(a.age-60)/10,'sex':h.ragender.map({1:1,2:0}).reindex(a.index),'edu':h.raeducl.reindex(a.index),
                    'wave9':a.wave.eq(9).astype(int),'full_z':raw.full_accel/ref,'w89bldwt':a.weight})
    z['age2']=z.age**2
    v8,v9,v10=core(8),core(9),core(10)
    checks={}
    for k in ('adl','iadl','recall','fluency'):
        if k in ('adl','iadl'):
            vs=h[['r9'+c for c in ('dressa batha eata beda toilta' if k=='adl' else 'moneya medsa shopa mealsa').split()]]
            y=vs.max(axis=1).where(vs.eq(1).any(axis=1)|vs.notna().all(axis=1))
        else:y=h['r9tr20' if k=='recall' else 'r9verbf']
        delta=(v9[k]-y).dropna()
        checks[k]=dict(n=len(delta),matched=int(delta.abs().lt(1e-7).sum()))
        assert len(delta)>3000 and delta.abs().lt(1e-7).mean()>.99
    prior=pd.concat([v8.reindex(a.index[a.wave.eq(8)]),v9.reindex(a.index[a.wave.eq(9)])])
    z['years']=(v10.date.reindex(z.index)-a.origin)/pd.Timedelta(days=365.25)
    z=z[prior.date.reindex(z.index).le(a.origin)&z.years.gt(0)]
    rows=[]
    for ep in ('adl','iadl','recall','fluency'):
        v=z.assign(prior=prior[ep],y=v10[ep]).dropna(subset=['age','sex','edu','full_z','prior','y'])
        cont=ep in ['recall','fluency']
        if not cont:v=v[v.prior.eq(0)]
        for label in ['unweighted','blood_weight_sensitivity']:
            d=v if label=='unweighted' else v[v.w89bldwt.gt(0)]
            cols=['age','age2','sex','edu','years','wave9','full_z']+(['prior'] if cont else [])
            X=sm.add_constant(pd.get_dummies(d[cols],columns=['edu'],drop_first=True,dtype=float)).astype(float)
            wgt=pd.Series(1.,index=d.index) if label=='unweighted' else d.w89bldwt/d.w89bldwt.mean()
            mod=sm.GLM(d.y,X,family=sm.families.Gaussian() if cont else sm.families.Binomial(),var_weights=wgt).fit(cov_type='HC0')
            b=mod.params.full_z;lo,hi=mod.conf_int().loc['full_z']
            trans=(lambda x:x) if cont else np.exp
            rows.append(dict(endpoint=ep,analysis=label,n=len(d),events=d.y.sum() if not cont else np.nan,years_median=d.years.median(),
                           estimate=trans(b),lo=trans(lo),hi=trans(hi),p=mod.pvalues.full_z))
    tab=pd.DataFrame(rows)
    tab['q']=tab.groupby('analysis').p.transform(lambda x:multipletests(x,method='fdr_bh')[1])
    tab.to_csv(OUT/'elsa_pooled_w89_to_w10.csv',index=False)
    if '--pace' in sys.argv:
        speed_rows=[]
        for first in [2,4,6]:
            pace=pd.read_csv(OUT/f'elsa_pooled_w89_pace_from_w{first}.csv.gz',dtype={'ID':str}).set_index('ID')
            for ep in ['adl','iadl']:
                sample=z.assign(prior=prior[ep],y=v10[ep]).join(pace[['baseline_ba','pace']]).loc[lambda v:v.prior.eq(0)]
                xcols=['age','age2','sex','edu','years','wave9','baseline_ba','pace']
                sample=sample.dropna(subset=['y',*xcols])
                X=sm.add_constant(pd.get_dummies(sample[xcols],columns=['edu'],drop_first=True,dtype=float)).astype(float)
                mod=sm.GLM(sample.y,X,family=sm.families.Binomial()).fit(cov_type='HC0')
                lo,hi=mod.conf_int().loc['pace']
                speed_rows.append(dict(first=first,endpoint=ep,n=len(sample),events=int(sample.y.sum()),OR=np.exp(mod.params.pace),
                                       lo=np.exp(lo),hi=np.exp(hi),p=mod.pvalues.pace))
        st=pd.DataFrame(speed_rows);st['q']=multipletests(st.p,method='fdr_bh')[1]
        st.to_csv(OUT/'elsa_pooled_w89_pace_to_w10.csv',index=False)
    (OUT/'elsa_pooled_w89_to_w10_sources.json').write_text(json.dumps(dict(recall_and_ADL_harmonized_wave9_recheck=checks,late_nurse_audit=late_audit,
         blood_dated=len(a),dates_compatible_returned=len(z),sources=META),indent=2,ensure_ascii=False),encoding='utf8')
if __name__=='__main__':
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    main()
