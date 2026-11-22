"""Validate paired organ-function measurements and longitudinal stability of frozen scores."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat as pr
import torch
import statsmodels.api as sm
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from prepare_healthspan import OUT,EXT,ROOT,DATA,META,ORGANS,date,read,sha
sys.path.insert(0,str(EXT/'scripts'))
from reference import BUNDLE, score_frame, verify

def source(c):
    p=EXT/'inputs'/f'{c}.csv.gz'
    META[str(p.relative_to(ROOT))]={'sha256':sha(p)}
    return pd.read_csv(p,dtype={'ID':str}).set_index('ID',verify_integrity=True)

def elsa(w):
    f={2:'wave_2_nurse_data_v2.dta',4:'wave_4_nurse_data.dta',6:'wave_6_elsa_nurse_data_v2.dta',
       8:'elsa_nurse_w8w9_data_eul.dta',9:'elsa_nurse_w8w9_data_eul.dta'}[w]
    file=next(DATA.rglob(f)); names=pr.read_dta(file,metadataonly=True)[1].column_names
    mapnames={x.lower():x for x in names}
    fields={'cholesterol':('chol',1/.0259),'hdl':('hdl',1/.0259),'triglycerides':('trig',1/.0113),'glucose':('fglu',1/.0555),
            'hba1c':('hba1c',1),'hgb':('hgb',1),'crp':('hscrp',.1),'bmi':('bmival',1),'waist':('wstval',1),
            'sbp':('sysval',1),'dbp':('diaval',1),'ferritin':('rtin',1),'fibrinogen':('cfib',1)}
    if w>=4:
        fields.update({'wbc':('wbc',1),'mch':('mch',1)})
    if w>=8:
        del fields['bmi'],fields['waist']
    ac='indager' if w>=6 else 'confage'
    cols=[v[0] for v in fields.values()]+['idauniq','visyear','vismon',ac,'pulse1','pulse2','pulse3']
    if w>=8:cols.append('wave')
    borig=read(file,[mapnames[x] for x in cols]); borig.columns=borig.columns.str.lower()
    if w>=8:borig=borig[borig.wave.eq(w)].copy()
    borig.index=borig.idauniq.astype('int64').astype(str)
    assert borig.index.is_unique
    b=pd.DataFrame({k:borig[s].where(borig[s].gt(0))*scale for k,(s,scale) in fields.items()},index=borig.index)
    b['pulse']=borig[['pulse1','pulse2','pulse3']].where(lambda x:x.gt(0)).mean(axis=1).where(b.sbp.notna()&b.dbp.notna())
    if w>=6: b['hba1c']=b.hba1c*.09148+2.152
    b['age']=borig[ac].replace({-7:90} if w>=6 else {99:90})
    b['origin']=date(borig.visyear,borig.vismon)
    b.loc[~b.age.between(20,110),'age']=np.nan
    return b

def charls():
    hm=next(DATA.rglob('H_CHARLS_D_Data.dta'))
    exams=[f'r3{t}{i}' for t in ('systo','diasto','pulse') for i in (1,2,3)]
    h=read(hm,['ID','ID_w1','r1iwy','r1iwm','r3iwy','r3iwm','r3agey','r3mbmi','r3mwaist']+exams).set_index('ID',verify_integrity=True)
    p=next(DATA.rglob('2011charls/Blood_20140429.dta'))
    b=read(p,['ID','cystatinc']).rename(columns={'ID':'ID_w1'}).set_index('ID_w1',verify_integrity=True)
    c1=source('charls').join(date(h.r1iwy,h.r1iwm).rename('origin'))
    id1=h.ID_w1[h.ID_w1.ne('')].to_frame().join(b,on='ID_w1',validate='one_to_one')
    c1['cystatin']=id1.cystatinc
    cols={'wbc':'bl_wbc','hgb':'bl_hgb','mcv':'bl_mcv','platelets':'bl_plt','triglycerides':'bl_tg','creatinine':'bl_crea',
          'bun':'bl_bun','hdl':'bl_hdl','cholesterol':'bl_cho','glucose':'bl_glu','uric_acid':'bl_ua','hba1c':'bl_hbalc','crp':'bl_crp','cystatin':'bl_cysc'}
    p=next(DATA.rglob('2015charls/Blood.dta'))
    d=read(p,['ID',*cols.values()]).set_index('ID',verify_integrity=True)
    d=d.rename(columns={s:k for k,s in cols.items()})
    d['crp']/=10
    d['age']=h.r3agey
    d['origin']=date(h.r3iwy,h.r3iwm)
    d['bmi']=h.r3mbmi.where(h.r3mbmi.between(8,100));d['waist']=h.r3mwaist.where(h.r3mwaist.between(35,200))
    for k,s in [('sbp','systo'),('dbp','diasto'),('pulse','pulse')]:
        val=h[[f'r3{s}{i}' for i in (1,2,3)]]
        d[k]=val.where(val.gt(0)).mean(axis=1)
    return c1,d

def egfr(cys,age,sex):
    z=cys.where(cys.gt(0))/.8
    return 133*z.clip(upper=1)**(-.499)*z.clip(lower=1)**(-1.328)*.996**age*np.where(sex.eq(0),.932,1)

def diagonal(b, outcome, group, cohort, name, extra=None, directional=True):
    cols=[o for o in ORGANS if b[o].notna().any()]
    d=b[['age','sex','full_z',*cols]].join(outcome.rename('response'))
    if extra is not None: d=d.join(extra)
    d=d.dropna()
    # Positive target direction is fixed: worse organ score implies higher marker
    # or a lower eGFR; eGFR is negated by caller.
    y=rankdata(d.response)
    Z=np.column_stack([rankdata(d[k]) for k in cols])
    cov=np.column_stack([np.ones(len(d)),d.age,d.age**2,d.sex,d.full_z]+([] if extra is None else [rankdata(d[k]) for k in extra.columns]))
    q,_=np.linalg.qr(cov,mode='reduced')
    res=lambda x:x-q@(q.T@x)
    Z=res(Z); Z/=np.sqrt((Z**2).mean(axis=0))
    y=res(y); y/=np.sqrt((y*y).mean())
    rho=(Z*y[:,None]).mean(axis=0)
    infl=Z*y[:,None]-.5*(Z**2+y[:,None]**2)*rho
    se=infl.std(axis=0,ddof=1)/np.sqrt(len(d))
    target=cols.index(group)
    if not directional:
        return [dict(cohort=cohort,marker=name,n=len(d),organ=o,match=(o==group),rho=rho[i],lo=rho[i]-1.96*se[i],
                     hi=rho[i]+1.96*se[i],target_minus_this=np.nan,one_sided_p_vs_this=np.nan,
                     match_best=np.nan,matched_stronger_than_all_p=np.nan) for i,o in enumerate(cols)]
    rng=np.random.default_rng(20261121)
    # A paired Rademacher multiplier test of each one-sided null that the target
    # partial rank correlation is no stronger than the competing correlation.
    delta=rho[target]-rho
    delta_phi=infl[:,[target]]-infl
    hit=np.zeros(len(cols))
    for _ in range(20):
        signs=rng.integers(0,2,size=(100,len(d)),dtype=np.int8)*2-1
        null=(signs@delta_phi)/len(d)
        hit+=(null>=delta).sum(axis=0)
    p=(hit+1)/2001
    maximum=max(p[i] for i in range(len(cols)) if i!=target)
    return [dict(cohort=cohort,marker=name,n=len(d),organ=o,match=(o==group),rho=rho[i],lo=rho[i]-1.96*se[i],
                 hi=rho[i]+1.96*se[i],target_minus_this=delta[i],one_sided_p_vs_this=p[i] if i!=target else np.nan,
                 match_best=rho[target]==max(rho),matched_stronger_than_all_p=maximum) for i,o in enumerate(cols)]

def icc(x,y):
    a=np.stack([x,y],axis=1); n=len(a)
    people=a.mean(axis=1); occasion=a.mean(axis=0); center=a.mean()
    mr=2*((people-center)**2).sum()/(n-1)
    mc=n*((occasion-center)**2).sum()
    me=((a-people[:,None]-occasion[None,:]+center)**2).sum()/(n-1)
    return (mr-me)/(mr+me+2*(mc-me)/n)

def scores(df,features):
    d=df.reindex(columns=['age',*features]).rename_axis('ID').reset_index()
    return score_frame(d,BUNDLE,'cpu').set_index('ID')

def repeat(c,w,first,later,features,frozen_baseline=True):
    ids=first.index.intersection(later.index)
    if frozen_baseline:ids=ids.intersection(source(c).index)
    labs=list(set(features)&set(['bun','creatinine','hba1c','hgb','hscrp','crp','glucose','wbc','hdl','cholesterol','platelets','mcv','triglycerides','uric_acid']))
    a,b=first.loc[ids],later.loc[ids]
    shared=a.reindex(columns=features).notna() & b.reindex(columns=features).notna()
    dt=(b.origin-a.origin)/pd.Timedelta(days=365.25)
    keep=shared[labs].any(axis=1)&dt.gt(0)&a.age.ge(20)&b.age.ge(20)
    a,b,dt,shared=a.loc[keep],b.loc[keep],dt[keep],shared.loc[keep]
    rows=[]; subject=pd.DataFrame({'time_yr':dt,'landmark_age':b.age},index=a.index)
    for mode in ['as_measured','same_markers']:
        if mode=='as_measured':
            x=scores(a,features); y=scores(b,features)
            if frozen_baseline:
                raw=pd.read_csv(EXT/'results'/c/'predictions.csv.gz',dtype={'ID':str}).set_index('ID').loc[a.index]
                np.testing.assert_allclose(x.full_lp,raw.full_lp,rtol=1e-5,atol=3e-5)
        else:
            x=scores(a.reindex(columns=features).where(shared).assign(age=a.age),features)
            y=scores(b.reindex(columns=features).where(shared).assign(age=b.age),features)
        for group in ['full',*ORGANS]:
            for value in ['ba','accel']:
                key=group+'_'+value
                ok=x[key].notna()&y[key].notna()
                if ok.sum()<10:continue
                v1=x.loc[ok,key].to_numpy();v2=y.loc[ok,key].to_numpy()
                change=(v2-v1)/dt[ok].to_numpy()
                rows.append({'cohort':c,'follow_wave':w,'panel_mode':mode,'group':group,'type':value,
                             'n':ok.sum(),'median_follow_yr':dt[ok].median(),'yearly_mean':change.mean(),'yearly_median':np.median(change),
                             'yearly_p25':np.quantile(change,.25),'yearly_p75':np.quantile(change,.75),'ICC_A1':icc(v1,v2),
                             'spearman':pd.Series(v1).corr(pd.Series(v2),method='spearman')})
                if (group,value,mode)==('full','ba','same_markers'):
                    subject['pace']=change;subject['baseline_ba']=v1
    return rows,subject

def main():
    bundle=verify(); torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    rows=[];diags=[];goals=[];pheno_coverage=[]
    pairs=charls(); stages={'charls':{1:pairs[0],3:pairs[1]},'elsa':{n:elsa(n) for n in (2,4,6)}}
    for c,ww in stages.items():
        base=pd.read_csv(OUT/f'{c}_baseline.csv.gz',dtype={'ID':str}).set_index('ID')
        v0=min(ww);a=ww[v0]
        if c=='charls':
            diags+=diagonal(base,a.cystatin,'kidney',c,'cystatin_at_baseline')
        else:
            for mark in ['ferritin','fibrinogen']:
                diags+=diagonal(base,a[mark],'immune_heme',c,mark+'_at_baseline')
        for w,b in ww.items():
            required='albumin creatinine glucose crp lymph_pct mcv rdw alp wbc'.split()
            source_values=b.reindex(columns=required)
            pheno_coverage.append({'cohort':c,'wave':w,'source_n':len(b),
                'required_n':len(required),'missing_measures':','.join(source_values.columns[source_values.isna().all()]),
                'any_whole_formula_n':int(source_values.notna().all(axis=1).sum())})
            if w==v0:continue
            dt=((b.origin-a.origin)/pd.Timedelta(days=365.25)).rename('dt')
            if c=='charls':
                g1=egfr(a.cystatin,a.age,base.sex); g2=egfr(b.cystatin,b.age,base.sex.reindex(b.index))
                for mark,x,y,group in [('negative_egfr_cys',-g1,-g2,'kidney'),('hba1c',a.hba1c,b.hba1c,'metabolic')]:
                    diags+=diagonal(base,y,group,c,mark+'_followup_adj_baseline',pd.concat([dt,x.rename('prior')],axis=1))
                    both=pd.concat([x,y,dt],axis=1,keys=['v1','v2','dt']).join(base[['age','sex','full_z',group]],how='inner').dropna()
                    change=both.v2-both.v1
                    goals.append(dict(cohort=c,marker=mark,n=len(both),baseline_median=both.v1.median(),followup_median=both.v2.median(),
                                  paired_median_change=change.median(),median_years=both.dt.median()))
            r,pat=repeat(c,w,a,b,bundle['features']);rows.extend(r)
            pat.to_csv(OUT/f'{c}_{w}_pace.csv.gz',index_label='ID',compression={'method':'gzip','mtime':0})
            print(c,w,len(pat),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'repeat_stability.csv',index=False)
    pd.DataFrame(goals).to_csv(OUT/'organ_change.csv',index=False)
    pd.DataFrame(pheno_coverage).to_csv(OUT/'phenoage_repeat_coverage.csv',index=False)
    matrix=pd.DataFrame(diags)
    sub=matrix.loc[matrix.match].copy()
    sub['diagonal_q']=multipletests(sub.matched_stronger_than_all_p,method='fdr_bh')[1]
    matrix=matrix.merge(sub[['cohort','marker','diagonal_q']],on=['cohort','marker'],validate='many_to_one')
    matrix.to_csv(OUT/'biomarker_diagonal.csv',index=False)
    (OUT/'repeat_sources.json').write_text(json.dumps(META,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=3):
        main()

