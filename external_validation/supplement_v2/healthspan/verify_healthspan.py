"""Recheck cause-specific ADL rosters and estimates using Python independently of R."""
import json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from prepare_healthspan import OUT
def main():
    rows=[]
    formal=pd.read_csv(OUT/'out.csv').query("endpoint=='adl'").set_index('cohort')
    for c,r in formal.iterrows():
        b=pd.read_csv(OUT/f'{c}_baseline.csv.gz',dtype={'ID':str}).set_index('ID')
        l=pd.read_csv(OUT/f'{c}_visits.csv.gz',dtype={'ID':str})
        baseline=l[l.wave.eq(l.wave.min())].set_index('ID').adl
        d=b[baseline.reindex(b.index).eq(0)].copy()
        obs=l[l.time.gt(0)&l.adl.notna()&l.ID.isin(d.index)].join(d[['death','death_upper','horizon']],on='ID')
        obs=obs[obs.time.le(obs.horizon)&(obs.death.eq(0)|obs.time.le(obs.death_upper+.003))]
        pos=obs[obs.adl.eq(1)].groupby('ID').time.min()
        last=obs.groupby('ID').time.max()
        d['pos']=pos;d['last']=last
        comp=d.death.eq(1)&d.death_or_censor.le(d.horizon)&d.pos.isna()
        d['t']=d.pos.fillna(d['last']).where(~comp,d.death_or_censor)
        d['y']=d.pos.notna()
        d=d.dropna(subset=['age','sex','education','fi0','full_z','ridge_lp','t'])
        d=d[d.t.gt(0)]
        assert len(d)==r.n and d.y.sum()==r.events
        x=d[['sex','education','full_z','t','y']].copy();x['age10']=(d.age-60)/10;x['a2']=x.age10**2
        x=pd.get_dummies(x,columns=['education'],drop_first=True,dtype=float)
        cox=CoxPHFitter().fit(x,'t','y',show_progress=False)
        estimate=np.exp(cox.params_.full_z)
        np.testing.assert_allclose(estimate,r.demog_est,rtol=0,atol=1e-4)
        rows.append(dict(cohort=c,n=len(d),events=int(d.y.sum()),R=r.demog_est,lifelines=estimate))
    bio=pd.read_csv(OUT/'biological_age_comparisons.csv')
    assert bio.groupby(['cohort','endpoint']).n.nunique().eq(1).all()
    assert bio.groupby(['cohort','endpoint']).method.nunique().eq(5).all()
    audits=[]
    extended=pd.read_csv(OUT/'charls_extended_events.csv')
    old=pd.read_csv(OUT/'extended_landmarks.csv').set_index('endpoint')
    for w in [1,3]:
        b=pd.read_csv(OUT/f'charls_extended_w{w}_base.csv.gz',dtype={'ID':str}).set_index('ID')
        l=pd.read_csv(OUT/f'charls_extended_w{w}_visits.csv.gz',dtype={'ID':str})
        if w==1:b=b.dropna(subset='fi0')
        for ep in ['adl','iadl','new_chronic','multimorbidity']:
            base=l[l.wave.eq(w)].set_index('ID')
            d=b[base[ep].reindex(b.index).eq(0)].copy()
            v=l[l.ID.isin(d.index)&l.wave.gt(w)&l.time.gt(0)&l[ep].notna()].join(d[['death','death_upper','horizon']],on='ID')
            v=v[v.time.le(v.horizon)&(v.death.eq(0)|v.time.le(v.death_upper+.003))]
            event=v[v[ep].eq(1)].groupby('ID').time.min().reindex(d.index)
            last=v.groupby('ID').time.max().reindex(d.index)
            comp=d.death.eq(1)&d.death_or_censor.le(d.horizon)&event.isna()
            d['t']=event.fillna(last).where(~comp,d.death_or_censor)
            d['y']=event.notna()
            d=d[d.t.gt(0)].dropna(subset=['sex','education','age','full_z'])
            r=extended[extended.baseline.eq(w)&extended.endpoint.eq(ep)].iloc[0]
            assert len(d)==r.n and d.y.sum()==r.events and comp.reindex(d.index).sum()==r.competing_deaths
            x=d[['sex','education','full_z','t','y']].copy();x['age10']=(d.age-60)/10;x['a2']=x.age10**2
            x=pd.get_dummies(x,columns=['education'],drop_first=True,dtype=float)
            hr=np.exp(CoxPHFitter().fit(x,'t','y').params_.full_z)
            np.testing.assert_allclose(hr,r.hr,rtol=0,atol=2e-5)
            if w==1 and ep in ['adl','iadl']:np.testing.assert_allclose(r.hr,old.loc[ep+'_through_2020','hr'],atol=1e-12)
            audits.append(dict(start=w,outcome=ep,n=len(d),events=int(d.y.sum()),R=r.hr,lifelines=hr))
    (OUT/'verification.json').write_text(json.dumps({'independent_adl_checks':rows,'same_people_M7':True,'extended_charls':audits},indent=2),encoding='utf8')
if __name__=='__main__':main()

