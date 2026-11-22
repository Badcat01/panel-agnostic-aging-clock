"""Exploratory next-test selection and partial-to-full score uncertainty."""
import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.neighbors import NearestNeighbors
from shared import HERE,Ensemble,get_cohorts,prepare_features

def main():
    dev,test,_=get_cohorts();net=Ensemble();b=net.bundle;features=b["features"]
    # Uniformly defined, routinely assayed targets, with a shared observable
    # counterfactual for every candidate and participant in the comparison.
    start=["sbp","glucose","creatinine","hgb","bmi"]
    candidates=["albumin","rdw","bun","platelets","sodium","mcv","ast","cholesterol"]
    common=start+candidates
    rng=np.random.default_rng(20261121)
    dev=dev.dropna(subset=common).reset_index(drop=True);test=test.dropna(subset=common).reset_index(drop=True)
    cache=[]
    for d in [dev,test]:
        full=net.score(d); initial=net.score(d[["age",*start]])
        augmented=np.array([net.score(d[["age",*start,f]]) for f in candidates]).T
        x,_,a=prepare_features(d,b); design=np.column_stack([a,d.male,x[:,[features.index(f) for f in start]]])
        cache.append((full,initial,augmented,design))
    df,di,da,dd=cache[0];tf,ti,ta,td=cache[1]
    gain=(df-di)[:,None]**2-(df[:,None]-da)**2
    indices=NearestNeighbors(n_neighbors=100,n_jobs=2).fit(dd).kneighbors(td,return_distance=False)
    personal=np.argmax(gain[indices].mean(1),axis=1)
    fixed=np.argmax(gain.mean(axis=0))
    rows=[]
    for mode,score in [("start",ti),("personalized",ta[np.arange(len(test)),personal]),
                       ("best_fixed_dev_"+candidates[fixed],ta[:,fixed]),
                       ("random",ta[np.arange(len(test)),rng.integers(0,len(candidates),len(test))]),
                       ("all_observed",tf)]:
        rows.append(dict(strategy=mode,n=len(test),events=int(test.event.sum()),development_n=len(dev),
                         score_MSE=((score-tf)**2).mean(),harrell_C=concordance_index(test.time_yr,-score,test.event)))
    pd.DataFrame(rows).to_csv(HERE/"next_test_value.csv",index=False)
    pd.Series(np.asarray(candidates)[personal]).value_counts().to_csv(HERE/"next_test_frequencies.csv")

    # Coverage target is the observable full-panel frozen LP, not true organ age.
    # Calibration and validation are time-separated within the independent temporal cohort.
    _,hold,_=get_cohorts();x,m,a=prepare_features(hold,b); full=net.score(hold)
    curves=[];cal=hold.cycle.le(2011).to_numpy();val=~cal
    for n in [1,3,6,12,20,30,43]:
        mask=m.copy()
        for i,row in enumerate(m):
            pos=np.flatnonzero(row)
            if len(pos)>n: mask[i,rng.choice(pos,len(pos)-n,replace=False)]=False
        pred=[]
        with torch.inference_mode():
            for lo in range(0,len(hold),256):
                inds=slice(lo,lo+256)
                args=[torch.as_tensor(v[inds],device=net.device,dtype=torch.float32) for v in (x,mask,a)]
                pred.append(torch.stack([model(*args) for model in net.models]).cpu().numpy().T)
        p=np.concatenate(pred);lp=p.mean(1);unc=p.std(1,ddof=1)
        err=np.abs(lp-full)
        # Split conformal quantile, using only 2009-12 residuals.
        size=int(cal.sum());index=min(size,int(np.ceil(.9*(size+1))))-1
        half=np.sort(err[cal])[index]
        scale=b["calibration"]["full"]["scale_years"]
        curves.append(dict(max_markers=n,calibration_n=size,evaluation_n=int(val.sum()),median_markers=np.median(mask[val].sum(1)),
                           target="full available frozen LP and equivalent years",nominal=.90,half_width_lp=half,
                           half_width_equivalent_years=half*scale,empirical_coverage=(err[val]<=half+1e-5).mean(),
                           mean_seed_SD_lp=unc[val].mean(),mean_abs_error_lp=err[val].mean()))
    pd.DataFrame(curves).to_csv(HERE/"panel_conformal.csv",index=False)

if __name__=="__main__":main()
