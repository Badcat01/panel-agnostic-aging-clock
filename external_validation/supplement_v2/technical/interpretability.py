"""Frozen model group Shapley, marker integrated gradients and response profiles."""
import itertools
import math
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from shared import Ensemble,HERE,formal,get_cohorts,prepare_features

def main():
    net=Ensemble()
    dev,test,_=get_cohorts()
    b=net.bundle; features=b["features"]; groups=list(b["groups"])
    rng=np.random.default_rng(20261121); rows=[]; paths=[]; curves=[]
    def scores(x,m,a):
        with torch.no_grad():return torch.stack([model(x,m,a) for model in net.models]).mean(0)
    def tensor(x):return torch.as_tensor(x,device=net.device,dtype=torch.float32)
    dev_quant={f:dev[f].dropna().quantile([.1,.25,.5,.75,.9]).to_numpy() for f in features}
    for cohort in ["temporal","nhanes3","hrs","elsa","charls","mimic","share","klosa","lasi"]:
        raw=test if cohort=="temporal" else formal(cohort)[0]
        raw=raw.loc[raw.reindex(columns=features).notna().any(axis=1)]
        for band,eligible in raw.groupby(pd.cut(raw.age,[19,49,64,79,np.inf],labels=["20–49","50–64","65–79","80+"],right=True),observed=True):
            # Fixed age-stratified diagnostic sample, independent of outcomes.
            subset=eligible.iloc[rng.choice(len(eligible),min(96,len(eligible)),replace=False)].copy()
            xx,obs,aa=prepare_features(subset,b)
            x,m,a=map(tensor,(xx,obs,aa))
            positions=[np.isin(features,b["groups"][p]) for p in groups]
            states={}
            for k in range(6):
                for subset_idx in itertools.combinations(range(5),k):
                    keep=np.any(np.stack([positions[j] for j in subset_idx]),axis=0) if k else np.zeros(len(features),bool)
                    states[frozenset(subset_idx)]=scores(x,m*tensor(keep),a).cpu().numpy()
            shap=[]
            for j,p in enumerate(groups):
                value=sum(math.factorial(len(key))*math.factorial(4-len(key))/math.factorial(5)*
                          (states[key|{j}]-u) for key,u in states.items() if j not in key)
                shap.append(value)
                rows.append(dict(cohort=cohort,age_group=band,panel=p,n=len(subset),
                                 n_panel_observed=int(obs[:,positions[j]].any(1).sum()),mean=value.mean(),mean_abs=np.abs(value).mean()))
            np.testing.assert_allclose(np.sum(shap,axis=0),states[frozenset(range(5))]-states[frozenset()],atol=5e-6)
            # Attribution relative to development-mean values, holding age and missingness fixed.
            target=(scores(x,m,a)-scores(torch.zeros_like(x),m,a)).cpu().numpy()
            ig=np.zeros_like(xx)
            needs=np.arange(len(xx))
            for nsteps in [32,128,512]:
                if not len(needs): break
                xs,ms,ages=x[needs],m[needs],a[needs]
                grad=torch.zeros_like(xs)
                # Integrate over a short batch of path points to avoid tiny GPU launches.
                chunk=max(1,min(nsteps,512//len(needs)))
                for start in range(0,nsteps,chunk):
                    vals=(torch.arange(start,min(start+chunk,nsteps),device=net.device)+0.5)/nsteps
                    point=(xs[None]*vals[:,None,None]).reshape(-1,xs.shape[1]).detach().requires_grad_(True)
                    with torch.enable_grad():
                        for model in net.models:
                            g=torch.autograd.grad(model(point,ms.repeat(len(vals),1),ages.repeat(len(vals))).sum(),point)[0]
                            grad+=g.detach().reshape(len(vals),*xs.shape).sum(0)/(nsteps*len(net.models))
                ig[needs]=(grad*xs).cpu().numpy()
                residual=ig.sum(axis=1)-target
                needs=np.flatnonzero(np.abs(residual)>.008)
            for j,f in enumerate(features):
                keep=obs[:,j]
                if not keep.any():continue
                paths.append(dict(cohort=cohort,age_group=band,feature=f,n=int(keep.sum()),mean_IG=ig[keep,j].mean(),
                                  mean_abs_IG=np.abs(ig[keep,j]).mean(),group_max_completeness_error=np.max(np.abs(residual))))
                q=dev_quant[f]
                for qlev,val in zip([.1,.25,.5,.75,.9],q):
                    change=x[keep].clone()
                    transformed=np.log(max(val,.001)) if f in b["log_features"] else val
                    change[:,j]=float(np.clip((transformed-b["mean"][j])/b["std"][j],-5,5))
                    diff=scores(change,m[keep],a[keep]).cpu().numpy()-states[frozenset(range(5))][keep]
                    curves.append(dict(cohort=cohort,age_group=band,feature=f,n=int(keep.sum()),dev_quantile=qlev,
                                       replacement_value=val,mean_LP_change=diff.mean()))
        pd.DataFrame(rows).to_csv(HERE/"organ_shapley_by_age.csv",index=False)
        pd.DataFrame(paths).to_csv(HERE/"marker_integrated_gradients.csv",index=False)
        pd.DataFrame(curves).to_csv(HERE/"marker_response_profiles.csv",index=False)
        print("attribution",cohort,flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
