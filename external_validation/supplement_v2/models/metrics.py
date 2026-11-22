"""Patient-paired Harrell C and exact 2,000 participant bootstrap replicates."""
import argparse
import hashlib
import os
import sys
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from numba import njit, prange, set_num_threads

from design import OUT, FEATS, EXT, original, panels, read
sys.path.insert(0,str(EXT/"scripts"))
from validate_external import weighted_c


@njit(parallel=True)
def pairs(t,e,ranks,reps):
    n=len(t)
    sims=np.empty((reps,2))
    for rep in prange(reps):
        np.random.seed(20261121+rep)
        w=np.bincount(np.random.randint(0,n,n),minlength=n)
        for j in range(2):
            sims[rep,j]=weighted_c(t,e,ranks[j],w)
    return sims


def summarize(p,full_inference):
    d=pd.read_csv(p,dtype={"ID":str},float_precision="round_trip")
    kind,panel=p.name.split(".")[0].split("_")
    cache=OUT/"metrics_cache"
    cache.mkdir(exist_ok=True)
    output=[]
    for name in d.columns[4:]:
        if name=="M0":
            continue
        mask=np.isfinite(d[name])&np.isfinite(d.M0)
        s=d.loc[mask,["ID","event","time_yr","M0",name,"n_observed"]]
        if len(s)==0 or s.event.sum()<10:
            continue
        key=hashlib.sha256(pd.util.hash_pandas_object(s,index=False).values.tobytes()).hexdigest()
        dest=cache/f"{kind}_{panel}_{name}.json"
        prior=read(dest) if dest.exists() else {}
        interval_ready="ci_low" in prior and (name!="M2_panel" or "ci90_low" in prior)
        if prior.get("hash")==key and (not full_inference or interval_ready):
            output.append(prior)
            continue
        s=s.sort_values("time_yr",ascending=False)
        t=s.time_yr.to_numpy(dtype=float); e=s.event.to_numpy(dtype=np.int64)
        ranks=np.array([np.unique(s[f],return_inverse=True)[1]+1 for f in ["M0",name]])
        w=np.ones(len(t),dtype=np.int64)
        c=[weighted_c(t,e,r,w) for r in ranks]
        for v,f in zip(c,["M0",name]):
            np.testing.assert_allclose(v,concordance_index(t,-s[f].to_numpy(),e),atol=1e-10)
        nominal = (43 if panel=="nhanes3" else 40) if kind=="actual" else len(panels()[panel])
        row={"setting":kind,"panel":panel,"model":name,"n":len(s),"deaths":int(s.event.sum()),
             "panel_size":nominal,"median_observed":float(s.n_observed.median()),
             "M0_C":c[0],"comparator_C":c[1],"delta_M0_minus_comparator":c[0]-c[1],"hash":key}
        if full_inference:
            sims=pairs(t,e,ranks,2000)
            assert np.isfinite(sims).all()
            diff=sims[:,0]-sims[:,1]
            low95,low90,high90,high95=np.percentile(diff,[2.5,5,95,97.5])
            row.update(ci_low=low95,ci_high=high95,ci90_low=low90,ci90_high=high90,
                       paired_se=diff.std(ddof=1),bootstrap=2000)
        original.save_json(dest,row)
        output.append(row)
    return output


def combined():
    rows=[read(p) for p in (OUT/"metrics_cache").glob("*.json")]
    if rows:
        f=pd.DataFrame(rows).sort_values(["setting","panel_size","model"])
        target=OUT/"comparisons.csv"
        tmp=target.with_name(f"comparisons_{os.getpid()}.tmp")
        f.to_csv(tmp,index=False,encoding="utf-8-sig")
        tmp.replace(target)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--bootstrap",action="store_true")
    parser.add_argument("--setting",choices=["sim","external","actual","primary"])
    args=parser.parse_args()
    set_num_threads(4)
    files=sorted((OUT/"predictions").glob(f"{args.setting or '*'}_*.csv.gz"))
    for p in files:
        rows=summarize(p,args.bootstrap)
        combined()
        print(p.name,len(rows),flush=True)
