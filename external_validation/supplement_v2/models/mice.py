"""M6: five development-trained chained imputations and ridge Cox, cycle nested CV."""
import itertools
import pickle
import sys

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from design import OUT, FEATS, RUN, original, read, develop

DIR=OUT/"fits/M6_mice"
PENALTIES=(.001,.01,.1)
CHAINS=5
SWEEPS=3
PREDICTORS=8


def prep(f):
    raw=f.reindex(columns=[*FEATS,"age"]).copy()
    for v in original.LOGF:
        raw[v]=np.log(raw[v].clip(lower=.001))
    return raw.to_numpy(dtype=float)


def scale(raw,mu,sd):
    return np.clip((raw-mu)/sd,-5,5)


def chained(z, prediction, chain, seed, recipe=None):
    rng=np.random.default_rng(seed+chain*10531)
    m=~np.isfinite(z)
    x=np.nan_to_num(z.copy())
    if not prediction:
        correlation=np.abs(np.corrcoef(x.T))
        correlation=np.nan_to_num(correlation)
        np.fill_diagonal(correlation,-np.inf)
        strategy=[]
    for sweep in range(SWEEPS):
        for j in range(len(FEATS)):
            obs=~m[:,j]
            if not prediction:
                # Keep age and the seven most correlated other markers. Rank using
                # development inputs only; outcome and censoring never enter MICE.
                others=np.argsort(-correlation[j,:len(FEATS)],kind="stable")[:PREDICTORS-1]
                cols=np.r_[len(FEATS),others]
                design=np.column_stack((np.ones(obs.sum()),x[obs][:,cols]))
                inv=np.linalg.inv(design.T@design + np.diag([1e-9]+[1e-4]*len(cols)))
                beta=inv @ (design.T@x[obs,j])
                err=x[obs,j]-design@beta
                sigma=np.sqrt(np.dot(err,err)/rng.chisquare(len(err)-len(beta)))
                beta=beta + sigma*np.linalg.cholesky(inv)@rng.normal(size=len(beta))
                strategy.append((cols,beta,sigma))
            else:
                cols,beta,sigma=recipe[sweep*len(FEATS)+j]
            ix=np.flatnonzero(~obs)
            if len(ix):
                draws=beta[0]+x[np.ix_(ix,cols)]@beta[1:]+rng.normal(0,sigma,len(ix))
                x[ix,j]=np.clip(draws,-5,5)
    return x, None if prediction else strategy


def ridge(x,t,e,penalty):
    order=np.argsort(t,kind="stable")
    x=np.ascontiguousarray(x[order]); t=t[order]; e=e[order]
    starts=np.flatnonzero(np.r_[True,t[1:]!=t[:-1]])
    events=np.add.reduceat(e,starts)
    ntot=e.sum()
    def objective(b):
        lp=x@b
        m=lp.max(); risk=np.exp(lp-m)
        denominators=np.cumsum(risk[::-1])[::-1][starts]
        value=((events@(np.log(denominators)+m))-(e@lp))/ntot
        increments=np.zeros(len(t)); increments[starts]=events/denominators
        grad=x.T@(risk*np.cumsum(increments)-e)/ntot
        return value + penalty*np.dot(b,b)/2, grad + penalty*b
    result=minimize(objective,np.zeros(x.shape[1]),method="L-BFGS-B",jac=True,
                    options={"ftol":1e-10,"maxiter":300})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def train(raw,d,omitted,penalties):
    idx=~d.cycle.isin(omitted).to_numpy()
    mu=np.nanmean(raw[idx],axis=0); sd=np.nanstd(raw[idx],axis=0,ddof=1)
    z=scale(raw,mu,sd)
    y=d.event.to_numpy(); t=d.permth_exm.to_numpy(); years=d.cycle.to_numpy()
    models=[]; total={p:np.zeros((~idx).sum()) for p in penalties}
    for chain in range(CHAINS):
        xx,recipe=chained(z[idx],False,chain,20261121)
        coef={p:ridge(xx,t[idx],y[idx],p) for p in penalties}
        if omitted:
            yy,_=chained(z[~idx],True,chain,6124,recipe)
            for p in penalties:
                total[p]+=yy@coef[p]/CHAINS
        models.append((recipe,coef))
    results=[]
    for yr in omitted:
        mask=(years[~idx]==yr)
        for p,arr in total.items():
            results.append({"year":yr,"lr":p,"steps":1,
                            "c":concordance_index(t[~idx][mask],-arr[mask],y[~idx][mask])})
    return (mu,sd,models),results


def run():
    DIR.mkdir(parents=True,exist_ok=True)
    d,_=develop(); raw=prep(d)
    outer,inner={},{}
    for size,store in [(2,inner),(1,outer)]:
        for yrs in itertools.combinations(original.CYCLES,size):
            target=DIR/("cv_"+"_".join(map(str,yrs))+".json")
            if not target.exists():
                _,v=train(raw,d,yrs,PENALTIES)
                original.save_json(target,v)
            store[yrs]=read(target)
            print("MICE",yrs,flush=True)
    nested=[]
    for year in original.CYCLES:
        p,_=original.choose([r for k,rows in inner.items() if year in k for r in rows if r["year"]!=year])
        nested.append(next(r for r in outer[(year,)] if r["lr"]==p))
    p,_=original.choose([r for rows in outer.values() for r in rows])
    fp=DIR/"model.pkl"
    if not fp.exists():
        final,_=train(raw,d,[],[p])
        tmp=fp.with_suffix(".tmp")
        with tmp.open("wb") as f: pickle.dump(final,f)
        tmp.replace(fp)
    original.save_json(DIR/"freeze.json",{"nested":nested,"penalty":p,"chains":CHAINS,"sweeps":SWEEPS,
                       "predictors_per_equation":PREDICTORS,"model_sha256":original.sha(fp)})
    with fp.open("rb") as f:mu,sd,models=pickle.load(f)
    for path in sorted((OUT/"cache").glob("*_*.csv.gz")):
        if (DIR/path.name).exists():
            prior=pd.read_csv(DIR/path.name,usecols=["ID","M6_MICE"],dtype={"ID":str})
            original_ids=pd.read_csv(path,usecols=["ID"],dtype={"ID":str})
            if prior.ID.equals(original_ids.ID) and np.isfinite(prior.M6_MICE).all():
                continue
        df=pd.read_csv(path,dtype={"ID":str}); z=scale(prep(df),mu,sd); result=np.zeros(len(df))
        for chain,(recipe,coef) in enumerate(models):
            for start in range(0,len(df),8192):
                sl=slice(start,start+8192)
                x,_=chained(z[sl],True,chain,10242+start,recipe)
                result[sl]+=x@coef[p]/CHAINS
        dest=DIR/path.name
        tmp=dest.with_suffix(".tmp")
        df[["ID"]].assign(M6_MICE=result).to_csv(tmp,index=False,compression={"method":"gzip","mtime":0})
        tmp.replace(dest)
        print(path.name,"completed",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run()
