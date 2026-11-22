"""Fair panel-specific ridge Cox comparator trained in the original nested cycle split."""
import itertools
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from threadpoolctl import threadpool_limits

from design import OUT, FEATS, original, panels, develop
from mice import prep, scale, ridge, PENALTIES


def estimate(raw, d, om, cols, penalties):
    train=~d.cycle.isin(om).to_numpy()
    mu=np.nanmean(raw[train],axis=0)
    sd=np.nanstd(raw[train],axis=0,ddof=1)
    x=np.nan_to_num(scale(raw,mu,sd),nan=0.0)[:,cols]
    event=d.event.to_numpy();times=d.permth_exm.to_numpy()
    rows=[]; coefficients={}
    for pen in penalties:
        b=ridge(x[train],times[train],event[train],pen)
        coefficients[pen]=b
        for yr in om:
            ix=d.cycle.eq(yr).to_numpy()
            rows.append(dict(year=yr,lr=pen,steps=1,c=concordance_index(times[ix],-(x[ix]@b),event[ix])))
    return rows,coefficients,mu,sd


def run():
    d,_=develop()
    raw=prep(d)
    temp=OUT/"fits/Ridge_panel"
    temp.mkdir(parents=True,exist_ok=True)
    for panel,feature_names in panels().items():
        cols=[FEATS.index(f) for f in feature_names]+[len(FEATS)]
        inner={};outer={}
        for count,store in ((2,inner),(1,outer)):
            for om in itertools.combinations(original.CYCLES,count):
                store[om]=estimate(raw,d,om,cols,PENALTIES)[0]
        outer_perf=[]
        for yr in original.CYCLES:
            chosen,_=original.choose([r for om,rows in inner.items() if yr in om for r in rows if r["year"]!=yr])
            outer_perf.append(next(r for r in outer[(yr,)] if r["lr"]==chosen))
        selected,_=original.choose([r for rows in outer.values() for r in rows])
        _,models,mu,sd=estimate(raw,d,(),cols,[selected])
        b=models[selected]
        original.save_json(temp/(panel+"_frozen.json"),{"features":[*feature_names,"age"],"coefficients":b.tolist(),
                           "mu":mu[cols].tolist(),"sd":sd[cols].tolist(),"penalty":selected,"outer":outer_perf})
        for kind in ("primary","sim","external"):
            path=OUT/"cache"/f"{kind}_{panel}.csv.gz"
            if not path.exists():continue
            frame=pd.read_csv(path,dtype={"ID":str})
            x=np.nan_to_num(scale(prep(frame),mu,sd),nan=0.0)[:,cols]
            frame[["ID"]].assign(ridge_panel=x@b).to_csv(temp/path.name,index=False)
        print(panel,selected,flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run()
