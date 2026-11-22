"""M3 XGBoost Cox, native NaN routing, with and without redraw per boosting step."""
import itertools
import sys

import numpy as np
from lifelines.utils import concordance_index

from design import OUT, RUN, original, read, FEATS, develop
from clock_network import GROUPS

import xgboost as xgb

DEPTHS = (2, 4)
ROUNDS = (32, 64, 128)
GROUP = np.array([i for i, xs in enumerate(GROUPS.values()) for x in xs])


def matrix(frame):
    # Log is monotone on positive raw values. Retain original thresholds and NaNs.
    return frame.reindex(columns=[*FEATS,"age"]).to_numpy(dtype=np.float32)


def masked(x, rng):
    mode = rng.random((len(x),1))
    drop = rng.random((len(x),1)) * .6
    keep = ((mode < .4) | ((mode < .75) & (rng.random((len(x),len(FEATS))) >= drop))
            | ((mode >= .75) & (rng.integers(len(GROUPS),size=(len(x),1)) == GROUP)))
    y = x.copy()
    y[:,:-1] = np.where(keep, y[:,:-1], np.nan)
    return y


def train(x, surv, years, trainrows, omitted, depth, masked_training, seed, rounds):
    t, e = surv
    x_dev = x[trainrows]
    label = np.where(e[trainrows] == 1, t[trainrows], -t[trainrows])
    params = {"objective":"survival:cox","tree_method":"hist","max_bin":128,"eta":.05,
              "max_depth":depth,"min_child_weight":5,"nthread":3,"seed":seed,"base_score":1.0}
    rng = np.random.default_rng(seed)
    dm = xgb.DMatrix(x_dev,label=label,nthread=3)
    evaluations = {yr:xgb.DMatrix(x[years==yr],nthread=3) for yr in omitted}
    booster = xgb.Booster(params=params, cache=[dm])
    results = []
    for i in range(rounds):
        batch = xgb.DMatrix(masked(x_dev,rng),label=label,nthread=3) if masked_training else dm
        booster.update(batch,i)
        if i+1 in ROUNDS:
            for yr, validation in evaluations.items():
                idx = years==yr
                c = concordance_index(t[idx],-booster.predict(validation,output_margin=True),e[idx])
                results.append({"year":yr,"lr":depth,"steps":i+1,"c":c})
    return booster, results


def run(d, variant):
    out=OUT/"fits"/("M3_xgb_"+variant)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"freeze.json").exists():
        return
    x=matrix(d)
    surv=(d.permth_exm.to_numpy(),d.event.to_numpy())
    years=d.cycle.to_numpy()
    inner, outer={}, {}
    for folds, store in ((2,inner),(1,outer)):
        for omit in itertools.combinations(original.CYCLES,folds):
            trainrows=~np.isin(years,omit)
            store[omit]=[]
            for depth in DEPTHS:
                dest=out/("cv_"+"_".join(map(str,omit))+f"_d{depth}.json")
                if not dest.exists():
                    _, rows=train(x,surv,years,trainrows,omit,depth,variant=="aug",0,max(ROUNDS))
                    original.save_json(dest, rows)
                store[omit].extend(read(dest))
            print(variant,omit,flush=True)
    nested=[]
    for yr in original.CYCLES:
        candidates=[r for key,rows in inner.items() if yr in key for r in rows if r["year"]!=yr]
        depth, steps=original.choose(candidates)
        nested.append(next(r for r in outer[(yr,)] if r["lr"]==depth and r["steps"]==steps))
    depth, steps=original.choose([r for rows in outer.values() for r in rows])
    spec={"xgboost":xgb.__version__,"objective":"survival:cox","mask_each_tree":variant=="aug",
          "depth":int(depth),"rounds":steps,"cycle_nested":nested,"seed":0,
          "tie_rule":"highest macro C, fewer trees, then shallower tree",
          "cohort_sha":read(RUN/"protocol.json")["train_ids_sha256"],"code_sha":original.sha(__file_path())}
    original.save_json(out/"cv.json",spec)
    model,_=train(x,surv,years,np.ones(len(d),bool),[],int(depth),variant=="aug",0,steps)
    model.save_model(out/"model.ubj")
    original.save_json(out/"freeze.json", {f:original.sha(out/f) for f in ("model.ubj","cv.json")})


def __file_path():
    from pathlib import Path
    return Path(__file__)


if __name__=="__main__":
    d, _ = develop()
    for v in sys.argv[1:] or ("raw","aug"):
        run(d,v)
