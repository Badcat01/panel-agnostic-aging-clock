"""Rebuild honest development-cycle held-out scores for the exposure replication; no held-out outcomes in scoring."""
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from threadpoolctl import threadpool_limits

from score_holdout import HERE, TRAIN_RUN, PROJECT, load_json, sha, verify, write_json
from train_clock_gpu import CYCLES, FEATS, GROUPS, SEEDS, Fold, fit_model, get_cohorts, infer
from lifelines.utils import concordance_index

OUT=HERE/"xwas_oof"


def slope(val,train):
    d=train[["permth_exm","event"]].copy()
    d["term"]=val
    return CoxPHFitter().fit(d,"permth_exm","event").params_["term"]


def run():
    verify()
    protocol=load_json(TRAIN_RUN/"protocol.json")
    for name,digest in protocol["project_scripts"].items():
        assert sha(PROJECT/"scripts"/name)==digest
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    train,_,_=get_cohorts()
    expected=hashlib.sha256(train.SEQN.to_numpy(dtype="<i8").tobytes()).hexdigest()
    assert expected==protocol["train_ids_sha256"] and len(train)==19426
    tmpcv=load_json(TRAIN_RUN/"cross_validation.json")
    OUT.mkdir(exist_ok=True)
    final=[]
    for setting in tmpcv["nested_outer"]:
        year=setting["year"]; fold=Fold(train,[y for y in CYCLES if y!=year])
        held=np.flatnonzero(fold.y==year)
        assert not set(fold.fit)&set(held)
        saved=OUT/f"held_{year}.csv.gz"; audit=OUT/f"audit_{year}.json"
        if saved.exists() and audit.exists() and sha(saved)==load_json(audit)["predictions_sha256"]:
            final.append(pd.read_csv(saved)); continue
        capture={"SEQN":train.SEQN.to_numpy(),"cycle":fold.y}
        all_rows=np.arange(len(train))
        collected=[]
        for seed in SEEDS:
            path=OUT/f"raw_{year}_{seed}.csv.gz"
            if not path.exists():
                print(f"held cycle {year}, seed {seed}",flush=True)
                model,_=fit_model(fold,setting["lr"],seed,setting["steps"])
                table=pd.DataFrame(capture)
                for panel,features in {"full":FEATS,**GROUPS}.items():
                    mask=torch.as_tensor(np.isin(FEATS,features),device="cuda")
                    available=train[features].notna().any(axis=1).to_numpy()
                    lp=np.full(len(train),np.nan)
                    lp[available]=infer(model,fold,all_rows[available],mask)
                    table[panel]=lp
                tmp=path.with_suffix(".tmp");table.to_csv(tmp,index=False,compression="gzip");tmp.replace(path)
            cache=pd.read_csv(path)
            np.testing.assert_array_equal(cache.SEQN,train.SEQN)
            if seed==0:
                c=concordance_index(train.permth_exm.iloc[held],-cache.full.iloc[held],train.event.iloc[held])
                # Report any reproducibility drift rather than changing hyperparameters in response.
                print(f"seed0 OOF C={c:.8f} archived={setting['c']:.8f}",flush=True)
            collected.append(cache)
        out=train.iloc[held][["SEQN","cycle"]].copy()
        age_coef=slope(train.age.iloc[fold.fit].to_numpy(),train.iloc[fold.fit])
        for panel in ["full",*GROUPS]:
            lp=np.mean([c[panel].to_numpy() for c in collected],axis=0)
            ix=fold.fit[np.isfinite(lp[fold.fit])]
            s=slope(lp[ix],train.iloc[ix])/age_coef
            age=train.age.iloc[ix].to_numpy()
            ba=(lp-lp[ix].mean())*s+age.mean()
            co=np.linalg.lstsq(np.column_stack([np.ones(len(ix)),age]),ba[ix],rcond=None)[0]
            out[f"{panel}_accel"]=ba[held]-co[0]-co[1]*train.age.iloc[held].to_numpy()
        out.to_csv(saved,index=False,compression={"method":"gzip","mtime":0})
        write_json(audit,dict(held=year,fit_cycles=[y for y in CYCLES if y!=year],n_fit=len(fold.fit),n_held=len(held),
                             prior_nested_settings=setting,seeds=SEEDS,seed0_reconstructed_C=c if np.isscalar(c) else None,
                             predictions_sha256=sha(saved),development_hash=expected,
                             calibration="age and LP Cox scales, age residuals fitted only to four training cycles"))
        final.append(out)
    frame=pd.concat(final).sort_values("SEQN")
    assert frame.SEQN.is_unique and set(frame.SEQN)==set(train.SEQN)
    frame.to_csv(OUT/"development_oof.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    print("OOF complete",len(frame),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=2):run()

