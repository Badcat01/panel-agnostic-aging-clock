"""Apply only fully frozen competitors to matched panel inputs, append atomically."""
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from design import EXT, FEATS, OUT, RUN, original, panels, read
from train_ablations import FTMask, MissingMLP
from train_tree import matrix, xgb
from score_clock import prepare_features
sys.path.insert(0,str(EXT/"scripts"))
from reference import REFERENCE, linear_score, verify


@torch.inference_mode()
def netrisk(frame, bundle, columns, weights, mlp=False):
    x, m, age=prepare_features(frame,bundle)
    ix=[FEATS.index(f) for f in columns]
    res=np.zeros(len(x),float)
    for weight in weights:
        net = MissingMLP(len(ix)) if mlp else FTMask(len(ix),**original.MODEL)
        net.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
        net=net.cuda().eval()
        for start in range(0,len(x),512):
            sl=slice(start,start+512)
            tensors=[torch.as_tensor(a,device="cuda") for a in (x[sl][:,ix],m[sl][:,ix],age[sl])]
            res[sl]+=net(*tensors).cpu().numpy()/len(weights)
    return res


def process(path,bundle,ref):
    prefix=path.name.split(".")[0]
    kind,panel=prefix.split("_",1)
    df=pd.read_csv(path,dtype={"ID":str})
    features=[f for f in FEATS if f in df] if kind=="actual" else panels()[panel]
    assert df.ID.is_unique and df.event.isin([0,1]).all() and np.all(df.time_yr>0)
    out=OUT/"predictions"/(prefix+".csv.gz")
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        values=pd.read_csv(out,dtype={"ID":str},float_precision="round_trip")
        assert values.ID.equals(df.ID)
    else:
        values=df[["ID","time_yr","event"]].copy()
        values["n_observed"]=df[features].notna().sum(axis=1)
    before=len(values.columns)
    if "M0" not in values:
        weights=[RUN/"bundle"/f"ftmask_seed{s}.pt" for s in original.SEEDS]
        values["M0"]=netrisk(df,bundle,FEATS,weights)
        z,_,a=prepare_features(df,bundle)
        values["ridge_frozen"]=linear_score(np.c_[z,a],ref["ridge"])
        values["age_sex"]=linear_score(df[["age","sex"]].to_numpy(),ref["age_sex"])
    for variant in ("raw","aug"):
        label="M3_XGB_"+variant
        directory=OUT/"fits"/("M3_xgb_"+variant)
        if label not in values and (directory/"freeze.json").is_file():
            assert all(original.sha(directory/f)==v for f,v in read(directory/"freeze.json").items())
            model=xgb.Booster(); model.load_model(directory/"model.ubj")
            values[label]=model.predict(xgb.DMatrix(matrix(df),nthread=2),output_margin=True)
    for name,outname in [("M1_nomask","M1_nomask"),("M5_mlp","M5_MLP"),("M2_"+panel,"M2_panel")]:
        if kind=="actual" and outname=="M2_panel":
            continue
        directory=OUT/"fits"/name
        if outname not in values and (directory/"freeze.json").exists():
            freeze=read(directory/"freeze.json")["sha256"]
            assert all(original.sha(directory/f)==v for f,v in freeze.items())
            cols=read(directory/"identity.json")["features"]
            files=[directory/f"seed{s}.pt" for s in original.SEEDS]
            values[outname]=netrisk(df,bundle,cols,files,mlp=name=="M5_mlp")
    for sub,p in [("M4_RSF",OUT/"fits"/("M4_rsf_"+panel)/f"{kind}.csv.gz"),
                  ("M6_MICE",OUT/"fits/M6_mice"/(prefix+".csv.gz")),
                  ("ridge_panel",OUT/"fits/Ridge_panel"/(prefix+".csv.gz")),
                  ("M7",OUT/"fits/M7_bioage"/(prefix+".csv.gz"))]:
        if p.exists():
            other=pd.read_csv(p,dtype={"ID":str}).set_index("ID")
            assert other.index.is_unique and set(other.index)==set(df.ID)
            for col in other:
                if col not in values:
                    values[col]=df.ID.map(other[col])
    if len(values.columns)>before:
        tmp=out.with_suffix(".tmp")
        values.to_csv(tmp,index=False,compression={"method":"gzip","mtime":0})
        tmp.replace(out)
    print(prefix,len(values),list(values.columns[4:]),flush=True)


if __name__=="__main__":
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    b=verify()
    with threadpool_limits(limits=2):
        for p in sorted((OUT/"cache").glob("*_*.csv.gz")):
            process(p,b,read(REFERENCE))
