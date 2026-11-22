"""Same people, random nested subsets of genuinely observed assays per patient."""
import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from threadpoolctl import threadpool_limits

from design import OUT, RUN, FEATS, original, read
from predict import matrix, netrisk, verify, xgb


def run():
    cache=OUT/"cache/primary_full.csv.gz"
    f=pd.read_csv(cache,dtype={"ID":str})
    bundle=verify()
    fits=OUT/"fits"
    models={"M0":([RUN/"bundle"/f"ftmask_seed{i}.pt" for i in original.SEEDS],False)}
    for dr,mname,mlp in [("M1_nomask","M1_nomask",False),("M5_mlp","M5_MLP",True)]:
        if (fits/dr/"freeze.json").exists():
            models[mname]=([fits/dr/f"seed{i}.pt" for i in original.SEEDS],mlp)
    tree={}
    for v in ["aug","raw"]:
        p=fits/("M3_xgb_"+v)
        if (p/"freeze.json").is_file():
            b=xgb.Booster(); b.load_model(p/"model.ubj")
            tree["M3_XGB_"+v]=b
    rows=[]
    original_values=f[FEATS].to_numpy()
    assert np.isfinite(original_values).sum(axis=1).min()>=22
    for seed in (20261121,431,2026):
        rng=np.random.default_rng(seed)
        priority=rng.random(original_values.shape)
        priority[~np.isfinite(original_values)]=np.inf
        ranks=np.argsort(np.argsort(priority,axis=1),axis=1)
        for k in (1,3,6,12,18,22):
            path=OUT/"richness_cache"/f"s{seed}_k{k}.csv"
            path.parent.mkdir(exist_ok=True)
            prior=pd.read_csv(path).to_dict("records") if path.is_file() else []
            done={r["model"] for r in prior}
            frame=f.copy()
            frame[FEATS]=np.where(ranks<k,original_values,np.nan)
            assert frame[FEATS].notna().sum(axis=1).eq(k).all()
            risk={}
            for name,(weights,mlp) in models.items():
                if name not in done:
                    risk[name]=netrisk(frame,bundle,FEATS,weights,mlp)
            for name,booster in tree.items():
                if name not in done:
                    risk[name]=booster.predict(xgb.DMatrix(matrix(frame),nthread=2),output_margin=True)
            for name,x in risk.items():
                c=concordance_index(frame.time_yr,-x,frame.event)
                prior.append({"model":name,"seed":seed,"available_assays":k,"n":len(frame),
                              "events":int(frame.event.sum()),"c_index":c})
            if risk:
                pd.DataFrame(prior).to_csv(path,index=False)
                print(seed,k,flush=True)
            rows.extend(prior)
    pd.DataFrame(rows).to_csv(OUT/"random_panel_richness.csv",index=False)


if __name__=="__main__":
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    with threadpool_limits(limits=2):
        run()
