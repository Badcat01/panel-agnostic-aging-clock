"""Train M1, six panel-specific M2 models and M5 using original cycle-nested CV."""
import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from design import OUT, RUN, FEATS, develop, panels, original, read
from clock_network import FTMask

ORIGINAL_FOLD = original.Fold


class MissingMLP(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*nf+1,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.1),
                                 nn.Linear(128,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self, x, observed, age):
        return self.net(torch.cat((x.masked_fill(~observed,0), observed.float(), age[:,None]), dim=1)).flatten()


def family(name, subset, d):
    path = OUT / "fits" / name
    path.mkdir(parents=True, exist_ok=True)
    features = np.flatnonzero(np.isin(FEATS, subset))
    original.augment_mask = lambda mask: mask
    original.FTMask = (lambda nf, **kwargs: MissingMLP(len(features))) if name == "M5_mlp" else (
        lambda nf, **kwargs: FTMask(len(features), **kwargs))

    class PanelFold(ORIGINAL_FOLD):
        def __init__(self, data, years):
            super().__init__(data, years)
            self.x = self.x[:,features].contiguous()
            self.m = self.m[:,features].contiguous()

    original.Fold = PanelFold
    identity = {"name":name,"features":subset,"roster":read(RUN/"protocol.json")["train_ids_sha256"],
                "code":{f:original.sha(OUT/f) for f in ("train_ablations.py","design.py")},
                "source_code":original.sha(Path(original.__file__)), "torch":str(torch.__version__),
                "seeds":list(original.SEEDS),"rates":list(original.RATES),"steps":list(original.GRID)}
    if (path/"identity.json").exists() and read(path/"identity.json") != identity:
        raise RuntimeError("Ablation protocol changed; preserve the existing run")
    original.save_json(path/"identity.json", identity)
    if (path/"freeze.json").exists():
        assert all(original.sha(path/f)==v for f,v in read(path/"freeze.json")["sha256"].items())
        return
    print(datetime.datetime.now(), "START",name,flush=True)
    cv = original.cross_validate(d, path)
    fold = PanelFold(d, original.CYCLES)
    files = ["cross_validation.json","identity.json"]
    for seed in original.SEEDS:
        target = path / f"seed{seed}.pt"
        if not target.exists():
            print(name, "final", seed, flush=True)
            net, _ = original.fit_model(fold, cv["final_lr"], seed, cv["final_steps"])
            temporary = target.with_suffix(".tmp")
            torch.save({k:t.detach().cpu() for k,t in net.state_dict().items()},temporary)
            temporary.replace(target)
        check = original.FTMask(len(features), **original.MODEL)
        check.load_state_dict(torch.load(target,map_location="cpu",weights_only=True))
        assert all(torch.isfinite(p).all() for p in check.parameters())
        files.append(target.name)
    original.save_json(path/"freeze.json", {"finished":original.utc(),"sha256":{f:original.sha(path/f) for f in files}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    d, _ = develop()
    masks = panels()
    jobs = {"M1_nomask": FEATS,"M5_mlp":FEATS, **{"M2_"+p:masks[p] for p in ("share","elsa","charls","mimic","hrs","nhanes3")}}
    for name in args.only or jobs:
        family(name, jobs[name], d)
    print("ALL REQUESTED NETWORKS FROZEN",flush=True)
