"""Read verified cohort inputs and run the frozen ensemble for the full panel."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXT = HERE.parents[1]
sys.path.insert(0, str(EXT / "scripts"))
from reference import BUNDLE, REFERENCE, TRAIN_RUN, get_cohorts, prepare_features, sha, verify
from clock_network import FTMask

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def formal(cohort):
    if cohort != "mimic":
        path = EXT / "inputs" / (cohort + ".csv.gz")
        assert sha(path) == read(EXT / "results" / cohort / "metrics.json")["source_sha256"]
        raw = pd.read_csv(path, dtype={"ID":str}).set_index("ID",verify_integrity=True)
        pred = pd.read_csv(EXT / "results" / cohort / "predictions.csv.gz",dtype={"ID":str}).set_index("ID",verify_integrity=True)
    else:
        src = EXT / "mimic_iv" / "formal_v3_1"
        audit = read(src / "audit.json")
        for name, checksum in (("predictions",audit["predictions_sha256"]),("landmark_index",audit["index_sha256"]),
                               ("lab_panel",audit["lab_audit"]["lab_panel_sha256"]),("vitals",audit["vitals_sha256"])):
            assert sha(src/(name+".csv.gz")) == checksum
        index = pd.read_csv(src/"landmark_index.csv.gz",dtype={"subject_id":str})
        vitals = pd.read_csv(src/"vitals.csv.gz",dtype={"subject_id":str},
                            usecols=["subject_id","hadm_id","sbp","dbp","pulse","bmi"])
        raw = index.merge(pd.read_csv(src/"lab_panel.csv.gz"),on="hadm_id",validate="one_to_one")
        raw = raw.merge(vitals,on=["subject_id","hadm_id"],how="left",validate="one_to_one")
        raw = raw.rename(columns={"subject_id":"ID"}).set_index("ID",verify_integrity=True)
        pred = pd.read_csv(src/"predictions.csv.gz",dtype={"ID":str}).set_index("ID",verify_integrity=True)
    raw=raw.loc[pred.index]
    np.testing.assert_allclose(raw.age,pred.age)
    if "event" in raw:
        np.testing.assert_allclose(raw.event,pred.event,equal_nan=True)
    return raw,pred

class Ensemble:
    def __init__(self, device=None):
        self.bundle=verify()
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(3)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.set_float32_matmul_precision("highest")
        self.models=[]
        for name in self.bundle["models"]:
            net=FTMask(len(self.bundle["features"]),**self.bundle["model_kwargs"])
            net.load_state_dict(torch.load(BUNDLE/name,map_location="cpu",weights_only=True))
            self.models.append(net.to(self.device).eval())

    @torch.inference_mode()
    def score(self,frame):
        x,mask,age=prepare_features(frame,self.bundle)
        valid=np.flatnonzero(mask.any(axis=1))
        out=np.full(len(frame),np.nan)
        for rows in np.array_split(valid, max(1,(len(valid)+255)//256)):
            if not len(rows):
                continue
            args=[torch.as_tensor(v[rows],device=self.device) for v in (x,mask.astype(np.float32,copy=False),age)]
            out[rows]=torch.stack([net(*args) for net in self.models]).mean(dim=0).cpu().numpy()
        return out

def h0(t,e,lp,h):
    order=np.argsort(t)
    times=t[order]
    ut,n=np.unique(times[(e[order]==1)&(times<=h)],return_counts=True)
    return (n/np.exp(lp[order])[::-1].cumsum()[::-1][np.searchsorted(times,ut)]).sum()
