"""Freeze the external comparison models and absolute-risk map using NHANES only."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
TRAIN_RUN = PROJECT / "runs" / "nhanes_gpu_20261121"
BUNDLE = TRAIN_RUN / "bundle"
sys.path[:0] = [str(BUNDLE), str(PROJECT / "scripts")]
from score_clock import prepare_features, score_frame
from train_clock_gpu import get_cohorts, save_json, sha

HORIZONS = [3, 5, 10]
REFERENCE = ROOT / "nhanes_reference.json"


def verify():
    freeze = json.loads((TRAIN_RUN / "model_freeze.json").read_text(encoding="utf8"))
    for name, digest in freeze["files_sha256"].items():
        if sha(BUNDLE / name) != digest:
            raise ValueError(f"Frozen file digest differs: {name}")
    return json.loads((BUNDLE / "clock_bundle.json").read_text(encoding="utf8"))


def fit_comparator(design, dev, penalty):
    outcomes = dev[["time_yr","event"]].reset_index(drop=True)
    model = CoxPHFitter(penalizer=penalty).fit(design.join(outcomes), "time_yr", "event")
    return {"columns": list(design), "center": design.mean().tolist(), "coef": model.params_.loc[design.columns].tolist()}


def linear_score(array, spec):
    return (array - np.array(spec["center"])) @ np.array(spec["coef"])


def build():
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    bundle = verify()
    dev, _, _ = get_cohorts()
    pd.testing.assert_frame_equal(dev[["SEQN","cycle"]], pd.read_csv(TRAIN_RUN/"development_ids.csv.gz"), check_dtype=False)
    train_digest = hashlib.sha256(dev.SEQN.to_numpy(dtype="<i8").tobytes()).hexdigest()
    protocol = json.loads((TRAIN_RUN/"protocol.json").read_text(encoding="utf8"))
    assert train_digest == protocol["train_ids_sha256"]
    x, _, a = prepare_features(dev, bundle)
    z = pd.DataFrame(x, columns=bundle["features"]).assign(age=a)
    demographic = dev[["age","male"]].astype(float)
    reference = dict(n=len(dev), events=int(dev.event.sum()), development_ids_sha256=train_digest,
                     freeze_sha256=sha(TRAIN_RUN / "model_freeze.json"),
                     protocol_sha256=sha(ROOT / "验证方案.md"),
                     age_sex=fit_comparator(demographic,dev,0), ridge=fit_comparator(z,dev,0.01))
    full = score_frame(dev, BUNDLE, "cuda")
    center = bundle["calibration"]["full"]["lp_mean"]
    slope = bundle["calibration"]["full"]["scale_years"] * bundle["age_cox_slope"]
    reference["risk_lp_mean"] = center
    reference["risk_lp_slope"] = slope
    reference["train_accel_sd"] = float(full.full_accel.std())
    t,e = dev.time_yr.to_numpy(), dev.event.to_numpy()
    order = np.argsort(t)
    sorted_times = t[order]
    unique, count = np.unique(t[e==1],return_counts=True)
    scores = {"clock": slope*(full.full_lp.to_numpy()-center),
              "age_sex": linear_score(demographic.to_numpy(),reference["age_sex"]),
              "ridge": linear_score(z.to_numpy(),reference["ridge"])}
    reference["baseline_hazard"] = {}
    for key, lp in scores.items():
        at_risk = np.cumsum(np.exp(lp[order])[::-1])[::-1]
        increments = count/at_risk[np.searchsorted(sorted_times,unique)]
        reference["baseline_hazard"][key] = {str(h):float(increments[unique<=h].sum()) for h in HORIZONS}
    save_json(REFERENCE, reference)
    print("Verified frozen ensemble and prepared development-only risk maps", len(dev))


if __name__ == "__main__":
    build()
