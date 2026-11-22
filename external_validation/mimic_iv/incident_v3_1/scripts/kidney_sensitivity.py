"""Assess the kidney score beyond creatinine and pre-landmark measurement count."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

RUN = Path(__file__).resolve().parents[1]
MIMIC = RUN.parent
sys.path.insert(0, str(MIMIC))
from mimic_iv_panels import ORGAN_REFERENCE, association, digest, oof_comparison, verify
from mimic_iv_run import write_json


def comparison(frame, reference):
    age = (frame.age.to_numpy() - 60) / 10
    creatinine = np.log(frame.baseline_value.to_numpy())
    pre_count = np.log(frame.baseline_creatinine_n.to_numpy())
    base = np.column_stack([age, age**2, frame.sex, creatinine, creatinine**2, pre_count,
                            frame.full_accel / reference["panels"]["full"]["accel_sd_years"]])
    kidney = frame.kidney_accel.to_numpy() / reference["panels"]["kidney"]["accel_sd_years"]
    event = frame.event.to_numpy(dtype=bool)
    if not np.isfinite(base).all() or not np.isfinite(kidney).all():
        raise ValueError("Non-finite available-at-landmark covariate")
    return {"n": len(frame), "events": int(event.sum()),
            **oof_comparison(base, kidney, event, seed=20261121),
            **association(base, kidney, event)}


def main():
    verify()
    prior = MIMIC / "formal_v3_1"
    score_path = prior / "predictions.csv.gz"
    input_path = RUN / "inputs" / "kidney.csv.gz"
    formal = json.loads((prior / "audit.json").read_text(encoding="utf8"))
    input_audit = json.loads((RUN / "audit" / "kidney.json").read_text(encoding="utf8"))
    reference = json.loads(ORGAN_REFERENCE.read_text(encoding="utf8"))
    if digest(score_path) != formal["predictions_sha256"] or digest(input_path) != input_audit["input_sha256"]:
        raise ValueError("Score or endpoint audit digest differs")
    if reference["model_freeze_sha256"] != formal["model_freeze_sha256"]:
        raise ValueError("Score/reference model hashes differ")
    scores = pd.read_csv(score_path, dtype={"ID": str},
                         usecols=["ID", "age", "sex", "full_accel", "kidney_accel"])
    outcome = pd.read_csv(input_path, dtype={"ID": str})
    analysis = outcome.merge(scores, on="ID", validate="one_to_one")
    if len(analysis) != len(outcome):
        raise ValueError("Outcome rows are missing frozen risk scores")
    with threadpool_limits(limits=2):
        results = {
            "full_eligible_roster": comparison(analysis, reference),
            "restrict_to_post_landmark_repeat": comparison(analysis.loc[analysis.postlab_observed], reference),
        }
    folder = RUN / "results"
    folder.mkdir(exist_ok=True)
    write_json(folder / "kidney_additional_sensitivity.json",
               {"endpoint": "Serum creatinine-defined AKI documented by day 7",
                "base_predictors": ["age", "age squared", "sex", "full-body acceleration",
                                    "log last available creatinine", "log last available creatinine squared",
                                    "log number of distinct available baseline creatinine draws"],
                "assumptions": "Requiring a repeat test after the landmark is a monitoring sensitivity only. "
                               "It conditions on future testing and does not represent a prospective eligibility rule.",
                "input_sha256": input_audit["input_sha256"], "results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
