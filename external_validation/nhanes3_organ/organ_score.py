"""Apply the frozen NHANES organ scores and the development-only 50/10 scale."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from reference import BUNDLE, ROOT, TRAIN_RUN, get_cohorts, save_json, score_frame, sha, verify

REFERENCE = HERE / "organ_score_reference.json"


def freeze_hash():
    return sha(TRAIN_RUN / "model_freeze.json")


def verify_reference():
    bundle = verify()
    ref = json.loads(REFERENCE.read_text(encoding="utf8"))
    if ref["model_freeze_sha256"] != freeze_hash() or list(ref["panels"])[1:] != list(bundle["groups"]):
        raise ValueError("Score scale and frozen model do not match")
    return ref


def fit_reference(device):
    bundle = verify()
    dev, _, _ = get_cohorts()
    roster = TRAIN_RUN / "development_ids.csv.gz"
    pd.testing.assert_frame_equal(dev[["SEQN", "cycle"]], pd.read_csv(roster), check_dtype=False)
    scored = score_frame(dev, BUNDLE, device)
    ref = {"score_center": 50, "points_per_sd": 10, "development_n": len(dev),
           "development_roster_sha256": sha(roster), "model_freeze_sha256": freeze_hash(), "panels": {}}
    for panel in ["full", *bundle["groups"]]:
        values = scored[f"{panel}_accel"].dropna()
        params = {"n_with_measurements": len(values), "accel_mean_years": float(values.mean()),
                  "accel_sd_years": float(values.std(ddof=1))}
        if not np.isfinite(list(params.values())).all() or params["accel_sd_years"] <= 0:
            raise ValueError(f"Invalid development reference for {panel}")
        if len(values) != bundle["calibration"][panel]["n"]:
            raise ValueError(f"Scorable development roster diverged for {panel}")
        ref["panels"][panel] = params
    if REFERENCE.exists():
        previous = verify_reference()
        for panel, params in ref["panels"].items():
            np.testing.assert_allclose(list(params.values()), list(previous["panels"][panel].values()), atol=1e-5)
        return
    save_json(REFERENCE, ref)


def standardize(scored, ref):
    result = scored[[column for column in ["ID", "SEQN", "age"] if column in scored]].copy()
    for panel, params in ref["panels"].items():
        accel = scored[f"{panel}_accel"]
        number = scored[f"{panel}_n_observed"]
        if not np.array_equal(accel.notna(), number.gt(0)):
            raise ValueError(f"Missingness does not match the measured inputs for {panel}")
        result[f"{panel}_score"] = ref["score_center"] + ref["points_per_sd"] * (
            (accel - params["accel_mean_years"]) / params["accel_sd_years"]
        )
        result[f"{panel}_risk_equivalent_age"] = scored[f"{panel}_ba"]
        result[f"{panel}_accel_years"] = accel
        result[f"{panel}_n_observed"] = number
    return result


def export_nhanes(output):
    ref = verify_reference()
    source = ROOT / "results" / "nhanes3" / "predictions.csv.gz"
    checks = json.loads((ROOT / "requested_cohorts" / "verification.json").read_text(encoding="utf8"))
    if checks["cohorts"]["nhanes3"]["predictions_sha256"] != sha(source):
        raise ValueError("Original NHANES III predictions changed")
    raw = pd.read_csv(source, dtype={"ID": str})
    complete = raw[[f"{panel}_accel" for panel in ref["panels"]]].notna().all(axis=1)
    result = standardize(raw.loc[complete], ref)
    assert len(result) == 14807 and result.ID.is_unique
    write(result, output)


def write(frame, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    options = {"method": "gzip", "mtime": 0} if output.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=options)
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="command", required=True)
    setup = mode.add_parser("reference", help="Rebuild the scale from the original NHANES development cohort")
    setup.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    nhanes = mode.add_parser("export-nhanes3", help="Export the existing complete five-panel NHANES III scores")
    nhanes.add_argument("--output", type=Path, default=HERE / "nhanes3_scores_14807.csv.gz")
    score = mode.add_parser("score", help="Score a new CSV in the frozen training units")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    if args.command == "reference":
        fit_reference(args.device)
    elif args.command == "export-nhanes3":
        export_nhanes(args.output)
    else:
        if args.input.resolve() == args.output.resolve():
            parser.error("The input and output must differ")
        source = pd.read_csv(args.input, dtype={"ID": str, "SEQN": str})
        write(standardize(score_frame(source, BUNDLE, args.device), verify_reference()), args.output)
