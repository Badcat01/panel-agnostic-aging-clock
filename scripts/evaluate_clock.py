"""Post-freeze, unweighted NHANES temporal evaluation for the mortality clock."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from score_clock import prepare_features, score_frame


def phenoage_years(frame):
    """Original fixed BioAge coefficients; NHANES input CRP is already in mg/dL.

    https://www.aging-us.com/article/101414/supplementary/SD1/0/aging-v10i4-101414-supplementary-material-SD1.pdf
    https://github.com/dayoonkwon/BioAge/blob/master/R/phenoage_calc.R
    """
    ln_crp = np.log(frame["crp"].where(frame["crp"] > 0))
    xb = (
        -19.90667 - 0.03359355 * (frame["albumin"] * 10)
        + 0.009506491 * (frame["creatinine"] * 88.42)
        + 0.1953192 * (frame["glucose"] * 0.0555)
        + 0.09536762 * ln_crp - 0.01199984 * frame["lymph_pct"]
        + 0.02676401 * frame["mcv"] + 0.3306156 * frame["rdw"]
        + 0.001868778 * frame["alp"] + 0.05542406 * frame["wbc"]
        + 0.08035356 * frame["age"]
    )
    # Substitute the cumulative hazard directly so large risks never round to 1.
    return 141.50225 + (np.log(0.0055305 * 1.51714 / 0.007692696) + xb) / 0.090165


def risk_c_index(time, event, score):
    # lifelines expects a higher prediction for a longer survival time.
    return float(concordance_index(time, -np.asarray(score), event))


def paired_difference(time, event, dl, linear, reps=300, seed=413):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(reps):
        idx = rng.integers(0, len(time), size=len(time))
        try:
            c_dl = risk_c_index(time[idx], event[idx], dl[idx])
            c_linear = risk_c_index(time[idx], event[idx], linear[idx])
        except ZeroDivisionError:  # no comparable pairs in this replicate
            continue
        differences.append(c_dl - c_linear)
    ci = np.percentile(differences, [2.5, 97.5]).tolist() if differences else None
    return {
        "estimate": risk_c_index(time, event, dl) - risk_c_index(time, event, linear),
        "ci95_percentile": ci, "bootstrap_seed": seed,
        "replicates_requested": reps, "replicates_valid": len(differences),
    }


def linear_inputs(frame, bundle):
    x, _, age = prepare_features(frame, bundle)
    design = pd.DataFrame(x, columns=bundle["features"], index=frame.index)
    design["age"] = age
    return design


def fit_and_predict(train_x, test_x, outcomes, penalty=0.0):
    model = CoxPHFitter(penalizer=penalty).fit(
        train_x.join(outcomes), duration_col="time_yr", event_col="event"
    )
    return model.predict_log_partial_hazard(test_x).to_numpy()


def evaluate(train, test, bundle_dir):
    """Fit baselines on development participants and score the 2009–2018 holdout."""
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    for part, allowed_cycles in (
        (train, {1999, 2001, 2003, 2005, 2007}),
        (test, {2009, 2011, 2013, 2015, 2017}),
    ):
        if not part["SEQN"].is_unique or part["SEQN"].isna().any():
            raise ValueError("Each row must identify a unique NHANES participant")
        if part.empty or not part["cycle"].isin(allowed_cycles).all():
            raise ValueError("Input cycles do not match the specified temporal split")
        if not part["male"].isin([0, 1]).all() or not part["event"].isin([0, 1]).all():
            raise ValueError("A valid binary event and sex value is required for each row")
        if not np.isfinite(part["time_yr"]).all() or (part["time_yr"] <= 0).any():
            raise ValueError("Follow-up time must be finite and positive")
    if set(train["SEQN"]) & set(test["SEQN"]):
        raise ValueError("Development and holdout participants overlap")

    bundle_dir = Path(bundle_dir)
    bundle = json.loads((bundle_dir / "clock_bundle.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = score_frame(test, bundle_dir, device)
    predictions[["cycle", "male", "time_yr", "event"]] = test[["cycle", "male", "time_yr", "event"]]
    outcomes = train[["time_yr", "event"]]
    predictors = ["age", "male"]
    predictions["age_sex_lp"] = fit_and_predict(train[predictors], test[predictors], outcomes)
    predictions["ridge_cox_lp"] = fit_and_predict(
        linear_inputs(train, bundle), linear_inputs(test, bundle), outcomes, penalty=0.01
    )
    predictions["phenoage"] = phenoage_years(test)

    scored = predictions.loc[predictions["full_lp"].notna()]
    time = scored["time_yr"].to_numpy()
    event = scored["event"].to_numpy()
    dl = scored["full_lp"].to_numpy()
    ridge = scored["ridge_cox_lp"].to_numpy()
    metrics = {
        "n": len(test), "deaths": int(test["event"].sum()),
        "n_scored": len(scored), "deaths_scored": int(event.sum()),
        "n_train_baselines": len(train), "deaths_train_baselines": int(train["event"].sum()),
        "C_DL": risk_c_index(time, event, dl),
        "C_age_sex": risk_c_index(time, event, scored["age_sex_lp"]),
        "C_ridge_cox": risk_c_index(time, event, ridge),
        "delta_C_DL_vs_ridge_cox": paired_difference(time, event, dl, ridge),
    }
    available = scored.loc[np.isfinite(scored["phenoage"])]
    same_people = {"n": len(available), "deaths": int(available["event"].sum())}
    for metric, column in {"C_PhenoAge": "phenoage", "C_DL": "full_lp", "C_ridge_cox": "ridge_cox_lp"}.items():
        try:
            same_people[metric] = risk_c_index(available["time_yr"], available["event"], available[column])
        except ZeroDivisionError:
            same_people[metric] = None
    metrics["phenoage_complete_cases"] = same_people

    association = scored[["age", "male", "time_yr", "event"]].copy()
    sd = float(scored["full_accel"].std())
    association["accel_per_sd"] = (scored["full_accel"] - scored["full_accel"].mean()) / sd
    fit = CoxPHFitter().fit(association, duration_col="time_yr", event_col="event")
    metrics["full_accel_adjusted_age_sex"] = {
        "hr_per_holdout_sd": float(np.exp(fit.params_["accel_per_sd"])),
        "ci95": np.exp(fit.confidence_intervals_.loc["accel_per_sd"]).tolist(),
        "p": float(fit.summary.loc["accel_per_sd", "p"]),
        "sd_years": sd,
    }
    destination = bundle_dir.parent / "temporal_predictions.csv.gz"
    temporary = destination.with_name(destination.name + ".tmp")
    predictions.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(destination)
    return metrics


def evaluate_frozen_run(run):
    """Verify the frozen inputs before accessing the temporal holdout."""
    run = Path(run).resolve()
    bundle_dir = run / "bundle"
    marker = run / "model_freeze.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Model is not frozen: {marker}")
    from train_clock_gpu import PROJECT, get_cohorts, save_json, sha

    recorded = json.loads(marker.read_text(encoding="utf-8"))["files_sha256"]
    if "clock_bundle.json" not in recorded:
        raise ValueError("clock_bundle.json is missing from the freeze manifest")
    for name, digest in recorded.items():
        path = bundle_dir / name
        if path.parent != bundle_dir or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Frozen bundle failed SHA-256 verification: {name}")
    bundle = json.loads((bundle_dir / "clock_bundle.json").read_text(encoding="utf-8"))
    if not bundle["models"] or not set(bundle["models"]).issubset(recorded):
        raise ValueError("Freeze manifest omits one or more models")
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["project_scripts"].items():
        if sha(PROJECT / "scripts" / name) != digest:
            raise ValueError(f"Training pipeline changed since protocol capture: {name}")
    for name, digest in protocol["data"].items():
        if sha(PROJECT / "data" / "processed" / name) != digest:
            raise ValueError(f"Processed data changed since protocol capture: {name}")
    train, test, _ = get_cohorts()
    roster = pd.read_csv(run / "development_ids.csv.gz")
    rebuilt = train[["SEQN", "cycle"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(roster, rebuilt, check_dtype=False)
    digest = hashlib.sha256(rebuilt["SEQN"].to_numpy(dtype="<i8").tobytes()).hexdigest()
    if digest != protocol["train_ids_sha256"] or len(train) != bundle["n_train"]:
        raise ValueError("Rebuilt training participants differ from the frozen training population")

    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    metrics = evaluate(train, test, bundle_dir)
    metrics["model_freeze_sha256"] = sha(marker)
    metrics["development_ids_sha256"] = digest
    metrics["predictions_sha256"] = sha(run / "temporal_predictions.csv.gz")
    save_json(run / "temporal_metrics.json", metrics)
    print(f"Temporal evaluation saved to {run / 'temporal_metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Frozen training run directory")
    evaluate_frozen_run(parser.parse_args().run)
