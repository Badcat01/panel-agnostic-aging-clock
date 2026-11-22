"""Evaluate frozen whole-body and organ scores at the 24-hour MIMIC landmark."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

EXTERNAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTERNAL / "scripts"))
from reference import BUNDLE, REFERENCE, TRAIN_RUN, linear_score, prepare_features, score_frame, verify
from validate_external import weighted_c

ORGAN_REFERENCE = EXTERNAL / "nhanes3_organ" / "organ_score_reference.json"
ENDPOINTS = {
    "index_hospital_mortality": ("hospital_event", "hospital_time_yr"),
    "post_landmark_365d_mortality": ("event", "time_yr"),
}


def digest(path):
    with path.open("rb") as file:
        out = hashlib.sha256()
        for block in iter(lambda: file.read(1 << 22), b""):
            out.update(block)
    return out.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ci95(value, var, bounds):
    margin = norm.ppf(0.975) * np.sqrt(max(var, 0))
    return [float(np.clip(value - margin, *bounds)), float(np.clip(value + margin, *bounds))]


def auc_placements(score, event):
    """Unit-level ROC concordances, giving half credit to tied scores."""
    cases, controls = score[event], score[~event]
    sorted_cases, sorted_controls = np.sort(cases), np.sort(controls)
    smaller = np.searchsorted(sorted_controls, cases, side="left")
    not_larger = np.searchsorted(sorted_controls, cases, side="right")
    control_below = (smaller + not_larger) / (2 * len(controls))
    smaller = np.searchsorted(sorted_cases, controls, side="left")
    not_larger = np.searchsorted(sorted_cases, controls, side="right")
    case_above = 1 - (smaller + not_larger) / (2 * len(cases))
    return control_below, case_above


def compare_auc(scores, event):
    """DeLong covariance of fixed scores on identical cases and controls."""
    n_cases = int(event.sum())
    n_controls = len(event) - n_cases
    if min(n_cases, n_controls) < 2:
        raise ValueError("Insufficient cases or controls to estimate AUC uncertainty")
    placements = [auc_placements(column, event) for column in scores.T]
    v_cases, v_controls = map(np.stack, zip(*placements))
    variance = np.atleast_2d(np.cov(v_cases, ddof=1)) / n_cases
    variance += np.atleast_2d(np.cov(v_controls, ddof=1)) / n_controls
    return v_cases.mean(axis=1), variance, n_cases * n_controls


def oof_comparison(base, addition, outcome, seed):
    """Fit nested logistic models in training folds and pair held-out predictions."""
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    extended = np.column_stack([base, addition])
    folds, average_precision, squared_error = [], [], np.zeros(2)
    for train, val in split.split(base, outcome):
        risk_models = []
        for features in (base, extended):
            design = features[train]
            center = design.mean(axis=0)
            scale = design.std(axis=0)
            scale[scale == 0] = 1
            fit = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
            fit.fit((design - center) / scale, outcome[train])
            risk_models.append(fit.predict_proba((features[val] - center) / scale)[:, 1])
        risk = np.column_stack(risk_models)
        squared_error += ((outcome[val, None] - risk) ** 2).sum(axis=0)
        folds.append(compare_auc(risk, outcome[val]))
        average_precision.append([average_precision_score(outcome[val], scores) for scores in risk.T])

    weights = np.asarray([pair_count for _, _, pair_count in folds], dtype=float)
    weights /= weights.sum()
    auc = sum(w * a for w, (a, _, _) in zip(weights, folds))
    pr = np.average(average_precision, axis=0, weights=weights)
    covariance = sum(w**2 * covariance for w, (_, covariance, _) in zip(weights, folds))
    contrast = np.array([-1, 1])
    delta = float(contrast @ auc)
    delta_var = float(contrast @ covariance @ contrast)
    p_value = 2 * norm.sf(abs(delta) / np.sqrt(delta_var)) if delta_var > 0 else float(delta == 0)
    return {
        "baseline_auc_oof": float(auc[0]),
        "augmented_auc_oof": float(auc[1]),
        "incremental_auc_oof": delta,
        "incremental_auc_oof_ci95": ci95(delta, delta_var, (-1, 1)),
        "incremental_auc_oof_p": float(p_value),
        "baseline_brier_oof": float(squared_error[0] / len(base)),
        "augmented_brier_oof": float(squared_error[1] / len(base)),
        "baseline_average_precision_oof": float(pr[0]),
        "augmented_average_precision_oof": float(pr[1]),
    }


def association(base, addition, outcome):
    design = sm.add_constant(np.column_stack([base, addition]), has_constant="add")
    fit = sm.GLM(outcome.astype(int), design, family=sm.families.Binomial()).fit(cov_type="HC0")
    if not fit.converged:
        raise RuntimeError("Adjusted logistic regression did not converge")
    ci = np.exp(fit.conf_int()[-1])
    return {
        "adjusted_OR_per_10_points": float(np.exp(fit.params[-1])),
        "adjusted_OR_per_10_points_ci95": ci.tolist(),
    }


def concordance(t, e, columns):
    order = np.argsort(-t, kind="stable")
    t, e = t[order], e[order].astype(np.int64)
    w = np.ones(len(t), dtype=np.int64)
    results = {}
    for name, values in columns.items():
        ranks = np.unique(values[order], return_inverse=True)[1] + 1
        results[name] = float(weighted_c(t, e, ranks, w))
    return results


def score_input(cohort, bundle):
    reference = read_json(REFERENCE)
    if reference["freeze_sha256"] != digest(TRAIN_RUN / "model_freeze.json"):
        raise ValueError("The NHANES comparators and the frozen model differ")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = score_frame(cohort, BUNDLE, device)
    x, _, age_z = prepare_features(cohort, bundle)
    for column in ["sex", "event", "time_yr", "hospital_event", "hospital_time_yr"]:
        scores[column] = cohort[column]
    scores["age_sex_lp"] = linear_score(scores[["age", "sex"]].to_numpy(), reference["age_sex"])
    scores["ridge_lp"] = linear_score(np.column_stack([x, age_z]), reference["ridge"])
    return scores, device


def check_inputs(scores, panels):
    if scores.ID.isna().any() or not scores.ID.is_unique:
        raise ValueError("Each row must be one distinct participant with an ID")
    if not np.isfinite(scores.age).all() or not scores.sex.dropna().isin([0, 1]).all():
        raise ValueError("Invalid age or sex")
    for endpoint, (event, observed_time) in ENDPOINTS.items():
        if not scores[event].isin([0, 1]).all() or not np.isfinite(scores[observed_time]).all():
            raise ValueError(f"Missing or invalid outcome {endpoint}")
        if not scores[observed_time].gt(0).all():
            raise ValueError(f"Nonpositive follow-up for {endpoint}")
    if not scores.time_yr.le(1).all() or not np.allclose(scores.loc[scores.event.eq(0), "time_yr"], 1):
        raise ValueError("The AUC requires every 365-day control to be observed at the horizon")
    for panel in panels:
        n_observed = scores[f"{panel}_n_observed"]
        mask = n_observed.gt(0)
        if not np.isfinite(n_observed).all() or not n_observed.eq(n_observed.round()).all() or n_observed.lt(0).any():
            raise ValueError(f"Invalid marker count for {panel}")
        for variable in ["lp", "accel"]:
            values = scores[f"{panel}_{variable}"]
            if not np.array_equal(mask, values.notna()) or np.isinf(values).any():
                raise ValueError(f"Scoring and measured markers disagree for {panel}")


def evaluate_panel(scores, panel, endpoint, ref, seed):
    event, observed_time = ENDPOINTS[endpoint]
    count = scores[f"{panel}_n_observed"]
    eligible = count.gt(0) & scores.full_lp.notna() & scores.sex.notna()
    d = scores.loc[eligible]
    n = len(d)
    n_cases = int(d[event].sum())
    result = dict(endpoint=endpoint, panel=panel, n_cohort=len(scores),
                  n_panel_scored=int(count.gt(0).sum()), n_compared=n,
                  deaths=n_cases, n_with_1_marker=int(d[f"{panel}_n_observed"].eq(1).sum()),
                  n_panel_markers_median=float(d[f"{panel}_n_observed"].median()) if n else None)
    if min(n_cases, n - n_cases) < 10:
        return dict(result, status="no_eligible_panel" if not n else "insufficient_events_or_controls")

    y = d[event].to_numpy(dtype=bool)
    times = d[observed_time].to_numpy(dtype=float)
    comparison = {"panel": d[f"{panel}_lp"].to_numpy()}
    if panel != "full":
        comparison["full"] = d.full_lp.to_numpy()
    for column in ("age_sex_lp", "ridge_lp"):
        if column in d:
            if not np.isfinite(d[column]).all():
                raise ValueError(f"Missing comparator value in {column}")
            comparison[column] = d[column].to_numpy()
    auc, variance, _ = compare_auc(np.column_stack(list(comparison.values())), y)
    direct = {
        name: {"auc": float(auc[i]), "auc_ci95": ci95(auc[i], variance[i, i], (0, 1))}
        for i, name in enumerate(comparison)
    }
    if not np.isclose(direct["panel"]["auc"], roc_auc_score(y, comparison["panel"]), atol=1e-12):
        raise ValueError("Fixed-score AUC cross-check failed")
    for name, value in concordance(times, y, comparison).items():
        direct[name]["harrell_c"] = value

    ages = (d.age.to_numpy() - 60) / 10
    base_columns = [ages, ages**2, d.sex.to_numpy()]
    if panel != "full":
        base_columns.append(d.full_accel.to_numpy() / ref["panels"]["full"]["accel_sd_years"])
    base = np.column_stack(base_columns)
    addition = d[f"{panel}_accel"].to_numpy() / ref["panels"][panel]["accel_sd_years"]
    delta = oof_comparison(base, addition, y, seed)
    adjusted = association(base, addition, y)
    return dict(result, status="evaluated", direct=direct, **delta, **adjusted)


def flatten(record):
    row = {}
    for key, value in record.items():
        if key == "direct":
            for score, metrics in value.items():
                row.update(flatten({f"frozen_{score}_{name}": val for name, val in metrics.items()}))
        elif key.endswith("_ci95"):
            row[f"{key}_lower"], row[f"{key}_upper"] = value
        else:
            row[key] = value
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--predictions", type=Path, help="Frozen scoring output of mimic_iv_validate.py")
    input_mode.add_argument("--cohort", type=Path, help="Landmarked cohort with feature-valued columns")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = (args.predictions or args.cohort).resolve()
    if args.out.resolve() == source:
        raise ValueError("Use a directory for outputs")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    bundle = verify()
    ref = read_json(ORGAN_REFERENCE)
    model_digest = digest(TRAIN_RUN / "model_freeze.json")
    if ref["model_freeze_sha256"] != model_digest:
        raise ValueError("The organ scale and the model manifest differ")
    panels = ["full", *bundle["groups"]]
    if panels != list(ref["panels"]):
        raise ValueError("The frozen organ scale is missing a panel")
    input_df = pd.read_csv(source, dtype={"ID": str})
    scores, device = score_input(input_df, bundle) if args.cohort else (input_df, "reused_frozen_predictions")
    check_inputs(scores, panels)
    seed = 20261121
    records = []
    with threadpool_limits(limits=2):
        for endpoint in ENDPOINTS:
            start_index = len(records)
            for panel in panels:
                t0 = time.perf_counter()
                record = evaluate_panel(scores, panel, endpoint, ref, seed)
                records.append(record)
                print(f"{endpoint} {panel} {record['status']} n={record['n_compared']:,} "
                      f"deaths={record['deaths']:,} ({time.perf_counter()-t0:.1f}s)", flush=True)

            organ_tests = [r for r in records[start_index:]
                           if r["panel"] != "full" and r["status"] == "evaluated"]
            if organ_tests:
                qs = multipletests([r["incremental_auc_oof_p"] for r in organ_tests], method="fdr_bh")[1]
                for rec, q in zip(organ_tests, qs):
                    rec["incremental_auc_oof_q_BH_organs"] = float(q)

    result = {
        "input_filename": source.name,
        "input_sha256": digest(source),
        "model_freeze_sha256": model_digest,
        "organ_reference_sha256": digest(ORGAN_REFERENCE),
        "score_device": device,
        "cohort_n": len(scores),
        "folds": 5,
        "fold_seed": seed,
        "evaluation": {
            "frozen_AUC": "Direct fixed LP on the same participants within each panel",
            "Harrell_C": "Time from 24h to first-stay death or competing alive discharge, or to death/365-day censoring",
            "hospital_AUC": "First-stay death vs alive discharge in the landmark cohort",
            "365_day_AUC": "All-cause death by 365 days after 24h landmark. Known 365-day status required",
            "incremental_AUC": "Within-fold case-control-pair weighted DeLong comparison of two fixed held-out predictions",
            "uncertainty": "DeLong conditional on fitted folds. Does not propagate repeated-training uncertainty",
            "organ_baseline": "Age, age squared, sex and frozen full-body age acceleration",
            "whole_baseline": "Age, age squared and sex",
            "adjusted_OR": "Same predictors fitted on available participants with HC0 intervals. Increment per 10 frozen score points",
            "organ_multiplicity": "BH FDR by endpoint among available organ panels only",
        },
        "panels": records,
    }
    json_path = args.out / "panel_evaluation.json"
    temp = json_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(json_path)
    csv_path = args.out / "panel_metrics.csv"
    temp = csv_path.with_suffix(".csv.tmp")
    pd.DataFrame([flatten(r) for r in records]).to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(csv_path)


if __name__ == "__main__":
    main()
