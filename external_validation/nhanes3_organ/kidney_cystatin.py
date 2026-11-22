"""Assess whether the frozen kidney score adds concurrent cystatin C information."""

import argparse
import io
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from reference import ROOT, TRAIN_RUN, save_json, sha, verify

SOURCE = "https://wwwn.cdc.gov/Nchs/data/nhanes3/27a/sscystat.xpt"
SOURCE_SHA256 = "87904d7f0a4a8eedf0faa4c117ea62505ed85265be5e92820ac4640ba12fa613"
CODEBOOK = "https://wwwn.cdc.gov/nchs/Data/Nhanes3/27A/SSCYSTAT.pdf"
METHODS_PAPER = "https://pubmed.ncbi.nlm.nih.gov/18295054/"
GROUPS = ["liver", "kidney", "metabolic", "immune_heme", "cardiovascular"]


def cystatin_source(local_file):
    if local_file is None:
        response = requests.get(SOURCE, timeout=30)
        response.raise_for_status()
        data = response.content
    else:
        data = local_file.read_bytes()
    if sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected source digest for official NHANES III SSCYSTAT")
    cystatin = pd.read_sas(io.BytesIO(data), format="xport")
    if cystatin.SEQN.isna().any() or cystatin.SEQN.duplicated().any():
        raise ValueError("Duplicate or missing NHANES III ID in cystatin source")
    if len(cystatin) != 7721 or int(cystatin.SSCYSTAT.notna().sum()) != 7670:
        raise ValueError("Counts differ from CDC released cystatin C file")
    return cystatin.rename(columns={"SEQN": "ID"})


def regression(design, outcome, clusters, weights=None):
    ordinary = sm.WLS(outcome, design, weights=weights) if weights is not None else sm.OLS(outcome, design)
    plain = ordinary.fit()
    robust = ordinary.fit(cov_type="cluster", cov_kwds={"groups": clusters}, use_t=True)
    return plain, robust


def association(robust, column):
    coefficient = float(robust.params[column])
    ci = robust.conf_int().loc[column].to_numpy()
    return {"beta_per_sd_log_mg_L": coefficient, "robust_95ci_log_mg_L": ci.tolist(),
            "percent_change_per_sd": float(100 * np.expm1(coefficient)),
            "robust_95ci_percent_change": (100 * np.expm1(ci)).tolist(),
            "cluster_robust_p": float(robust.pvalues[column])}


def cv_performance(designs, outcome, clusters):
    n = len(outcome)
    y = outcome.to_numpy()
    predictions = {name: np.empty(n) for name in designs}
    completed = np.zeros(n, dtype=np.int8)
    splits = GroupKFold(n_splits=5).split(designs["base"], outcome, groups=clusters)
    for train, test in splits:
        completed[test] += 1
        for name, design in designs.items():
            x = design.to_numpy()
            coefficients = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
            predictions[name][test] = x[test] @ coefficients
    if not (completed == 1).all():
        raise ValueError("Incomplete held-out prediction")

    squared_error = {name: (pred - y) ** 2 for name, pred in predictions.items()}
    # Resample clusters in the paired held-out predictions; folds stay fixed.
    _, inverse = np.unique(clusters, return_inverse=True)
    counts = np.bincount(inverse)
    weights = np.random.default_rng(9527).multinomial(
        len(counts), np.repeat(1 / len(counts), len(counts)), size=1000
    )
    base_sse = np.bincount(inverse, weights=squared_error["base"])
    rng_baseline = np.sqrt((weights @ base_sse) / (weights @ counts))
    result = {}
    for name, error in squared_error.items():
        rmse = np.sqrt(error.mean())
        sse_by_cluster = np.bincount(inverse, weights=error)
        rng_rmse = np.sqrt((weights @ sse_by_cluster) / (weights @ counts))
        result[name] = {
            "rmse_log_mg_L": float(rmse),
            "r2": float(1 - error.mean() / np.var(y)),
            "rmse_reduction_vs_base": float(np.sqrt(squared_error["base"].mean()) - rmse),
            "paired_cluster_bootstrap_95ci_rmse_reduction": np.percentile(rng_baseline - rng_rmse, [2.5, 97.5]).tolist(),
        }
    return result


def main(local_file):
    verify()
    input_path = ROOT / "inputs" / "nhanes3.csv.gz"
    scores_path = ROOT / "results" / "nhanes3" / "predictions.csv.gz"
    audit = json.loads((scores_path.parent / "metrics.json").read_text(encoding="utf8"))
    if audit["source_sha256"] != sha(input_path):
        raise ValueError("NHANES III score source does not match its evaluated input")
    raw = pd.read_csv(input_path, usecols=["ID", "SDPPSU6", "SDPSTRA6"])
    scores = pd.read_csv(scores_path)
    group_columns = [name + "_accel" for name in GROUPS]
    if not scores.ID.equals(raw.ID):
        raise ValueError("NHANES III score and source identifiers are not aligned")
    complete = scores.dropna(subset=group_columns).merge(raw, on="ID", validate="one_to_one")
    if len(complete) != 14807:
        raise ValueError("Unexpected all-five-panel cohort size")
    older = complete[complete.age >= 60]
    linked = older.merge(cystatin_source(local_file), on="ID", validate="one_to_one")
    data = linked.loc[linked.SSCYSTAT.gt(0)].reset_index(drop=True)
    outcome = np.log(data.SSCYSTAT)
    if data[["SDPSTRA6", "SDPPSU6"]].isna().any().any() or not data.WTCYPEX6.gt(0).all():
        raise ValueError("Missing sampling design columns in the analytic sample")
    clusters = data.SDPSTRA6.astype(str) + ":" + data.SDPPSU6.astype(str)
    age = (data.age - 70) / 10
    base = pd.DataFrame({"constant": 1., "age_decade": age, "age_decade_squared": age ** 2,
                         "sex": data.sex, "whole_body_accel": data.full_accel})
    designs = {"base": base}
    fit_base, _ = regression(base, outcome, clusters)
    effects = {}
    for name, column in zip(GROUPS, group_columns):
        sd = data[column].std()
        design = base.assign(panel_score=(data[column] - data[column].mean()) / sd)
        ordinary, robust = regression(design, outcome, clusters)
        lr, p, df = ordinary.compare_lr_test(fit_base)
        effects[name] = association(robust, "panel_score")
        effects[name].update(panel_sd_years=float(sd), gaussian_lr_statistic=float(lr),
                             gaussian_lr_p=float(p), gaussian_lr_df=int(df))
        designs[name] = design
    _, weighted = regression(designs["kidney"], outcome, clusters, data.WTCYPEX6)
    result = {
        "counts": {"all_five_baseline": len(complete), "age_60_or_older": len(older),
                   "linked_to_cystatin": len(linked), "missing_after_linkage": len(linked) - len(data),
                   "analytic": len(data), "survey_psus": int(clusters.nunique())},
        "source": {"url": SOURCE, "sha256": SOURCE_SHA256, "codebook": CODEBOOK, "paper": METHODS_PAPER,
                   "variable": "SSCYSTAT (mg/L, per codebook variable label and original methods)",
                   "timing": "Stored serum from the primary NHANES III survey exam",
                   "selection": "All age 60+ with a stored specimen were sampled. Ages 12–59 had 25% random selection supplemented with high-creatinine participants."},
        "provenance": {"frozen_model_sha256": sha(TRAIN_RUN / "model_freeze.json"),
                       "frozen_scores_sha256": sha(scores_path), "input_sha256": sha(input_path)},
        "estimands": {"outcome": "Natural log of concurrent cystatin C concentration (mg/L)",
                       "base": "Quadratic age, sex and the frozen whole-body acceleration; all fits use the same participants.",
                       "association": "Unweighted OLS with cluster-robust t inference by masked PSU within stratum; Gaussian likelihood ratio as supplementary.",
                       "validation": "Five-fold GroupKFold, held-out PSU within stratum; 1000 paired cluster resamples of fixed out-of-fold predictions.",
                       "interpretation": "Concurrent independent measurement not used by the model. Cystatin C reflects renal function and may have nonrenal determinants. These are neither incident CKD nor measured GFR."},
        "association": effects,
        "kidney_weighted_sensitivity": association(weighted, "panel_score"),
        "validation": cv_performance(designs, outcome, clusters),
    }
    save_json(HERE / "kidney_cystatin.json", result)
    print(f"NHANES III age ≥60 ancillary cystatin C analysis complete: {len(data):,} participants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xpt", type=Path, help="Optional local copy of the official CDC SSCYSTAT.xpt")
    main(parser.parse_args().xpt)
