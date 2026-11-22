"""Validate frozen organ-group scores against NHANES III linked causes of death."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index
from numba import njit
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from reference import BUNDLE, TRAIN_RUN, save_json, sha, verify

PANELS = ["liver", "kidney", "metabolic", "immune_heme", "cardiovascular"]
COVARIATES = ["age10", "age10_sq", "sex"]
TARGETS = {"cardiovascular": "cardiovascular", "renal": "kidney", "diabetes": "metabolic",
           "influenza_pneumonia": "immune_heme", "diabetes_mentioned": "metabolic"}
PRIMARY = list(TARGETS)[:4]
HORIZON = 20.0


def cohort():
    bundle = verify()
    predictions = ROOT / "results" / "nhanes3" / "predictions.csv.gz"
    audit = json.loads((ROOT / "audit" / "nhanes3.json").read_text(encoding="utf8"))
    source = ROOT / "nhanes3_source" / "NHANES_III_MORT_2019_PUBLIC.dat"
    digest = sha(source)
    if digest != audit["sources"][source.name]["sha256"]:
        raise ValueError("CDC mortality file differs from the validation source")
    summary = json.loads((ROOT / "requested_cohorts" / "verification.json").read_text(encoding="utf8"))
    if sha(predictions) != summary["cohorts"]["nhanes3"]["predictions_sha256"]:
        raise ValueError("Prior frozen NHANES III predictions have changed")
    scores = pd.read_csv(predictions, dtype={"ID": str})
    panels = ["full", *PANELS]
    observed = [f"{panel}_n_observed" for panel in panels]
    ages = [f"{panel}_accel" for panel in panels]
    if list(bundle["groups"]) != PANELS:
        raise ValueError("Unexpected score groups in frozen manifest")
    if not np.array_equal(scores[ages].notna().to_numpy(), scores[observed].gt(0).to_numpy()):
        raise ValueError("Observation counts do not agree with existing scores")
    data = scores.loc[scores[ages].notna().all(axis=1)].copy()
    data["ID"] = pd.to_numeric(data["ID"], errors="raise")
    cols = ["ID", "eligible", "event_source", "ucod", "diabetes_mcod", "hypertension_mcod", "months"]
    linkage = pd.read_fwf(source, colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (45, 48)],
                         names=cols, dtype={"ucod": str}, na_values=["."])
    data = data.merge(linkage, on="ID", validate="one_to_one")
    assert len(data) == 14807 and data.eligible.eq(1).all()
    assert data.sex.isin([0, 1]).all() and data.event.eq(data.event_source).all()
    np.testing.assert_allclose(data.time_yr, data.months / 12)
    data["unknown"] = data.event.eq(1) & data.ucod.isna()
    assert data.ucod.dropna().isin([f"{i:03d}" for i in range(1, 11)]).all()
    assert not data.loc[data.event.eq(0), "ucod"].notna().any()
    data["age10"] = (data.age - 60) / 10
    data["age10_sq"] = data.age10**2
    scale = {}
    for panel in panels:
        column = data[f"{panel}_accel"]
        sd = float(column.std())
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid score variance: {panel}")
        data[panel] = (column - column.mean()) / sd
        scale[panel] = sd
    return data, {"original_n": len(data), "original_deaths": int(data.event.sum()),
                  "unknown_causes": int(data.unknown.sum()), "score_sd_years": scale,
                  "mortality_sha256": digest, "predictions_sha256": sha(predictions),
                  "model_freeze_sha256": sha(TRAIN_RUN / "model_freeze.json")}


def outcomes(d):
    return {"all_cause": d.event.eq(1), "cardiovascular": d.ucod.isin(["001", "005"]),
            "renal": d.ucod.eq("009"), "diabetes": d.ucod.eq("007"),
            "influenza_pneumonia": d.ucod.eq("008"),
            "diabetes_mentioned": d.diabetes_mcod.eq(1),
            "malignant_neoplasm": d.ucod.eq("002"), "accidents": d.ucod.eq("004")}


def fit_cox(d, columns, outcome):
    x = d[["time_yr", *columns]].assign(failure=outcome.to_numpy(dtype=int))
    for step in (0.95, 0.25):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = CoxPHFitter().fit(x, duration_col="time_yr", event_col="failure",
                                     fit_options={"step_size": step, "max_steps": 700})
        if not any(issubclass(item.category, ConvergenceWarning) for item in caught):
            return model
    raise RuntimeError(f"Cox convergence failed for {columns}, n={len(x)}, events={int(x.failure.sum())}")


def cox_result(base, extended, panel):
    summary = extended.summary.loc[panel]
    stat = 2 * (extended.log_likelihood_ - base.log_likelihood_)
    return {"hr": float(summary["exp(coef)"]), "lo": float(summary["exp(coef) lower 95%"]),
            "hi": float(summary["exp(coef) upper 95%"]), "lrt_stat": float(stat), "lrt_p": float(chi2.sf(stat, 1))}


@njit
def weight_concordance(rank, target, sizes, weights):
    """Return weighted concordant pairs and all case-control pairs at one horizon."""
    event_hist = np.zeros(sizes)
    control_hist = np.zeros(sizes)
    for i in range(len(rank)):
        if target[i]:
            event_hist[rank[i]] += weights[i]
        else:
            control_hist[rank[i]] += weights[i]
    below = wins = 0.0
    for r in range(sizes):
        wins += event_hist[r] * (below + 0.5 * control_hist[r])
        below += control_hist[r]
    return wins, event_hist.sum() * control_hist.sum()


def oof_prediction(d, target, panel):
    short_event = (target & d.time_yr.le(HORIZON)).to_numpy(dtype=int)
    if (d.event.eq(0) & d.time_yr.lt(HORIZON)).any():
        raise ValueError("Fixed-time AUC requires known 20-year status for every participant")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261121)
    models = [COVARIATES + [field] for field in ["full", panel]] + [COVARIATES + ["full", panel]]
    folds, briers, cs = [], [], []
    for train, test in splitter.split(d, short_event):
        dev, val = d.iloc[train], d.iloc[test]
        labels = short_event[test]
        ranks, steps, errors, cs_fold = [], [], [], []
        for fields in models:
            classifier = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-8)
            classifier.fit(dev[fields], short_event[train])
            risk = classifier.predict_proba(val[fields])[:, 1]
            distinct, inverse = np.unique(risk, return_inverse=True)
            ranks.append(inverse)
            steps.append(len(distinct))
            errors.append(float(np.square(risk-labels).sum()))
            cox = fit_cox(dev, fields, target.iloc[train])
            score = cox.predict_log_partial_hazard(val[fields])
            c = concordance_index(val.time_yr, -score.to_numpy(), target.iloc[test])
            cs_fold.append(float(c))
        folds.append((labels, ranks, steps))
        briers.append(errors)
        cs.append(cs_fold)

    def scores(sampled):
        numerator = np.zeros(len(models))
        denominator = np.zeros(len(models))
        for w, (labels, ranks, steps) in zip(sampled, folds):
            for j, rank in enumerate(ranks):
                wins, pairs = weight_concordance(rank, labels, steps[j], w)
                numerator[j] += wins
                denominator[j] += pairs
        return numerator/denominator

    auc = scores([np.ones(len(k[0]), dtype=int) for k in folds])
    rng = np.random.default_rng(2201)
    replicates = []
    for _ in range(500):
        sample = [np.bincount(rng.integers(0, len(k[0]), len(k[0])), minlength=len(k[0])) for k in folds]
        a = scores(sample)
        if np.isfinite(a).all():
            replicates.append(a)
    simulations = np.asarray(replicates)
    return {"horizon_years": HORIZON, "events_by_horizon": int(short_event.sum()),
            "other_deaths_by_horizon": int((d.event.eq(1) & ~target & d.time_yr.le(HORIZON)).sum()),
            "auc_full_organ_combined": auc.tolist(),
            "auc_organ_minus_full": float(auc[1]-auc[0]),
            "auc_organ_minus_full_ci": np.percentile(simulations[:,1]-simulations[:,0],[2.5,97.5]).tolist(),
            "auc_combined_minus_full": float(auc[2]-auc[0]),
            "auc_combined_minus_full_ci": np.percentile(simulations[:,2]-simulations[:,0],[2.5,97.5]).tolist(),
            "cause_c_fold_mean_full_organ_combined": np.mean(cs, axis=0).tolist(),
            "brier_full_organ_combined": (np.sum(briers, axis=0)/len(d)).tolist(),
            "bootstrap_n": len(simulations)}


def main():
    data, audit = cohort()
    relatives = data[["full", *PANELS]].corr()
    relatives.to_csv(HERE / "score_correlations.csv")
    audit["followup_median_years"] = float(data.time_yr.median())
    audit["shortest_alive_followup_years"] = float(data.loc[data.event.eq(0),"time_yr"].min())
    audit["underlying_cause_counts"] = {str(k): int(v) for k,v in data.loc[data.event.eq(1),"ucod"].value_counts().items()}
    audit["diabetes_mcod_count"] = int(data.diabetes_mcod.eq(1).sum())
    audit["hypertension_mcod_count"] = int(data.hypertension_mcod.eq(1).sum())
    audit["unknown_mcod_for_deaths"] = int(data.loc[data.event.eq(1),"diabetes_mcod"].isna().sum())
    save_json(HERE / "source_audit.json", audit)
    result, rows, oof = {}, [], {}
    for name in outcomes(data):
        d = data if name == "all_cause" else data.loc[~data.unknown].copy()
        target = outcomes(d)[name]
        assert target.sum() >= 20
        print(f"{name}: {len(d)} participants, {int(target.sum())} events", flush=True)
        base = fit_cox(d, COVARIATES + ["full"], target)
        age_sex = fit_cox(d, COVARIATES, target)
        for panel in PANELS:
            combined = fit_cox(d, COVARIATES + ["full", panel], target)
            fitted = cox_result(base, combined, panel)
            rows.append({"endpoint": name, "panel": panel, "n":len(d), "deaths":int(target.sum()), **fitted})
            if panel != TARGETS.get(name):
                continue
            organ_only = fit_cox(d, COVARIATES + [panel], target)
            unknown = outcomes(data)[name]
            base_unknown = fit_cox(data, COVARIATES + ["full"], unknown)
            extended_unknown = fit_cox(data, COVARIATES + ["full", panel], unknown)
            result[name] = {"organ": panel, "n": len(d), "events": int(target.sum()),
                            "other_deaths": int(d.event.sum()-target.sum()), "incremental": fitted,
                            "organ_without_full": cox_result(age_sex, organ_only, panel),
                            "full_without_organ": cox_result(age_sex, base, "full"),
                            "unknown_as_competing": cox_result(base_unknown, extended_unknown, panel)}
            oof[name] = oof_prediction(d, target, panel)
    matrix = pd.DataFrame(rows)
    matrix["exploratory_matrix_fdr"] = multipletests(matrix.lrt_p, method="fdr_bh")[1]
    matrix.to_csv(HERE / "cause_score_matrix.csv", index=False)
    save_json(HERE / "mortality_associations.json", result)
    save_json(HERE / "mortality_prediction.json", oof)


if __name__ == "__main__":
    main()
