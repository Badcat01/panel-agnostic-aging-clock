"""Summarize the five data sources requested for this validation round."""

import argparse
import json

import pandas as pd

from reference import REFERENCE, ROOT, TRAIN_RUN, sha, save_json, verify

COHORTS = {"nhanes3": "NHANES III", "charls": "CHARLS", "hrs": "HRS 2016", "elsa": "ELSA"}
OUTPUT = ROOT / "requested_cohorts"


def read(file):
    return json.loads(file.read_text(encoding="utf8"))


def collect(cohorts):
    verify()
    earlier = ROOT / "results" / "completion_check.json"
    previous = read(earlier)["cohort_checks"] if earlier.exists() else {}
    rows, horizons, checks = [], [], {}
    for key in cohorts:
        name = COHORTS[key]
        source = ROOT / "inputs" / f"{key}.csv.gz"
        folder = ROOT / "results" / key
        metric_file = folder / "metrics.json"
        metrics = read(metric_file)
        pred_digest = sha(folder / "predictions.csv.gz")
        if metrics.get("predictions_sha256", pred_digest) != pred_digest:
            raise ValueError(f"Predictions differ from current evaluation for {key}")
        if key in previous and previous[key]["predictions_sha256"] != pred_digest:
            raise ValueError(f"Previously verified predictions changed for {key}")
        if metrics["source_sha256"] != sha(source) or metrics["reference_sha256"] != sha(REFERENCE):
            raise ValueError(f"Evaluation provenance mismatch for {key}")
        data = pd.read_csv(source, dtype={"ID": str})
        pred = pd.read_csv(folder / "predictions.csv.gz", dtype={"ID": str})
        if not data.ID.is_unique or not data.ID.equals(pred.ID):
            raise ValueError(f"Prediction IDs do not align with input for {key}")
        valid = pred.time_yr.gt(0) & pred.event.notna() & pred.sex.notna() & pred.full_lp.notna()
        if int(valid.sum()) != metrics["n_sex"] or int(pred.loc[valid, "event"].sum()) != metrics["deaths_sex"]:
            raise ValueError(f"Event counts do not align for {key}")
        checks[key] = {"metrics_sha256": sha(metric_file), "input_sha256": metrics["source_sha256"],
                       "predictions_sha256": pred_digest}
        common = metrics["same_sex_available"]
        row = dict(cohort=name, n=metrics["n_sex"], deaths=metrics["deaths_sex"], markers=metrics["median_features"],
                   n_markers_ge22=metrics["sensitivities"]["at_least22"]["n"],
                   followup_median_years=metrics["observed_followup_median_years"])
        for label, field in [("clock", "full_lp"), ("age_sex", "age_sex_lp"), ("ridge", "ridge_lp")]:
            row[f"c_{label}"] = common[field]["C"]
            row[f"c_{label}_lower"], row[f"c_{label}_upper"] = common[field]["CI95"]
            if label != "clock":
                row[f"delta_vs_{label}"] = common[field]["delta_clock_minus_comparator"]
                row[f"delta_{label}_lower"], row[f"delta_{label}_upper"] = common[field]["delta_CI95"]
        assoc = metrics["accel_association"]
        row.update(accel_HR=assoc["HR"], accel_HR_lower=assoc["CI95"][0], accel_HR_upper=assoc["CI95"][1],
                   accel_SD_years=assoc["SD_years"])
        rows.append(row)
        for year, item in metrics["fixed_times"].items():
            for method in ["clock", "age_sex", "ridge"]:
                estimate = item.get(method, {})
                horizons.append(dict(cohort=name, horizon_years=int(year), method=method, supported=item["supported"],
                                     deaths_by_horizon=item["cases"], later_followup=item["controls"],
                                     censor_survival=item["G"], observed_death_risk=item.get("observed_km"),
                                     predicted_death_risk=estimate.get("predicted_mean"), auc=estimate.get("auc"),
                                     brier=estimate.get("brier")))
    save_json(OUTPUT / "verification.json", dict(frozen_model=sha(TRAIN_RUN / "model_freeze.json"), cohorts=checks))
    return pd.DataFrame(rows), pd.DataFrame(horizons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit participant alignment and summarize the requested cohorts.")
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    cohort_table, horizons = collect(args.cohorts)
    cohort_table.to_csv(OUTPUT / "cohorts.csv", index=False, encoding="utf-8-sig")
    horizons.to_csv(OUTPUT / "horizons.csv", index=False, encoding="utf-8-sig")
