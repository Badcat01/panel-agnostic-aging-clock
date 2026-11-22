"""Explore whether admission-day creatinine change adds to the static AKI baseline."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_kidney import select_creatinines
from evaluate_incident import MIMIC, ORGAN_REFERENCE, RUN, digest, oof_comparison, read_json, write_json


def main():
    source = RUN / "cache"
    results = RUN / "results"
    kidney_outcomes = RUN / "inputs" / "kidney.csv.gz"
    score_path = MIMIC / "formal_v3_1" / "predictions.csv.gz"
    index_path = source / "incident_index.csv.gz"
    lab_path = source / "serial_labs.csv.gz"
    audit = read_json(RUN / "audit" / "kidney.json")
    hashes = audit["source_sha256"]
    for path, sha in [(kidney_outcomes, audit["input_sha256"]), (lab_path, hashes["serial_labs"]),
                      (index_path, hashes["index"]), (score_path, hashes["verified_frozen_scores"])]:
        if digest(path) != sha:
            raise ValueError(f"Incident analysis input hash differs: {path.name}")
    outcomes = pd.read_csv(kidney_outcomes, dtype={"ID": str})
    index = pd.read_csv(index_path, dtype={"ID": str}, parse_dates=["admittime", "landmark", "stop_time"])
    cohort = outcomes[["ID"]].merge(index, on="ID", how="left", validate="one_to_one")
    if cohort.hadm_id.isna().any() or not cohort.hadm_id.is_unique:
        raise ValueError("Missing or repeated admission")

    labs, _ = select_creatinines(lab_path, cohort)
    joined = labs.merge(cohort[["ID", "hadm_id", "admittime", "landmark"]], on="hadm_id",
                        validate="many_to_one")
    available = joined.loc[joined.charttime.ge(joined.admittime) & joined.charttime.lt(joined.landmark)
                           & joined.storetime.le(joined.landmark)].sort_values(["ID", "charttime"])
    if available.duplicated(["ID", "charttime"]).any():
        raise ValueError("Repeated specimen times after source reconciliation")
    trends = available.groupby("ID", as_index=False).creatinine.agg(
        first_available="first", last_available="last", n_times="size")
    scores = pd.read_csv(score_path, dtype={"ID": str}, usecols=["ID", "age", "sex", "full_accel"])
    frame = outcomes.merge(trends, on="ID", how="left", validate="one_to_one").merge(
        scores, on="ID", how="left", validate="one_to_one")
    np.testing.assert_array_equal(frame.ID, outcomes.ID)
    np.testing.assert_array_equal(frame.n_times, frame.baseline_creatinine_n)
    np.testing.assert_allclose(frame.last_available, frame.baseline_value, rtol=0, atol=1e-12)
    if frame.n_times.min() < 2 or not frame[["age", "sex", "full_accel", "first_available"]].notna().all().all():
        raise ValueError("Incomplete admission-day predictors")

    age = (frame.age.to_numpy() - 60) / 10
    log_cr = np.log(frame.baseline_value)
    reference = read_json(ORGAN_REFERENCE)
    if reference["model_freeze_sha256"] != read_json(MIMIC / "formal_v3_1" / "audit.json")["model_freeze_sha256"]:
        raise ValueError("Organ reference and frozen model differ")
    full = frame.full_accel / reference["panels"]["full"]["accel_sd_years"]
    base = np.column_stack([age, age**2, frame.sex, log_cr, log_cr**2, np.log(frame.n_times), full])
    signed_change = (frame.last_available - frame.first_available).to_numpy()
    change = np.column_stack([signed_change, np.abs(signed_change)])
    with threadpool_limits(limits=2):
        comparison = oof_comparison(base, change, frame.event.to_numpy(dtype=bool), seed=20261121)
    prior = read_json(results / "kidney_additional_sensitivity.json")["results"]["full_eligible_roster"]
    np.testing.assert_array_equal([len(frame), frame.event.sum()], [prior["n"], prior["events"]])
    np.testing.assert_allclose(comparison["baseline_auc_oof"], prior["baseline_auc_oof"], rtol=0, atol=1e-12)
    write_json(results / "diagnostic_kidney_trend.json", {
        "endpoint": "First creatinine-defined AKI within seven days after hour 24, while still hospitalized",
        "analysis": "Post hoc exploratory comparison on the primary AKI risk set and original five folds",
        "n": len(frame), "events": int(frame.event.sum()), "endpoint_sha256": audit["input_sha256"],
        "labs_sha256": hashes["serial_labs"], "prediction_sha256": hashes["verified_frozen_scores"],
        "eligibility_and_timing": "Identical eligibility and outcome to the primary analysis. Creatinine "
                                  "specimens collected from admission through strictly before hour 24 with "
                                  "results stored by hour 24. Original analyte, unit and same-specimen rules.",
        "base_predictors": ["age", "age squared", "sex", "log latest creatinine", "squared log latest creatinine",
                            "log number of measured times", "frozen full-body acceleration"],
        "two_additional_predictors": ["last minus first available creatinine (mg/dL)", "absolute change (mg/dL)"],
        "baseline_last_and_count_match_original_for_all": True,
        "increases": int((signed_change > 0).sum()), "ties": int((signed_change == 0).sum()),
        "decreases": int((signed_change < 0).sum()), "fold_seed": 20261121,
        "limits": "DeLong CI conditional on the fitted folds, exploratory, without adjustment for multiple "
                  "diagnostics. Creatinine is also the analyte defining this laboratory outcome.",
        "paired_oof": comparison,
    })


if __name__ == "__main__":
    main()
