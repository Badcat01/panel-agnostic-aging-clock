"""Assess organ score overlap on the same patients and folds as the incident study."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from evaluate_incident import ENDPOINTS, MIMIC, ORGAN_REFERENCE, RUN, digest, oof_comparison, read_json


def main():
    seed = 20261121
    directory = RUN / "results"
    primary = pd.read_csv(directory / "incident_metrics.csv").set_index("panel")
    evaluation = read_json(directory / "incident_evaluation.json")
    reference = read_json(ORGAN_REFERENCE)["panels"]
    prediction_file = MIMIC / "formal_v3_1" / "predictions.csv.gz"
    if digest(prediction_file) != evaluation["frozen_predictions_sha256"]:
        raise ValueError("Frozen prediction checksum differs from the incident study")
    columns = ["ID", "age", "sex", "full_accel", *(f"{p}_accel" for p in ENDPOINTS)]
    scores = pd.read_csv(prediction_file, usecols=columns, dtype={"ID": str})
    rows = []
    with threadpool_limits(limits=2):
        for panel, endpoint in ENDPOINTS.items():
            path = RUN / "inputs" / f"{panel}.csv.gz"
            expected = primary.loc[panel]
            if digest(path) != expected.input_sha256:
                raise ValueError(f"Endpoint file differs from the primary analysis for {panel}")
            outcomes = pd.read_csv(path, usecols=["ID", "event"], dtype={"ID": str})
            data = outcomes.merge(scores, on="ID", how="left", validate="one_to_one", sort=False)
            age = (data.age.to_numpy() - 60) / 10
            demographics = np.column_stack([age, age**2, data.sex])
            organ = data[f"{panel}_accel"].to_numpy() / reference[panel]["accel_sd_years"]
            full = data.full_accel.to_numpy() / reference["full"]["accel_sd_years"]
            event = data.event.to_numpy(dtype=bool)
            np.testing.assert_array_equal([len(event), event.sum()], [expected.n, expected.events])
            organ_vs_demo = oof_comparison(demographics, organ, event, seed)
            full_vs_demo = oof_comparison(demographics, full, event, seed)
            both_vs_full = oof_comparison(np.column_stack([demographics, full]), organ, event, seed)
            np.testing.assert_allclose(
                [organ_vs_demo["baseline_auc_oof"], full_vs_demo["augmented_auc_oof"],
                 both_vs_full["baseline_auc_oof"], both_vs_full["augmented_auc_oof"]],
                [full_vs_demo["baseline_auc_oof"], expected.baseline_auc_oof,
                 expected.baseline_auc_oof, expected.augmented_auc_oof], rtol=0, atol=1e-12,
            )
            rows.append({
                "panel": panel, "endpoint": endpoint, "n": len(event), "events": int(event.sum()),
                "spearman_organ_full_accel": spearmanr(organ, full).statistic,
                "auc_demographics": organ_vs_demo["baseline_auc_oof"],
                "auc_demographics_organ": organ_vs_demo["augmented_auc_oof"],
                "auc_demographics_full": both_vs_full["baseline_auc_oof"],
                "auc_demographics_full_organ": both_vs_full["augmented_auc_oof"],
                "delta_organ_after_demographics": organ_vs_demo["incremental_auc_oof"],
                "delta_organ_after_demographics_lower": organ_vs_demo["incremental_auc_oof_ci95"][0],
                "delta_organ_after_demographics_upper": organ_vs_demo["incremental_auc_oof_ci95"][1],
                "delta_organ_after_demographics_full": both_vs_full["incremental_auc_oof"],
                "delta_organ_after_demographics_full_lower": both_vs_full["incremental_auc_oof_ci95"][0],
                "delta_organ_after_demographics_full_upper": both_vs_full["incremental_auc_oof_ci95"][1],
            })
    output = directory / "diagnostic_overlap.csv"
    staged = output.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(staged, index=False, encoding="utf-8-sig")
    staged.replace(output)


if __name__ == "__main__":
    main()
