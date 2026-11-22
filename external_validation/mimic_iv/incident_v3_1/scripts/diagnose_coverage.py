"""Describe baseline marker availability and follow-up ascertainment in each endpoint roster."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[2]
FORMAL = RUN.parent / "formal_v3_1"
TRAIN = PROJECT / "runs" / "nhanes_gpu_20261121"
sys.path[:0] = [str(RUN.parent), str(TRAIN / "bundle")]
from mimic_iv_validate import digest
from nhanes_calibration import prepare_table


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def check_file(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Audit checksum mismatch for {path}")
    return path


def main():
    report = read_json(RUN / "results" / "incident_evaluation.json")
    freeze = read_json(check_file(TRAIN / "model_freeze.json", report["model_freeze_sha256"]))
    for filename in ["clock_bundle.json", "nhanes_calibration.py"]:
        check_file(TRAIN / "bundle" / filename, freeze["files_sha256"][filename])
    groups = read_json(TRAIN / "bundle" / "clock_bundle.json")["groups"]
    roster = pd.read_csv(TRAIN / "development_ids.csv.gz")
    original = read_json(TRAIN / "protocol.json")["train_ids_sha256"]
    if hashlib.sha256(roster.SEQN.to_numpy(dtype="<i8").tobytes()).hexdigest() != original:
        raise ValueError("Training subject roster mismatch")
    calibrated, _, _ = prepare_table(PROJECT)
    dev = roster.merge(calibrated, on=["SEQN", "cycle"], validate="one_to_one")
    assert len(roster) == len(dev)

    audit = read_json(FORMAL / "audit.json")
    idx = pd.read_csv(check_file(FORMAL / "landmark_index.csv.gz", audit["index_sha256"]),
                      usecols=["subject_id", "hadm_id"])
    labs = pd.read_csv(check_file(FORMAL / "lab_panel.csv.gz", audit["lab_audit"]["lab_panel_sha256"]))
    vitals = pd.read_csv(check_file(FORMAL / "vitals.csv.gz", audit["vitals_sha256"]))
    baseline = idx.merge(labs, on="hadm_id", validate="one_to_one").merge(
        vitals, how="left", on=["subject_id", "hadm_id"], validate="one_to_one").set_index("subject_id")
    scores = pd.read_csv(check_file(FORMAL / "predictions.csv.gz", report["frozen_predictions_sha256"]),
                         index_col="ID")
    assert baseline.index.is_unique and scores.index.is_unique and set(baseline.index) == set(scores.index)
    rows = []
    for result in report["primary"]:
        group = result["panel"]
        endpoint = pd.read_csv(check_file(RUN / "inputs" / f"{group}.csv.gz", result["input_sha256"]),
                               index_col="ID")
        assert endpoint.index.is_unique and len(endpoint) == result["n"] and endpoint.event.sum() == result["events"]
        feature_names = groups[group]
        markers = baseline.loc[endpoint.index].reindex(columns=feature_names).notna()
        available = markers.sum(axis=1)
        expected = scores.loc[endpoint.index, f"{group}_n_observed"]
        assert available.gt(0).all() and available.eq(expected).all()
        singles = available.eq(1)
        reference = dev[feature_names].notna().sum(axis=1)
        column = "postlab_observed" if group == "kidney" else "post_observed"
        observed = endpoint[column].astype("string").str.lower().map({"true": True, "false": False, "0": False, "1": True})
        assert observed.notna().all() and observed.sum() == result["post_observed_n"]
        quintile = pd.qcut(scores.loc[endpoint.index, f"{group}_accel"], 5, labels=False)
        assert set(quintile) == set(range(5))
        counts = markers.sum().sort_values(ascending=False)
        sole_counts = markers[singles].sum().sort_values(ascending=False)
        rows.append({
            "panel": group, "n": len(endpoint), "events": result["events"], "panel_size": len(feature_names),
            "nhanes_development_n": len(dev), "nhanes_development_observed_median": reference.median(),
            "baseline_q25": available.quantile(.25), "baseline_median": available.median(),
            "baseline_q75": available.quantile(.75), "baseline_single_n": int(singles.sum()),
            "baseline_single_fraction": singles.mean(), "baseline_complete_n": int(available.eq(len(feature_names)).sum()),
            "marker_n": json.dumps(counts.to_dict()), "sole_marker_n": json.dumps(sole_counts[sole_counts.gt(0)].to_dict()),
            "post_observed_n": int(observed.sum()), "post_observed_fraction": observed.mean(),
            "post_observed_n_in_lowest_score_quintile": int(observed[quintile.eq(0)].sum()),
            "lowest_score_quintile_n": int(quintile.eq(0).sum()),
            "post_observed_fraction_lowest_score_quintile": observed[quintile.eq(0)].mean(),
            "post_observed_n_in_highest_score_quintile": int(observed[quintile.eq(4)].sum()),
            "highest_score_quintile_n": int(quintile.eq(4).sum()),
            "post_observed_fraction_highest_score_quintile": observed[quintile.eq(4)].mean(),
            "median_days_to_event_observation_end_or_day7": endpoint.time_days.median(),
            "primary_input_sha256": result["input_sha256"],
        })
    target = RUN / "results" / "diagnostic_coverage.csv"
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)
    print(pd.DataFrame(rows)[["panel", "n", "nhanes_development_observed_median", "baseline_median",
                              "baseline_single_n", "post_observed_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
