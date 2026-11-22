"""Cross-check saved landmark labels and capture observation-limited rhythm metrics."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
MIMIC = RUN.parent
sys.path.insert(0, str(MIMIC))
from mimic_iv_panels import ORGAN_REFERENCE, digest, verify
from mimic_iv_run import write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    verify()
    pred = MIMIC / "formal_v3_1" / "predictions.csv.gz"
    report = read_json(RUN / "results" / "incident_evaluation.json")
    assert digest(pred) == report["frozen_predictions_sha256"]
    scores = pd.read_csv(pred, dtype={"ID": str})
    index = pd.read_csv(RUN / "cache" / "incident_index.csv.gz", dtype={"ID": str}, parse_dates=["landmark", "stop_time"])
    assert scores.ID.is_unique and index.ID.is_unique and set(scores.ID) == set(index.ID)
    index["available_days"] = (index.stop_time-index.landmark) / pd.Timedelta(days=1)
    index["available_days"] = index.available_days.clip(upper=7)
    checks = {}
    for row in report["primary"]:
        panel = row["panel"]
        file = RUN / "inputs" / f"{panel}.csv.gz"
        assert digest(file) == row["input_sha256"]
        d = pd.read_csv(file, dtype={"ID": str})
        assert d.ID.is_unique
        d = d.merge(index[["ID", "available_days"]], on="ID", validate="one_to_one").merge(
            scores[["ID", "age", "sex", "full_lp", "full_accel", f"{panel}_lp", f"{panel}_accel"]],
            on="ID", validate="one_to_one")
        assert len(d) == row["n"] and int(d.event.sum()) == row["events"]
        assert (d.time_days.le(d.available_days.add(1e-6)) & d.time_days.gt(0)).all()
        for variable, key in [("full_lp", "full_raw_auc"), (f"{panel}_lp", "organ_raw_auc")]:
            np.testing.assert_allclose(roc_auc_score(d.event, d[variable]), row[key], atol=1e-12)
        checks[panel] = {"n":len(d), "events":int(d.event.sum()), "earliest_event_days":float(d.loc[d.event.eq(1),"time_days"].min()),
                         "input_sha256":row["input_sha256"], "auc_recomputed":True}
        if panel == "cardiovascular":
            scales = read_json(ORGAN_REFERENCE)["panels"]
            designs = pd.DataFrame({"time":d.time_days, "event":d.event, "age10":(d.age-60)/10, "sex":d.sex,
                                    "full":d.full_accel/scales["full"]["accel_sd_years"],
                                    "cardiovascular":d.cardiovascular_accel/scales[panel]["accel_sd_years"]})
            designs["age10_squared"] = designs.age10**2
            with threadpool_limits(limits=2):
                fit = CoxPHFitter().fit(designs, "time", "event")
            g = KaplanMeierFitter().fit(d.time_days, d.end_status.isin(["monitor_end", "icu_exit"]))
            cardiac = {"by_end_of_observation_c_index_full":float(concordance_index(d.time_days,-d.full_lp,d.event)),
                       "by_end_of_observation_c_index_cardiovascular":float(concordance_index(d.time_days,-d.cardiovascular_lp,d.event)),
                       "adjusted_cause_specific_hr_per_10_points": float(np.exp(fit.params_["cardiovascular"])),
                       "hr_ci95": np.exp(fit.confidence_intervals_.loc["cardiovascular"]).tolist(),
                       "n_at_seven_days_without_event": int(d.time_days.eq(7).sum()),
                       "monitor_end_or_icu_exit_count": int(d.end_status.isin(["monitor_end","icu_exit"]).sum()),
                       "reverse_km_monitoring_survival": {str(i):float(g.predict(i)) for i in [1,2,3,7]},
                       "interpretation": "Follow-up stops at first documented AF/flutter, monitoring end, competing hospital event, or day 7. Informative monitoring loss remains possible."}
            write_json(RUN/"results"/"rhythm_monitoring_sensitivity.json",cardiac)
    snapshot = {name: digest(RUN/"results"/name) for name in
                ["incident_evaluation.json","incident_metrics.csv","incident_sensitivity.csv","rhythm_monitoring_sensitivity.json"]}
    write_json(RUN / "results" / "verification.json", {"parent_n":len(scores), "panels":checks, "files_sha256":snapshot})
    print("Verified event counts, timed follow-up, and independently recomputed fixed-score AUCs.", flush=True)


if __name__ == "__main__":
    main()
