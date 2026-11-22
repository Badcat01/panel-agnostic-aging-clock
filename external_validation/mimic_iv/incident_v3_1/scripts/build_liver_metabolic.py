"""Build timed post-landmark liver-enzyme and level-2 hypoglycemia outcomes."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
MIMIC = RUN.parent
sys.path.insert(0, str(MIMIC))
from mimic_iv_run import digest, write_csv, write_json

TYPES = {
    "liver": {"itemid": 50861, "units": ["iu/l", "u/l"], "source": "ALT",
              "baseline_clear": "all measured ALT through landmark ≤80 U/L",
              "event": "first ALT ≥200 U/L after landmark", "clinical_status": "biochemical signal"},
    "metabolic": {"itemid": 50931, "units": ["mg/dl"], "source": "Glucose",
                  "baseline_clear": "all glucose measured through landmark ≥70 mg/dL",
                  "event": "first glucose <54 mg/dL after landmark", "clinical_status": "documented ADA level-2 hypoglycemia"},
}


def frame_for(kind, index, labs, pred, source_hash):
    config = TYPES[kind]
    m = labs.loc[labs.itemid.eq(config["itemid"])].merge(
        index[["subject_id", "landmark"]], on="subject_id", validate="many_to_one")
    raw_rows = len(m)
    expected = m.valueuom.astype("string").str.strip().str.lower().isin(config["units"])
    physical = m.valuenum.gt(0) & np.isfinite(m.valuenum)
    v = m.loc[expected & physical].copy()
    pre = v.loc[v.charttime.le(v.landmark)]
    available = pre.loc[pre.charttime.lt(pre.landmark) & pre.storetime.ge(pre.charttime) & pre.storetime.le(pre.landmark)]
    ready = available.sort_values(["charttime", "storetime", "labevent_id"], kind="stable").drop_duplicates("subject_id", keep="last")
    if kind == "liver":
        excluded = pre.valuenum.gt(80)
    else:
        excluded = pre.valuenum.lt(70)
    baseline_clear = ~ready.subject_id.isin(pre.loc[excluded, "subject_id"])
    score_ids = pred.loc[pred[f"{kind}_lp"].notna(), "ID"]
    eligible_ids = ready.loc[baseline_clear & ready.subject_id.isin(score_ids), "subject_id"]
    members = index.loc[index.subject_id.isin(eligible_ids)].copy()
    members = members.join(ready.set_index("subject_id").valuenum.rename("baseline_value"), on="subject_id", validate="one_to_one")
    members["max_time"] = members.landmark + pd.Timedelta(days=7)
    members["endpoint_time"] = members[["stop_time", "max_time"]].min(axis=1)
    m = v.loc[v.charttime.gt(v.landmark) & v.subject_id.isin(eligible_ids)].copy()
    m = m.join(members.set_index("subject_id").endpoint_time, on="subject_id", validate="many_to_one")
    if not m.charttime.le(m.endpoint_time).all():
        raise ValueError("Serial cache contains a post-endpoint measurement")
    hit = m.loc[m.valuenum.ge(200) if kind == "liver" else m.valuenum.lt(54)].copy()
    first = hit.groupby("subject_id").charttime.min()
    members = members.join(first.rename("event_time"), on="subject_id", validate="one_to_one")
    event = members.event_time.notna()
    ends_in_stay = members.stop_time.le(members.max_time)
    status = np.select([ends_in_stay & members.hospital_event.eq(1), ends_in_stay], ["death", "discharge"], default="censor")
    members["end_status"] = np.where(event, "event", status)
    members["time_days"] = (members.event_time.fillna(members.endpoint_time) - members.landmark) / pd.Timedelta(days=1)
    members["event"] = event.astype(int)
    members["post_observed"] = members.subject_id.isin(m.subject_id)
    cols = ["ID", "event", "time_days", "end_status", "post_observed", "baseline_value"]
    extra = {}
    if kind == "liver":
        hits = hit.join(first.rename("first"), on="subject_id", validate="many_to_one")
        flags = hits.loc[hits.charttime.ge(hits["first"] + pd.Timedelta(days=1))].groupby("subject_id").charttime.min()
        confirmed = members.subject_id.isin(flags.index)
        confirmed_time = members.subject_id.map(flags).fillna(members.endpoint_time)
        members["confirmed_event"] = confirmed.astype(int)
        members["confirmed_time_days"] = (confirmed_time - members.landmark) / pd.Timedelta(days=1)
        members["confirmed_end_status"] = np.where(confirmed, "event", status)
        cols += ["confirmed_event", "confirmed_time_days", "confirmed_end_status"]
        extra["repeated_24h_events"] = int(confirmed.sum())
    assert members.ID.is_unique and members.time_days.gt(0).all() and members.time_days.le(7).all()
    info = {"definition": config, "index_n": len(index), "raw_item_rows": raw_rows,
            "bad_units_or_nonpositive": int((~(expected & physical)).sum()), "baseline_result_available_n": len(ready),
            "excluded_by_baseline_threshold": int((~baseline_clear).sum()), "included_n": len(members),
            "event_count": int(event.sum()), "post_observed_n": int(members.post_observed.sum()),
            "end_status_counts": members.end_status.value_counts().to_dict(),
            "serial_labs_sha256": source_hash, **extra}
    return members[cols], info


def main():
    source = RUN / "cache" / "serial_labs.csv.gz"
    cache = json.loads((RUN / "cache" / "serial_labs_audit.json").read_text(encoding="utf8"))
    checksum = digest(source)
    if checksum != cache["serial_labs_sha256"]:
        raise ValueError("Serial laboratory file changed")
    index = pd.read_csv(RUN / "cache" / "incident_index.csv.gz", parse_dates=["landmark", "stop_time"])
    release = MIMIC / "formal_v3_1" / "predictions.csv.gz"
    p = pd.read_csv(release, usecols=["ID", "liver_lp", "metabolic_lp"])
    labs = pd.read_csv(source, parse_dates=["charttime", "storetime"])
    for directory in ("inputs", "audit"):
        (RUN / directory).mkdir(exist_ok=True)
    for group in TYPES:
        frame, audit = frame_for(group, index, labs, p, checksum)
        dest = RUN / "inputs" / f"{group}.csv.gz"
        write_csv(dest, frame)
        write_json(RUN / "audit" / f"{group}.json", {**audit, "input_sha256": digest(dest)})
        print(group, audit["included_n"], audit["event_count"], audit["post_observed_n"], flush=True)


if __name__ == "__main__":
    main()
