"""Build post-landmark first-charted AF/flutter in the first admission's monitored ICU stay."""

import argparse
import io
import json
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from study_paths import MIMIC_ARCHIVE
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as arrow_csv

INCIDENT = Path(__file__).resolve().parents[1]
MIMIC = INCIDENT.parent
sys.path.insert(0, str(MIMIC))
from mimic_iv_run import write_csv, write_json
from mimic_iv_vitals import HashedStream, PREFIX, file_hash, read_verified, source_manifest


RHYTHM_ID = 220048
COLS = ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "storetime", "value"]
KEYS = ["subject_id", "hadm_id", "stay_id"]
PAIR_SEPARATION = pd.Timedelta(minutes=10)
PAIR_WINDOW = pd.Timedelta(hours=12)
FOLLOW_UP_DAYS = 7
AF_LABELS = {"af (atrial fibrillation)", "a flut (atrial flutter)"}
OTHER_RHYTHMS = {
    "sr (sinus rhythm)", "st (sinus tachycardia)", "sb (sinus bradycardia)", "sa (sinus arrhythmia)",
    "1st av (first degree av block)", "2nd av w-m1 (second degree av block wenckebach - mobitz1)",
    "2nd av m2 (second degree av block - mobitz 2)", "3rd av (complete heart block)",
    "v paced", "a paced", "av paced", "rbbb (right bundle branch block)", "lbbb (left bundle branch block)",
    "jr (junctional rhythm)", "jt (junctional tachycardia)", "pjt (paroxysmal junctional tachycardia)",
    "svt (supra ventricular tachycardia)", "vt (ventricular tachycardia)", "vf (ventricular fibrillation)",
    "wap (wandering atrial pacemaker)", "mat (multifocal atrial tachycardia)",
    "pat (paroxysmal atrial tachycardia)", "idioventricular", "asystole",
}


def nested_rows(archive, manifest, stay_ids):
    member = "icu/chartevents.csv.gz"
    member_size = archive.getinfo(PREFIX + member).file_size
    row_count = 0
    results = []
    types = {name: pa.int64() for name in COLS[:4]}
    types.update(charttime=pa.timestamp("s"), storetime=pa.timestamp("s"), value=pa.string())
    with archive.open(PREFIX + member) as source:
        hashed = HashedStream(source)
        native = pa.PythonFile(io.BufferedReader(hashed, 1 << 20), mode="r")
        compressed = pa.CompressedInputStream(native, "gzip")
        reader = arrow_csv.open_csv(
            compressed,
            read_options=arrow_csv.ReadOptions(block_size=1 << 26),
            parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
            convert_options=arrow_csv.ConvertOptions(include_columns=COLS, column_types=types,
                                                      strings_can_be_null=True),
        )
        for num, records in enumerate(reader, start=1):
            row_count += records.num_rows
            keep = pc.and_(pc.equal(records["itemid"], RHYTHM_ID), pc.is_in(records["stay_id"], value_set=stay_ids))
            rows = records.filter(keep)
            if rows.num_rows:
                results.append(rows.to_pandas())
            if num == 1 or num % 20 == 0:
                print(f"rhythm: {row_count:,} rows scanned, {hashed.bytes_read / member_size:.1%} compressed read", flush=True)
        reader.close()
        if hashed.bytes_read != member_size or hashed.digest.hexdigest() != manifest[member]:
            raise ValueError("ICU source failed archive hash check")
    if not results:
        raise ValueError("No charted rhythms in eligible ICU stays")
    return pd.concat(results, ignore_index=True), row_count


def load_cohort():
    formal = MIMIC / "formal_v3_1"
    index_path = formal / "landmark_index.csv.gz"
    score_path = formal / "predictions.csv.gz"
    audit = json.loads((formal / "audit.json").read_text(encoding="utf8"))
    if file_hash(index_path) != audit["index_sha256"] or file_hash(score_path) != audit["predictions_sha256"]:
        raise ValueError("Landmark roster or frozen predictions failed hash check")
    population = pd.read_csv(index_path, parse_dates=["admittime", "landmark"])
    scores = pd.read_csv(score_path, usecols=["ID", "cardiovascular_n_observed", "cardiovascular_lp", "full_lp"])
    scores = scores.rename(columns={"ID": "subject_id"})
    if not scores.cardiovascular_lp.notna().equals(scores.cardiovascular_n_observed.gt(0)):
        raise ValueError("Frozen cardiovascular score does not match organ feature availability")
    d = population.merge(scores, on="subject_id", validate="one_to_one")
    d = d.loc[d.cardiovascular_lp.notna() & d.full_lp.notna()].copy()
    if d.empty or not d.landmark.sub(d.admittime).eq(pd.Timedelta(hours=24)).all():
        raise ValueError("Invalid landmark cohort")
    return d, audit


def raw_rhythms(archive_path):
    cohort, audit = load_cohort()
    with ZipFile(archive_path) as source:
        manifest = source_manifest(source)
        dictionary = read_verified(source, manifest, "icu/d_items.csv.gz")
        definition = dictionary.set_index("itemid", verify_integrity=True).loc[RHYTHM_ID]
        if definition.label != "Heart Rhythm" or definition.linksto != "chartevents":
            raise ValueError("Unexpected heart rhythm item in v3.1 dictionary")
        icu = read_verified(source, manifest, "icu/icustays.csv.gz", usecols=[*KEYS, "intime", "outtime"],
                            parse_dates=["intime", "outtime"])
        icu = icu.merge(cohort[["subject_id", "hadm_id", "admittime", "landmark"]], on=KEYS[:2], validate="many_to_one")
        icu = icu.loc[icu.outtime.gt(icu.intime) & icu.outtime.gt(icu.admittime)]
        active = icu.loc[icu.intime.lt(icu.landmark) & icu.outtime.gt(icu.landmark)]
        duplicate_active_ids = active.loc[active.subject_id.duplicated(keep=False), "subject_id"].unique()
        active = active.loc[~active.subject_id.isin(duplicate_active_ids)]
        stay_subset = icu.loc[icu.subject_id.isin(active.subject_id) &
                              (icu.intime.lt(icu.landmark) | icu.stay_id.isin(active.stay_id))]
        rows, row_count = nested_rows(source, manifest, pa.array(stay_subset.stay_id, type=pa.int64()))
        members = {name: manifest[name] for name in ("icu/d_items.csv.gz", "icu/icustays.csv.gz", "icu/chartevents.csv.gz")}
    rows = rows.merge(stay_subset, on=KEYS, validate="many_to_one")
    temporal = (rows.charttime.ge(rows.intime) & rows.charttime.le(rows.outtime) &
                rows.charttime.ge(rows.admittime) & rows.storetime.ge(rows.charttime))
    rows = rows.loc[temporal].copy()
    mapping = dict(index_sha256=audit["index_sha256"], predictions_sha256=audit["predictions_sha256"],
                   member_sha256=members, scored_with_cardiovascular=len(cohort),
                   excluded_multiple_active_icu_stays=len(duplicate_active_ids),
                   active_stay_at_landmark=len(active), source_rows_scanned=row_count)
    return rows, active, cohort, mapping


def classify(values):
    clean = values.astype("string").str.strip().str.casefold()
    af = clean.isin(AF_LABELS)
    alternate = clean.isin(OTHER_RHYTHMS)
    state = pd.Series(0, index=values.index, dtype="int8")
    state.loc[af] = 1
    state.loc[alternate] = -1
    return state


def confirmed_events(post):
    """Second distinct chart timestamp agrees within 10 minutes to 12 hours."""
    confirmed = {}
    mixed = 0
    positive = post.loc[post.state.eq(1), "subject_id"].unique()
    observations = post.loc[post.subject_id.isin(positive)]
    for patient, records in observations.groupby("subject_id", sort=False):
        at_time = records.groupby("charttime").state.agg(["min", "max"]).sort_index()
        run = deque()
        for when, codes in at_time.iterrows():
            if codes["min"] != codes["max"]:
                mixed += 1
            if codes["min"] != 1 or codes["max"] != 1:
                run.clear()
                continue
            while run and when - run[0] > PAIR_WINDOW:
                run.popleft()
            if run and when - run[0] >= PAIR_SEPARATION:
                confirmed[patient] = when
                break
            run.append(when)
    return pd.Series(confirmed, name="confirmed_time", dtype="datetime64[ns]"), len(positive), mixed


def finish(rows, active, mapping):
    cohort, formal_audit = load_cohort()
    if mapping["index_sha256"] != formal_audit["index_sha256"] or mapping["predictions_sha256"] != formal_audit["predictions_sha256"]:
        raise ValueError("Cached rhythm rows came from a different formal MIMIC validation run")
    if not active.subject_id.is_unique:
        raise ValueError("The ICU monitor stay at landmark must be unique")
    timeline_path = INCIDENT / "cache" / "incident_index.csv.gz"
    columns = ["subject_id", "hadm_id", "landmark", "stop_time", "hospital_event"]
    timeline = pd.read_csv(timeline_path, usecols=columns, parse_dates=["landmark", "stop_time"])
    if not timeline.subject_id.is_unique or not timeline.stop_time.gt(timeline.landmark).all():
        raise ValueError("Invalid shared incident follow-up index")
    a = active[KEYS + ["intime", "outtime"]].merge(timeline, on=KEYS[:2], validate="one_to_one")
    if len(a) != len(active) or not set(a.subject_id).issubset(cohort.subject_id):
        raise ValueError("Monitored ICU stays do not all map to the shared incident index")
    a["cap"] = a.landmark + pd.Timedelta(days=FOLLOW_UP_DAYS)
    a["observational_cap"] = a[["stop_time", "outtime", "cap"]].min(axis=1)
    rows["state"] = classify(rows.value)
    baseline_af = set(rows.loc[rows.charttime.le(rows.landmark) & rows.state.eq(1), "subject_id"])
    within_active = rows.merge(a[KEYS + ["observational_cap"]], on=KEYS, validate="many_to_one")
    baseline_candidates = (within_active.state.eq(-1) &
                           within_active.charttime.between(within_active.landmark - PAIR_WINDOW, within_active.landmark) &
                           within_active.storetime.le(within_active.landmark))
    baseline_non_af = set(within_active.loc[baseline_candidates, "subject_id"])
    baseline_sinus = set(within_active.loc[baseline_candidates &
                         within_active.value.astype("string").str.contains("sinus|^(?:sr|sb|st|nsr)$", case=False, na=False),
                         "subject_id"])
    eligible = a.loc[~a.subject_id.isin(baseline_af) & a.subject_id.isin(baseline_non_af)].copy()
    later = within_active.loc[within_active.subject_id.isin(eligible.subject_id) &
                             within_active.charttime.gt(within_active.landmark) &
                             within_active.charttime.le(within_active.observational_cap)].copy()
    observed = later.loc[later.state.ne(0)]
    latest = observed.groupby("subject_id").charttime.max().rename("last_known_rhythm")
    eligible = eligible.join(latest, on="subject_id", validate="one_to_one")
    has_time = eligible.last_known_rhythm.gt(eligible.landmark)
    post = later.loc[later.subject_id.isin(eligible.loc[has_time, "subject_id"]) & later.state.ne(0)]
    events, positive, mixed = confirmed_events(post)
    frame = eligible.loc[has_time].copy().join(events, on="subject_id", validate="one_to_one")
    frame["post_observed"] = True
    frame["baseline_sinus"] = frame.subject_id.isin(baseline_sinus)
    out_end = frame[["last_known_rhythm", "outtime", "cap", "stop_time"]].min(axis=1)
    case = frame.confirmed_time.notna() & frame.confirmed_time.le(out_end)
    hospital_end = out_end.eq(frame.stop_time)
    frame["end_status"] = np.select(
        [case, hospital_end & frame.hospital_event.eq(1), hospital_end, out_end.eq(frame.cap), out_end.eq(frame.outtime)],
        ["event", "death", "discharge", "censor", "icu_exit"],
        default="monitor_end",
    )
    frame["event"] = case.astype(int)
    frame["time_days"] = (out_end.where(~case, frame.confirmed_time) - frame.landmark) / pd.Timedelta(days=1)
    frame["n_post_rhythm_records"] = frame.subject_id.map(observed.groupby("subject_id").charttime.nunique())
    result = frame.rename(columns={"subject_id": "ID"})[
        ["ID", "event", "time_days", "end_status", "post_observed", "baseline_sinus", "n_post_rhythm_records"]
    ].sort_values("ID")
    if not result.ID.is_unique or not result.time_days.gt(0).all() or not result.time_days.le(FOLLOW_UP_DAYS).all():
        raise ValueError("Endpoint has duplicate IDs or follow-up outside (0, 7] days")
    if not result.event.eq(result.end_status.eq("event").astype(int)).all():
        raise ValueError("Event and event status disagree")

    audit = {**mapping,
             "endpoint": "First incident ICU chart-recorded AF/flutter observed after the 24h hospital landmark, confirmed by two distinct chart timestamps.",
             "definition": "Item 220048 Heart Rhythm. An eligible active index ICU stay spans the 24-hour landmark. "
                           "Require an explicitly classified non-AF/flutter rhythm in the last 12h, charted and stored by the landmark in that stay. "
                           "Exclude any AF/flutter charted at or before the landmark in any observed ICU stay of the index admission. "
                           "The case-defining pair is at least 10 minutes and at most 12h apart without a recorded non-AF or conflicting time between; "
                           "the second timestamp is the event time. Unclassified records are ignored. No post-discharge ICD was used.",
             "incident_index_sha256": file_hash(timeline_path),
             "followup": "Days from the landmark to the first confirmed event, end of observation (last classifiable rhythm, ICU exit, or the earlier of in-hospital death and discharge), or 7 days. "
                         "Positive follow-up requires a subsequent classifiable rhythm. Monitoring loss is flagged in end_status.",
             "clinical_scope": "New chart-recorded episode, may be recurrent AF with a history outside the observed preceding window. "
                               "No reviewed ECG tracings. ICU subset and electronic observation frequencies limit generalizability.",
             "references": ["https://mimic.mit.edu/docs/iv/modules/icu/chartevents/",
                            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11381622/",
                            "https://www.jacc.org/doi/10.1016/j.jacc.2023.08.017"],
             "rhythms_analyzable_rows": len(rows),
             "recognized_or_unclassifiable_labels": [
                 {"value": str(value), "state": int(c), "n_rows": int(n)}
                 for (value, c), n in rows.groupby(["value", "state"], dropna=False).size().items()
             ],
             "prevalent_af_flutter": len(baseline_af),
             "excluded_no_recent_documented_non_af": int((~a.subject_id.isin(baseline_af) & ~a.subject_id.isin(baseline_non_af)).sum()),
             "eligible_pre_landmark": len(eligible),
             "n_with_postlandmark_monitoring": int(has_time.sum()),
             "excluded_no_post_landmark_rhythm_by_end_of_window": int((~has_time).sum()),
             "any_postlandmark_af_flutter": positive,
             "conflicting_post_rhythm_timestamps_examined": mixed,
             "confirmed_af_flutter": int(result.event.sum()),
             "n_baseline_sinus": int(result.baseline_sinus.sum()),
             "followup_median_days": float(result.time_days.median()),
             "status_counts": {k: int(v) for k, v in result.end_status.value_counts().items()}}
    out_path = INCIDENT / "inputs" / "cardiovascular.csv.gz"
    write_csv(out_path, result)
    audit["input_sha256"] = file_hash(out_path)
    write_json(INCIDENT / "audit" / "cardiovascular.json", audit)
    print(f"Eligible {len(eligible):,}; monitored analytic cohort {len(result):,}, confirmable AF/flutter {result.event.sum():,}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=MIMIC_ARCHIVE)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--scan-only", action="store_true")
    stages.add_argument("--use-cached-scan", action="store_true")
    args = parser.parse_args()
    out = INCIDENT / "inputs"
    audit_dir = INCIDENT / "audit"
    out.mkdir(exist_ok=True, parents=True)
    audit_dir.mkdir(exist_ok=True, parents=True)
    rhythm_path, stay_path = out / "cardio_rhythm_intermediate.csv.gz", out / "cardio_stay_intermediate.csv.gz"
    if args.use_cached_scan:
        rows = pd.read_csv(rhythm_path, parse_dates=["charttime", "storetime", "intime", "outtime", "landmark", "admittime"])
        active = pd.read_csv(stay_path, parse_dates=["intime", "outtime"])
        mapping = json.loads((audit_dir / "cardio_scan.json").read_text(encoding="utf8"))
    else:
        rows, active, _, mapping = raw_rhythms(args.archive)
        write_csv(rhythm_path, rows)
        write_csv(stay_path, active)
        write_json(audit_dir / "cardio_scan.json", mapping)
        print("Rhythm values", rows.value.value_counts().to_dict(), flush=True)
    if not args.scan_only:
        finish(rows, active, mapping)


if __name__ == "__main__":
    main()
