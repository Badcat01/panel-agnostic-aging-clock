"""Build the seven-day creatinine-defined AKI endpoint in MIMIC-IV v3.1."""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from study_paths import MIMIC_ARCHIVE
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT.parent / "formal_v3_1"
KEYS = ["subject_id", "hadm_id"]
CREATININE = 50912
DAY_NS = 86_400_000_000_000
HORIZON = 7
KDIGO = "https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf"
MIT_KDIGO = "https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/organfailure/kdigo_stages.sql"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            checksum.update(block)
    return checksum.hexdigest()


def dialysis_flags(archive_path, cohort):
    """Identify retrospective chronic ESRD statuses and dated early RRT."""
    with ZipFile(archive_path) as archive:
        prefix = "mimic-iv-3.1/"
        lines = archive.read(prefix + "SHA256SUMS.txt").decode("ascii").splitlines()
        declared = {path.lstrip("*"): sha for sha, path in (row.split(maxsplit=1) for row in lines if row.strip())}
        verified = {}

        def load(table, columns, code):
            member = f"hosp/{table}.csv.gz"
            compressed = archive.read(prefix + member)
            checksum = hashlib.sha256(compressed).hexdigest()
            if checksum != declared[member]:
                raise ValueError(f"Archive member differs from its declared hash: {member}")
            verified[member] = checksum
            return pd.read_csv(io.BytesIO(compressed), compression="gzip", usecols=KEYS + columns, dtype={code: str})

        diagnoses = load("diagnoses_icd", ["icd_code", "icd_version"], "icd_code")
        procedures = load("procedures_icd", ["icd_code", "icd_version", "chartdate"], "icd_code")
        hcpcs = load("hcpcsevents", ["hcpcs_cd", "chartdate"], "hcpcs_cd")
    diagnosis = diagnoses.icd_code.str.strip().str.upper()
    esrd = ((diagnoses.icd_version.eq(9) & diagnosis.isin({"5856", "V4511"}))
            | (diagnoses.icd_version.eq(10) & diagnosis.isin({"N186", "Z992"})))
    chronic = cohort[KEYS].merge(diagnoses.loc[esrd, KEYS].drop_duplicates(), on=KEYS,
                                 how="inner", validate="one_to_one")
    code = procedures.icd_code.str.strip().str.upper()
    filtration = ((procedures.icd_version.eq(9) & code.isin({"3995", "5498"}))
                  | (procedures.icd_version.eq(10) & code.str.match(r"^5A1D[06789]0Z$", na=False)))
    billing = hcpcs.hcpcs_cd.str.strip().str.upper().isin({"90935", "90937", "90945", "90947", "90999", "G0257"})
    columns = KEYS + ["chartdate"]
    dates = pd.concat([procedures.loc[filtration, columns], hcpcs.loc[billing, columns]], ignore_index=True)
    timed = dates.merge(cohort[KEYS + ["admittime", "landmark"]], on=KEYS, validate="many_to_one")
    day = pd.to_datetime(timed.chartdate, errors="raise")
    # Dates on the admission day necessarily precede a 24-hour landmark.
    known_early = day.ge(timed.admittime.dt.normalize()) & day.lt(timed.landmark.dt.normalize())
    return set(chronic.hadm_id), set(timed.loc[known_early, "hadm_id"]), verified


def select_creatinines(path, cohort):
    columns = KEYS + ["itemid", "charttime", "storetime", "valuenum", "valueuom"]
    index = cohort[KEYS + ["admittime", "landmark", "stop_time"]]
    chunks = []
    candidates = 0
    for part in pd.read_csv(path, usecols=columns, chunksize=500_000):
        is_creat = part.itemid.eq(CREATININE) & part.hadm_id.isin(cohort.hadm_id)
        if not is_creat.any():
            continue
        raw = part.loc[is_creat, columns].merge(index, on=KEYS, how="inner", validate="many_to_one")
        candidates += len(raw)
        units = raw.valueuom.astype("string").str.strip().str.casefold()
        raw["charttime"] = pd.to_datetime(raw.charttime, errors="coerce")
        raw["storetime"] = pd.to_datetime(raw.storetime, errors="coerce")
        raw["valuenum"] = pd.to_numeric(raw.valuenum, errors="coerce")
        valid = (units.eq("mg/dl") & raw.valuenum.gt(0) & np.isfinite(raw.valuenum)
                 & raw.charttime.ge(raw.admittime - pd.Timedelta(days=7))
                 & raw.charttime.lt(raw.stop_time)
                 & raw.charttime.le(raw.landmark + pd.Timedelta(days=HORIZON))
                 & raw.storetime.ge(raw.charttime))
        chunks.append(raw.loc[valid, ["hadm_id", "charttime", "storetime", "valuenum"]])
    if not chunks:
        raise ValueError("No creatinine records match the landmark cohort")
    raw = pd.concat(chunks, ignore_index=True)
    # Use identical specimen times once, keeping the first available result.
    timestamps = ["hadm_id", "charttime"]
    earliest = raw.groupby(timestamps).storetime.transform("min")
    reduced = (raw.loc[raw.storetime.eq(earliest)].groupby(timestamps, as_index=False)
               .agg(storetime=("storetime", "first"), creatinine=("valuenum", "median")))
    return reduced, {"matched_creatinine_rows": candidates, "valid_creatinine_rows": len(raw),
                     "unique_sampling_times": len(reduced), "invalid_or_outside_window_rows": candidates - len(raw)}


@njit
def aki_flags(times, values):
    flags = np.zeros(len(times), dtype=np.uint8)
    for j in range(1, len(times)):
        minimum_48h = minimum_7d = np.inf
        for i in range(j - 1, -1, -1):
            span = times[j] - times[i]
            if span > 7 * DAY_NS:
                break
            minimum_7d = min(minimum_7d, values[i])
            if span <= 2 * DAY_NS:
                minimum_48h = min(minimum_48h, values[i])
        if values[j] >= minimum_48h + 0.3 - 1e-9 or values[j] >= 1.5 * minimum_7d - 1e-9:
            flags[j] = 1
    return flags


@njit
def assess_groups(left, right, chart, stored, values, admission, landmark):
    n = len(left)
    baseline_n = np.zeros(n, dtype=np.int64)
    post_n = np.zeros(n, dtype=np.int64)
    prevalent = np.zeros(n, dtype=np.bool_)
    late_prevalent = np.zeros(n, dtype=np.bool_)
    first_t = np.full(n, -1, dtype=np.int64)
    last_pre = np.full(n, np.nan)
    for i in range(n):
        t, s, cr = chart[left[i]:right[i]], stored[left[i]:right[i]], values[left[i]:right[i]]
        flags = aki_flags(t, cr)
        for j in range(len(t)):
            if admission[i] <= t[j] < landmark[i] and s[j] <= landmark[i]:
                baseline_n[i] += 1
                last_pre[i] = cr[j]
            if t[j] <= landmark[i]:
                if flags[j]:
                    prevalent[i] = True
                    if s[j] > landmark[i]:
                        late_prevalent[i] = True
            else:
                post_n[i] += 1
                if flags[j] and first_t[i] == -1:
                    first_t[i] = t[j]
    return baseline_n, post_n, prevalent, late_prevalent, first_t, last_pre


def ns(series):
    return series.to_numpy(dtype="datetime64[ns]").view(np.int64)


def build(args):
    source = pd.read_csv(args.index, parse_dates=["admittime", "landmark", "stop_time"], dtype={"ID": str})
    if not source.ID.is_unique or not source.hadm_id.is_unique:
        raise ValueError("Incident index lacks unique patient or admission IDs")
    if not (source.landmark.sub(source.admittime).eq(pd.Timedelta(hours=24)) & source.stop_time.gt(source.landmark)).all():
        raise ValueError("Index does not match the 24-hour landmark definition")
    predictions = FORMAL / "predictions.csv.gz"
    expected = json.loads((FORMAL / "audit.json").read_text(encoding="utf8"))["predictions_sha256"]
    if digest(predictions) != expected:
        raise ValueError("Frozen scores differ from their formal validation audit")
    scored = pd.read_csv(predictions, dtype={"ID": str}, usecols=["ID", "kidney_lp", "full_lp"])
    panel_ok = scored.kidney_lp.notna() & scored.full_lp.notna()
    cohort = source.merge(scored.loc[panel_ok, ["ID"]], on="ID", how="inner", validate="one_to_one")
    cohort = cohort.sort_values("hadm_id", kind="stable").reset_index(drop=True)
    if len(cohort) != int(panel_ok.sum()):
        raise ValueError("Scorable kidney patients are absent from the incident index")
    chronic_hadm, early_rrt, source_hashes = dialysis_flags(args.archive, cohort)
    labs, counts = select_creatinines(args.labs, cohort)
    labs = labs.sort_values(["hadm_id", "charttime"], kind="stable").reset_index(drop=True)
    identifiers = labs.hadm_id.to_numpy(dtype=np.int64)
    left = np.searchsorted(identifiers, cohort.hadm_id, side="left")
    right = np.searchsorted(identifiers, cohort.hadm_id, side="right")
    bn, pn, before, late, event_times, cr = assess_groups(
        left, right, ns(labs.charttime), ns(labs.storetime), labs.creatinine.to_numpy(),
        ns(cohort.admittime), ns(cohort.landmark))

    dialysis = cohort.hadm_id.isin(early_rrt).to_numpy()
    chronic = cohort.hadm_id.isin(chronic_hadm).to_numpy()
    baseline = bn >= 2
    valid = baseline & ~before & ~dialysis & ~chronic
    fixed_end = ns(cohort.landmark) + HORIZON * DAY_NS
    in_end = ns(cohort.stop_time)
    follow_end = np.minimum(in_end, fixed_end)
    event = event_times != -1
    if not (event_times[event] <= follow_end[event]).all():
        raise ValueError("Observed AKI occurs after the permitted horizon")
    stop = np.where(event, event_times, follow_end)
    status = np.select([event, in_end > fixed_end, cohort.hospital_event.eq(1)],
                       ["event", "censor", "death"], default="discharge")
    output = pd.DataFrame({"ID": cohort.ID, "event": event.astype(int),
                           "time_days": (stop - ns(cohort.landmark)) / DAY_NS,
                           "end_status": status, "postlab_observed": pn > 0,
                           "postlab_n": pn, "baseline_value": cr, "baseline_creatinine_n": bn})
    output = output.loc[valid].reset_index(drop=True)
    if not output.ID.is_unique or not output.time_days.gt(0).all() or not output.time_days.le(HORIZON).all():
        raise ValueError("Incident outcome identifiers or follow-up failed validation")
    if not output.loc[output.event.eq(1), "postlab_observed"].all() or output.event.sum() == 0:
        raise ValueError("No verified incident AKI events with post-landmark creatinine")

    out = args.output / "inputs" / "kidney.csv.gz"
    audit = args.output / "audit" / "kidney.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    audit.parent.mkdir(exist_ok=True, parents=True)
    tmp = out.with_name(out.name + ".tmp")
    output.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    tmp.replace(out)
    details = {
        "endpoint": "First documented creatinine-defined AKI by day 7 after the 24-hour landmark",
        "definition": "Serum/plasma creatinine (50912, mg/dL) rise >=0.3 from the minimum observed in the preceding "
                      "48 hours or >=1.5 times the minimum observed in the preceding 7 days, strictly before current sample.",
        "lookback": "Up to 7 days before admission; outpatient values may contribute when matched to the same person.",
        "eligibility": "Scorable kidney and whole-body panels, at least two distinct in-hospital specimen times with "
                       "results available at the landmark, and no pre-landmark creatinine AKI or exclusion for dialysis.",
        "timing": "Pre-landmark sample evidence excludes prevalent AKI, even when its report becomes available later. "
                  "Post-landmark AKI onset is the sample time, not the report time.",
        "monitoring": "No requirement for repeat tests after the landmark. Output includes postlab_observed for "
                      "sensitivity analysis. An unmeasured outcome can be missed.",
        "dialysis": "Exclude discharge-coded ESRD/dialysis-dependence ICD-9 5856/V4511 or ICD-10 N186/Z992 "
                    "as retrospective chronic-status exclusions, without assigning onset time. Exclude any verified "
                    "dialysis/filtration procedure date on the first admission calendar day.",
        "limitations": "Creatinine criterion only. Oliguric AKI and occult baseline AKI may be missed. Diagnosis depends on "
                       "test frequency and preadmission lookback. Discharge/death compete with recorded in-hospital AKI.",
        "cohort_counts": {
            "scorable_kidney_roster": len(cohort),
            "excluded_no_two_available_admission_baselines": int((~baseline).sum()),
            "then_excluded_pre_landmark_aki": int((baseline & before).sum()),
            "includes_delayed_baseline_positive_evidence": int((baseline & before & late).sum()),
            "then_excluded_dated_early_rrt": int((baseline & ~before & dialysis).sum()),
            "then_excluded_end_stage_status": int((baseline & ~before & ~dialysis & chronic).sum()),
            "analyzed_n": len(output), "events": int(output.event.sum()),
            "repeated_after_landmark": int(output.postlab_observed.sum()),
            "status": {k: int(v) for k, v in output.end_status.value_counts().items()},
        },
        "source_sha256": {"index": digest(args.index), "verified_frozen_scores": expected, "serial_labs": digest(args.labs),
                          **source_hashes},
        "lab_audit": counts, "input_sha256": digest(out), "sources": [KDIGO, MIT_KDIGO],
    }
    audit.write_text(json.dumps(details, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(details["cohort_counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-file", dest="labs", type=Path, default=ROOT / "cache" / "serial_labs.csv.gz")
    parser.add_argument("--index", type=Path, default=ROOT / "cache" / "incident_index.csv.gz")
    parser.add_argument("--archive", type=Path, default=MIMIC_ARCHIVE)
    parser.add_argument("--output", type=Path, default=ROOT)
    build(parser.parse_args())
