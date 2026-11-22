"""Build a 24-hour MIMIC-IV landmark and validate the frozen NHANES clock."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAP = {
    50862: ("albumin", "Albumin", "g/dl"),
    50863: ("alp", "Alkaline Phosphatase", "iu/l"),
    50878: ("ast", "Asparate Aminotransferase (AST)", "iu/l"),
    50861: ("alt", "Alanine Aminotransferase (ALT)", "iu/l"),
    50927: ("ggt", "Gamma Glutamyltransferase", "iu/l"),
    50885: ("bilirubin", "Bilirubin, Total", "mg/dl"),
    50976: ("total_protein", "Protein, Total", "g/dl"),
    50930: ("globulin", "Globulin", "g/dl"),
    50954: ("ldh", "Lactate Dehydrogenase (LD)", "iu/l"),
    50912: ("creatinine", "Creatinine", "mg/dl"),
    51006: ("bun", "Urea Nitrogen", "mg/dl"),
    51007: ("uric_acid", "Uric Acid", "mg/dl"),
    50971: ("potassium", "Potassium", "meq/l"),
    50983: ("sodium", "Sodium", "meq/l"),
    50902: ("chloride", "Chloride", "meq/l"),
    50882: ("bicarbonate", "Bicarbonate", "meq/l"),
    50970: ("phosphorus", "Phosphate", "mg/dl"),
    50893: ("calcium", "Calcium, Total", "mg/dl"),
    50931: ("glucose", "Glucose", "mg/dl"),
    50852: ("hba1c", "% Hemoglobin A1c", "%"),
    51000: ("triglycerides", "Triglycerides", "mg/dl"),
    50907: ("cholesterol", "Cholesterol, Total", "mg/dl"),
    50904: ("hdl", "Cholesterol, HDL", "mg/dl"),
    51301: ("wbc", "White Blood Cells", "k/ul"),
    51244: ("lymph_pct", "Lymphocytes", "%"),
    51254: ("mono_pct", "Monocytes", "%"),
    51256: ("neut_pct", "Neutrophils", "%"),
    51200: ("eos_pct", "Eosinophils", "%"),
    50889: ("crp", "C-Reactive Protein", "mg/l"),
    51279: ("rbc", "Red Blood Cells", "m/ul"),
    51222: ("hgb", "Hemoglobin", "g/dl"),
    51250: ("mcv", "MCV", "fl"),
    51248: ("mch", "MCH", "pg"),
    51277: ("rdw", "RDW", "%"),
    51265: ("platelets", "Platelet Count", "k/ul"),
    52142: ("mpv", "Mean Platelet Volume", "fl"),
    50952: ("iron", "Iron", "ug/dl"),
}
DIFF = {51244, 51254, 51256, 51200}
YEAR = pd.Timedelta(days=365)
DAY = pd.Timedelta(days=1)
TABLES = ("patients", "admissions", "d_labitems", "labevents")
SOURCE = "https://physionet.org/content/mimiciv/3.1/"
DEMO = "https://physionet.org/content/mimic-iv-demo/2.2/"
LABELS = {item: entry[1].casefold() for item, entry in MAP.items()}
FEATURES = {item: entry[0] for item, entry in MAP.items()}
UNITS = {item: entry[2] for item, entry in MAP.items()}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def files_for(root):
    roots = (root / "hosp", root)
    for folder in roots:
        files = {}
        for name in TABLES:
            matches = [folder / f"{name}{suffix}" for suffix in (".csv.gz", ".csv")
                       if (folder / f"{name}{suffix}").is_file()]
            if len(matches) == 1:
                files[name] = matches[0]
        if len(files) == len(TABLES):
            return files
    raise FileNotFoundError("Expected patients, admissions, d_labitems and labevents together (CSV or CSV.GZ)")


def one_per_patient(admissions, patients, audit):
    if not patients.subject_id.is_unique or not admissions.hadm_id.is_unique:
        raise ValueError("Duplicate patients.subject_id or admissions.hadm_id")
    if admissions.subject_id.isna().any() or not set(admissions.subject_id).issubset(patients.subject_id):
        raise ValueError("Admission without linked patient")
    audit["admissions"] = len(admissions)
    audit["patients"] = len(patients)
    admissions = admissions.sort_values(["subject_id", "admittime", "hadm_id"], kind="stable")
    # A missing admission timestamp makes the first encounter indeterminate.
    admission_time_unknown = admissions.loc[admissions.admittime.isna(), "subject_id"]
    first = admissions.drop_duplicates("subject_id").copy().rename(columns={"deathtime": "index_deathtime"})
    hosp_death = admissions.loc[admissions.hospital_expire_flag.eq(1), ["subject_id", "deathtime", "dischtime"]].copy()
    hosp_death["death"] = hosp_death.deathtime.fillna(hosp_death.dischtime)
    hosp_death["recorded_exact"] = hosp_death.deathtime.notna()
    known_death = hosp_death.sort_values(["death", "recorded_exact"], ascending=[True, False]).drop_duplicates("subject_id")
    known_death = known_death.set_index("subject_id")[["death", "recorded_exact"]]
    last_contact = admissions.groupby("subject_id").agg(last_discharge=("dischtime", "max"),
                                                       last_admission=("admittime", "max"))
    first = first.join(last_contact, on="subject_id", validate="one_to_one")
    first = first.join(known_death, on="subject_id", validate="one_to_one")
    first = first.merge(patients, on="subject_id", validate="one_to_one")
    first["age"] = (first.anchor_age + first.admittime.dt.year - first.anchor_year).where(first.anchor_age.lt(91), 91)
    first.loc[first.admittime.isna() | first.anchor_age.isna() | first.anchor_year.isna(), "age"] = np.nan
    first["sex"] = first.gender.map({"M": 1, "F": 0})
    first["landmark"] = first.admittime + DAY
    first["horizon"] = first.landmark + YEAR
    audit["first_admissions"] = len(first)

    eligible = ~first.subject_id.isin(admission_time_unknown) & first.age.between(20, 120)
    audit["excluded_age_or_unknown_first_admission"] = int((~eligible).sum())
    still_here = first.dischtime.gt(first.landmark)
    audit["excluded_discharged_or_dead_at_or_before_24h"] = int((eligible & ~still_here).sum())
    eligible &= still_here
    flag_valid = first.hospital_expire_flag.isin([0, 1])
    index_death = first.index_deathtime.fillna(first.dischtime).where(first.hospital_expire_flag.eq(1))
    index_conflict = (first.hospital_expire_flag.eq(0) & first.index_deathtime.notna()) | (
        index_death.notna() & ((index_death < first.admittime) | (index_death > first.dischtime + DAY))
    )
    uncertain = first.dod.notna() & first.dod.eq(first.landmark.dt.normalize()) & first.death.isna()
    state_conflict = (first.dod.notna() & first.death.notna() &
                      first.death.dt.normalize().sub(first.dod).abs().gt(DAY))
    date_only = first.dod + pd.Timedelta(hours=12)
    # An alive discharge on the date of a date-only death occurred first.
    date_only = date_only.mask(
        first.hospital_expire_flag.eq(0) & first.dod.eq(first.dischtime.dt.normalize()) & (date_only < first.dischtime),
        first.dischtime,
    )
    death = first.death.fillna(date_only)
    death_in_stay_with_live_discharge = (
        first.hospital_expire_flag.eq(0) & death.notna() & (death < first.dischtime)
    )
    impossible_next_admission = (
        (first.death.notna() & first.last_admission.gt(first.death)) |
        (first.death.isna() & first.dod.notna() & first.last_admission.ge(first.dod + DAY))
    )
    # Require a disposition and enough release-defined follow-up at this landmark.
    censor_coverage = (first.last_discharge + YEAR).ge(first.horizon)
    death_flag_without_time = first.subject_id.isin(hosp_death.loc[hosp_death.death.isna(), "subject_id"])
    endpoint_bad = (~flag_valid | index_conflict | state_conflict | uncertain |
                    death_in_stay_with_live_discharge | impossible_next_admission | death_flag_without_time |
                    (death.notna() & (death <= first.landmark)) | ~censor_coverage)
    audit["excluded_death_conflict_coverage_or_landmark_ambiguity"] = int((eligible & endpoint_bad).sum())
    keep = eligible & ~endpoint_bad
    event = death.notna() & death.le(first.horizon)
    first["event"] = event.astype(int)
    first["time_yr"] = (death.where(event, first.horizon) - first.landmark) / YEAR
    first["hospital_event"] = first.hospital_expire_flag.where(flag_valid).astype("Int64")
    first["hospital_time_yr"] = (index_death.fillna(first.dischtime) - first.landmark) / YEAR
    first["death_timing"] = np.select(
        [~event, first.death.notna() & first.recorded_exact.eq(True), first.death.notna()],
        ["administrative_365d", "hospital_timestamp", "expired_discharge_timestamp"],
        default="linked_date_midday",
    )
    impossible_hospital_time = ~first.hospital_time_yr.gt(0)
    audit["excluded_nonpositive_index_hospital_time"] = int((keep & impossible_hospital_time).sum())
    keep &= ~impossible_hospital_time
    first = first.loc[keep].copy()
    if not first.subject_id.is_unique or not first.time_yr.between(0, 1, inclusive="right").all():
        raise ValueError("Invalid participant count or 365-day survival")
    audit["eligible_landmark_before_labs"] = len(first)
    return first


def lab_values(path, dictionary, index, chunk_size, audit):
    labels = dictionary.set_index("itemid", verify_integrity=True).reindex(MAP)
    if not labels.label.astype("string").str.casefold().eq(pd.Series(LABELS)).fillna(False).all():
        raise ValueError("This release uses missing or changed MIMIC laboratory item labels")
    if not labels.fluid.str.casefold().eq("blood").all():
        raise ValueError("A configured item is not labelled as a blood specimen")
    source = index[["subject_id", "admittime", "landmark", "hadm_id"]]
    keys = ["hadm_id", "itemid"]
    sort = keys + ["charttime", "storetime", "labevent_id"]
    types = {"subject_id": "int64", "hadm_id": "Int64", "labevent_id": "int64", "itemid": "int64"}
    columns = ["subject_id", "hadm_id", "itemid", "labevent_id", "charttime", "storetime", "valuenum", "valueuom"]
    candidates = []
    unit_counts = {}
    invalid_labs = 0
    n_window = 0
    n_valid = 0
    rows_read = 0
    for chunk, lab in enumerate(pd.read_csv(path, usecols=columns, dtype=types, chunksize=chunk_size), start=1):
        rows_read += len(lab)
        hit = lab.itemid.isin(MAP) & lab.hadm_id.isin(source.hadm_id)
        lab = lab.loc[hit].merge(source, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
        lab["charttime"] = pd.to_datetime(lab.charttime, errors="coerce")
        lab["storetime"] = pd.to_datetime(lab.storetime, errors="coerce")
        observed = (lab.charttime.ge(lab.admittime) & lab.charttime.lt(lab.landmark) &
                    lab.storetime.ge(lab.charttime) & lab.storetime.le(lab.landmark))
        lab = lab.loc[observed].copy()
        n_window += len(lab)
        expected = lab.itemid.map(UNITS)
        units = lab.valueuom.astype("string").str.lower().str.strip().str.replace("µ", "u").str.replace("μ", "u")
        known_unit = (units.eq(expected) |
                      (expected.eq("iu/l") & units.eq("u/l")) |
                      (expected.eq("meq/l") & units.eq("mmol/l"))).fillna(False)
        for (item, unit), count in lab.loc[~known_unit].assign(unit=units.fillna("[missing]")).groupby(["itemid", "unit"]).size().items():
            key = f"{item}:{unit}"
            unit_counts[key] = unit_counts.get(key, 0) + int(count)
        value = pd.to_numeric(lab.valuenum, errors="coerce")
        proportions = lab.itemid.isin(DIFF)
        # Accept reported zero for the differential and CRP; physical units and normal ranges are distinct.
        physical = (value.gt(0) | (lab.itemid.isin(DIFF | {50889}) & value.eq(0))) & np.isfinite(value)
        physical &= ~proportions | value.le(100)
        invalid_labs += int((known_unit & ~physical).sum())
        keep = known_unit & physical
        n_valid += int(keep.sum())
        lab["value"] = value.where(~lab.itemid.eq(50889), value / 10)
        candidate = lab.loc[keep, sort + ["value"]].sort_values(sort, kind="stable")
        candidates.append(candidate.drop_duplicates(keys))
        if chunk % 10 == 0:
            print(f"labevents: {rows_read:,} rows scanned; {n_valid:,} qualifying measurements", flush=True)
    if not candidates or not n_valid:
        raise ValueError("No valid measurements for the specified landmark and item map")
    winners = pd.concat(candidates, ignore_index=True).sort_values(sort, kind="stable").drop_duplicates(keys)
    winners["feature"] = winners.itemid.map(FEATURES)
    wide = winners.pivot(index="hadm_id", columns="feature", values="value")
    audit.update(lab_rows_scanned=rows_read, window_candidate_lab_rows=n_window, eligible_lab_rows=n_valid, invalid_numeric_lab_rows=invalid_labs,
                 unsupported_units=unit_counts, lab_features={name: int(n) for name, n in wide.count().items()})
    return wide


def formal_score(frame, out):
    external = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(external / "scripts"))
    import torch
    from lifelines import CoxPHFitter
    from sklearn.metrics import roc_auc_score
    from reference import BUNDLE, REFERENCE, TRAIN_RUN, linear_score, prepare_features, score_frame, verify
    from validate_external import discrimination

    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = verify()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if ref["freeze_sha256"] != digest(TRAIN_RUN / "model_freeze.json"):
        raise ValueError("NHANES comparators do not correspond to the frozen clock")
    if not set(FEATURES.values()).issubset(bundle["features"]):
        raise ValueError("MIMIC-IV features differ from the frozen clock")
    scores = score_frame(frame, BUNDLE, device)
    x, _, age_z = prepare_features(frame, bundle)
    for c in ["sex", "time_yr", "event", "hospital_event", "hospital_time_yr", "death_timing"]:
        scores[c] = frame[c]
    scores["age_sex_lp"] = linear_score(scores[["age", "sex"]].to_numpy(), ref["age_sex"])
    scores["ridge_lp"] = linear_score(np.c_[x, age_z], ref["ridge"])
    complete = scores[scores.sex.notna()].copy()
    columns = ["full_lp", "age_sex_lp", "ridge_lp"]
    if complete.event.nunique() != 2 or complete.hospital_event.nunique() != 2:
        raise ValueError("Both outcomes must have observed cases and controls")
    result = {
        "n_scored": len(scores),
        "n_365d_deaths_all_scored": int(scores.event.sum()),
        "n_index_hospital_deaths_all_scored": int(scores.hospital_event.sum()),
        "n_compared": len(complete),
        "n_365d_deaths": int(complete.event.sum()),
        "n_index_hospital_deaths": int(complete.hospital_event.sum()),
        "index_hospital_ROC_AUC": {key: float(roc_auc_score(complete.hospital_event, complete[key])) for key in columns},
        "post_landmark_365d_Harrell_C": discrimination(complete, columns),
        "post_landmark_365d_ROC_AUC": {key: float(roc_auc_score(complete.event, complete[key])) for key in columns},
        "reference_SHA256": digest(REFERENCE),
        "device": device,
    }
    covariates = complete[["age", "sex", "event", "time_yr"]].copy()
    accel_sd = complete.full_accel.std()
    if accel_sd > 0 and complete.sex.nunique() == 2 and complete.event.sum() >= 20:
        covariates["accel_sd"] = (complete.full_accel - complete.full_accel.mean()) / accel_sd
        fit = CoxPHFitter().fit(covariates, "time_yr", "event")
        result["accel_association"] = {
            "HR": float(np.exp(fit.params_["accel_sd"])),
            "CI95": np.exp(fit.confidence_intervals_.loc["accel_sd"]).tolist(),
            "SD_years": float(accel_sd),
        }
    scores.to_csv(out / "predictions.csv.gz", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Local PhysioNet release root or its hosp folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", choices=["official-3.1", "demo-2.2"], required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()
    files = files_for(args.data_root.resolve())
    release = args.release
    demo = release == "demo-2.2"
    out = args.out.resolve()
    if out.is_relative_to(args.data_root.resolve()):
        raise ValueError("Use an output directory separate from the source data")
    admissions = pd.read_csv(files["admissions"], parse_dates=["admittime","dischtime","deathtime"],
                             usecols=["subject_id","hadm_id","admittime","dischtime","deathtime","hospital_expire_flag"])
    patients = pd.read_csv(files["patients"], parse_dates=["dod"],
                           usecols=["subject_id","gender","anchor_age","anchor_year","dod"])
    if demo and len(patients) != 100:
        raise ValueError("The official demo contains 100 patients")
    if not demo and (len(patients) != 364_627 or len(admissions) != 546_028):
        raise ValueError("The source row counts do not match the full PhysioNet MIMIC-IV 3.1 release")
    dictionary = pd.read_csv(files["d_labitems"], usecols=["itemid","label","fluid"])
    audit = {"release": release, "reference": DEMO if demo else SOURCE,
             "formal_validation_run": not demo}
    index = one_per_patient(admissions, patients, audit)
    values = lab_values(files["labevents"], dictionary, index, args.chunk_size, audit)
    frame = index.merge(values, on="hadm_id", how="inner", validate="one_to_one").reset_index(drop=True)
    frame = frame.rename(columns={"subject_id": "ID"})
    audit["n_scoring_eligible"] = len(frame)
    audit["excluded_no_window_mapped_lab"] = len(index) - len(frame)
    marker_counts = frame.reindex(columns=FEATURES.values()).notna().sum(axis=1)
    audit["observed_markers_median"] = float(marker_counts.median())
    audit["people_with_at_least_22_markers"] = int(marker_counts.ge(22).sum())
    audit["people_with_age_topcoded"] = int(frame.anchor_age.eq(91).sum())
    audit["people_with_unknown_sex"] = int(frame.sex.isna().sum())
    audit["death_date_quality_counts"] = {str(k):int(v) for k,v in frame.death_timing.value_counts().items()}
    audit["recorded_deaths_by_365_days"] = int(frame.event.sum())
    audit["recorded_index_hospital_deaths"] = int(frame.hospital_event.sum())
    audit["source_data"] = {k: {"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for k, p in files.items()}
    out.mkdir(parents=True, exist_ok=True)
    if not demo:
        formal_score(frame, out)
    (out / ("demo_qc.json" if demo else "audit.json")).write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{release}: landmark data pipeline verified for {len(frame)} people", flush=True)


if __name__ == "__main__":
    main()
