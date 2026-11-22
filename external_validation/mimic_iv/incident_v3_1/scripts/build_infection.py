"""Build a time-stamped, culture-confirmed bloodstream pathogen endpoint."""

import argparse
import gzip
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


HERE = Path(__file__).resolve().parents[1]
MIMIC = HERE.parent
PREFIX = "mimic-iv-3.1/"
MEMBER = "hosp/microbiologyevents.csv.gz"
MICRO_DOC = "https://mimic.mit.edu/docs/iv/modules/hosp/microbiologyevents.html"
NHSN_DOC = "https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf"
NHSN_CC = "https://www.cdc.gov/nhsn/xls/master-organism-com-commensals-lists.xlsx"
HORIZON = pd.Timedelta(days=7)

# This conservative exact-name list uses common, clinically recognized
# bloodstream pathogens. Potential skin/oral commensals, presumptive
# morphology-only labels, and cancellation notices cannot qualify.
PATHOGENS = {
    "STAPH AUREUS COAG +", "STAPHYLOCOCCUS AUREUS",
    "ESCHERICHIA COLI", "ENTEROCOCCUS FAECIUM", "ENTEROCOCCUS FAECALIS",
    "KLEBSIELLA PNEUMONIAE", "KLEBSIELLA OXYTOCA", "PSEUDOMONAS AERUGINOSA",
    "BACTEROIDES FRAGILIS GROUP", "BACTEROIDES FRAGILIS", "PROTEUS MIRABILIS",
    "ENTEROBACTER CLOACAE COMPLEX", "ENTEROBACTER CLOACAE",
    "ENTEROBACTER AEROGENES", "KLEBSIELLA AEROGENES", "SERRATIA MARCESCENS",
    "CITROBACTER FREUNDII COMPLEX", "CITROBACTER FREUNDII", "CITROBACTER KOSERI",
    "STENOTROPHOMONAS MALTOPHILIA", "ACINETOBACTER BAUMANNII COMPLEX",
    "ACINETOBACTER BAUMANNII", "MORGANELLA MORGANII", "STREPTOCOCCUS PNEUMONIAE",
    "BETA STREPTOCOCCUS GROUP A", "BETA STREPTOCOCCUS GROUP B",
    "STREPTOCOCCUS PYOGENES", "STREPTOCOCCUS AGALACTIAE",
    "SALMONELLA SPECIES", "SALMONELLA ENTERITIDIS", "SALMONELLA TYPHI",
    "LISTERIA MONOCYTOGENES", "FUSOBACTERIUM NUCLEATUM",
    "HAEMOPHILUS INFLUENZAE", "HAEMOPHILUS INFLUENZAE, BETA-LACTAMASE NEGATIVE",
    "HAEMOPHILUS INFLUENZAE, BETA-LACTAMASE POSITIVE",
    "CANDIDA ALBICANS", "CANDIDA GLABRATA", "CANDIDA TROPICALIS",
    "CANDIDA PARAPSILOSIS", "CANDIDA DUBLINIENSIS", "CANDIDA KRUSEI",
    "CRYPTOCOCCUS NEOFORMANS",
}
CULTURE = {90201: "Blood Culture, Routine", 90265: "BLOOD/FUNGAL CULTURE"}
COLUMNS = [
    "subject_id", "hadm_id", "micro_specimen_id", "charttime", "chartdate",
    "spec_itemid", "spec_type_desc", "test_itemid", "test_name", "org_itemid",
    "org_name", "storetime", "comments",
]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cohort(formal, shared_index):
    index = pd.read_csv(
        shared_index,
        usecols=["ID", "subject_id", "hadm_id", "admittime", "landmark", "stop_time", "hospital_event"],
        parse_dates=["admittime", "landmark", "stop_time"],
    ).rename(columns={"stop_time": "hospital_stop"})
    preds = pd.read_csv(formal / "predictions.csv.gz", usecols=["ID", "immune_heme_lp"])
    if not index.subject_id.is_unique or not index.hadm_id.is_unique or not preds.ID.is_unique:
        raise ValueError("Source cohort is not one index hospitalization per participant")
    if not index.ID.eq(index.subject_id).all() or set(index.ID) != set(preds.ID):
        raise ValueError("Shared incident index differs from the frozen scoring roster")
    if not index.landmark.sub(index.admittime).eq(pd.Timedelta(hours=24)).all():
        raise ValueError("Unexpected 24-hour landmark anchor")
    scored = preds.loc[preds.immune_heme_lp.notna(), ["ID"]].rename(columns={"ID": "subject_id"})
    index = scored.merge(index, on="subject_id", validate="one_to_one")
    if len(index) != len(scored) or not index.hospital_event.isin([0, 1]).all():
        raise ValueError("Some frozen immune/hematologic scores lack a confirmed hospitalization")
    if not index.hospital_stop.gt(index.landmark).all():
        raise ValueError("Invalid waiting time to discharge or in-hospital death")
    return index.drop(columns="ID")


def load_specimens(archive, index):
    with ZipFile(archive) as source:
        expected = {
            name.lstrip("*"): digest
            for digest, name in (
                row.split(maxsplit=1)
                for row in source.read(PREFIX + "SHA256SUMS.txt").decode("ascii").splitlines()
                if row.strip()
            )
        }[MEMBER]
        payload = source.read(PREFIX + MEMBER)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("MIMIC-IV microbiology source failed archive SHA-256 verification")
    blocks = []
    total = selected = without_hadm = 0
    types = {"hadm_id": "Int64", "subject_id": "int64", "micro_specimen_id": "int64"}
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        for rows in pd.read_csv(stream, usecols=COLUMNS, dtype=types, chunksize=500_000):
            total += len(rows)
            eligible = rows.spec_itemid.isin([70011, 70012]) & rows.test_itemid.isin(CULTURE)
            blood = rows.loc[eligible]
            selected += len(blood)
            without_hadm += int(blood.hadm_id.isna().sum())
            if not blood.test_name.eq(blood.test_itemid.map(CULTURE)).all():
                raise ValueError("Configured blood culture test labels differ in the source release")
            blocks.append(blood.loc[blood.hadm_id.isin(index.hadm_id)])
    raw = pd.concat(blocks, ignore_index=True)
    group = raw.groupby("micro_specimen_id")
    key_count = group[["subject_id", "hadm_id", "charttime", "chartdate"]].nunique()
    if key_count.gt(1).any().any():
        raise ValueError("A specimen ID maps to inconsistent patient/stay/sampling time")
    raw["org"] = raw.org_name.astype("string").str.strip().str.upper()
    confirmed = raw.org.isin(PATHOGENS)
    if raw.loc[confirmed, "org_itemid"].isna().any():
        raise ValueError("A typed pathogen lacks its organism item ID")
    notes = raw.comments.astype("string")
    raw["test_negative"] = raw.org_name.isna() & raw.org_itemid.isna() & notes.str.contains(
        r"\bNO GROWTH\b", case=False, na=False
    )
    raw["result_growth"] = raw.org.notna() & ~raw.org.isin(["CANCELLED", "CANCELED", "NO GROWTH", "VOIDED"])
    raw["test_confirmed"] = confirmed
    raw["negative_storetime"] = pd.to_datetime(raw.storetime.where(raw.test_negative))
    raw["charttime"] = pd.to_datetime(raw.charttime)
    raw["chartdate"] = pd.to_datetime(raw.chartdate)
    grouped = raw.groupby("micro_specimen_id", sort=False).agg(
        subject_id=("subject_id", "first"), hadm_id=("hadm_id", "first"),
        charttime=("charttime", "first"), chartdate=("chartdate", "first"),
        confirmed_pathogen=("test_confirmed", "any"), any_growth=("result_growth", "any"),
        no_growth_record=("test_negative", "any"), negative_storetime=("negative_storetime", "min"),
    )
    grouped["negative"] = grouped.no_growth_record & ~grouped.any_growth
    named = raw.loc[confirmed, ["micro_specimen_id", "org"]].drop_duplicates().groupby("org").size()
    audit = {
        "microbiology_member_sha256": expected, "microbiology_rows_scanned": total,
        "qualifying_source_culture_rows": selected, "without_hadm_id_rows_in_source_cultures": without_hadm,
        "selected_cohort_specimens": len(grouped), "selected_cohort_positive_pathogen_specimens": int(grouped.confirmed_pathogen.sum()),
        "selected_cohort_definitive_negative_specimens": int(grouped.negative.sum()),
        "source_culture_test_itemids": CULTURE, "source_blood_specimen_itemids": [70011, 70012],
        "pathogen_names_in_definition": sorted(PATHOGENS),
        "recognized_pathogen_names_seen": {key: int(value) for key, value in named.items()},
    }
    return grouped.reset_index(), audit


def endpoint(index, specimens, audit):
    d = specimens.merge(index, on=["subject_id", "hadm_id"], validate="many_to_one")
    window_end = d.landmark + HORIZON
    exact = d.charttime.notna()
    base = exact & d.charttime.ge(d.admittime) & d.charttime.le(d.landmark)
    qualified_followup = exact & d.charttime.gt(d.landmark) & d.charttime.lt(d.hospital_stop)
    in_seven = qualified_followup & d.charttime.le(window_end)
    early_pathogen = d.confirmed_pathogen & exact & d.charttime.le(d.landmark)
    # An hour-less pathogen-positive specimen cannot qualify as a dated event.
    # Remove admissions where such a specimen could overlap baseline or follow-up.
    day_uncertain = d.confirmed_pathogen & ~exact & d.chartdate.notna() & (
        d.chartdate.lt(d.hospital_stop) & d.chartdate.le(window_end)
        & (d.chartdate + pd.Timedelta(days=1)).gt(d.admittime)
    )
    base_neg = d.loc[base & d.negative]
    baseline_negative = base_neg.groupby("subject_id").size()
    baseline_negative_known = base_neg.loc[
        base_neg.negative_storetime.ge(base_neg.charttime) & base_neg.negative_storetime.le(base_neg.landmark)
    ].groupby("subject_id").size()
    positive = d.loc[qualified_followup & d.confirmed_pathogen].groupby("subject_id").charttime.min()
    monitored = d.loc[in_seven & (d.confirmed_pathogen | d.negative)].groupby("subject_id").size()
    survivors = index.copy()
    survives_negative = survivors.subject_id.isin(baseline_negative.index)
    exclusion_early = survivors.subject_id.isin(d.loc[early_pathogen, "subject_id"])
    exclusion_day = survivors.subject_id.isin(d.loc[day_uncertain, "subject_id"])
    eligible = survives_negative & ~exclusion_early & ~exclusion_day
    a = survivors.loc[eligible].copy()
    a["culture_positive_time"] = a.subject_id.map(positive)
    a["horizon"] = a.landmark + HORIZON
    a["event"] = a.culture_positive_time.notna() & a.culture_positive_time.le(a.horizon)
    ended_in_window = a.hospital_stop.le(a.horizon)
    a["end_status"] = np.select(
        [a.event, ended_in_window & a.hospital_event.eq(1), ended_in_window],
        ["event", "death", "discharge"], default="censor",
    )
    a["endtime"] = a[["horizon", "hospital_stop"]].min(axis=1).where(~a.event, a.culture_positive_time)
    a["time_days"] = (a.endtime - a.landmark).dt.total_seconds() / 86400
    a["post_observed"] = a.subject_id.isin(monitored.index)
    a["early_negative_available_by_landmark"] = a.subject_id.isin(baseline_negative_known.index)
    audit.update(
        immune_heme_scorable_candidates=len(index),
        baseline_definitively_negative_candidate_people=int(survives_negative.sum()),
        early_confirmed_pathogen_among_candidates=int(exclusion_early.sum()),
        early_confirmed_pathogen_excluded_among_baseline_negative=int((survives_negative & exclusion_early).sum()),
        date_only_confirmed_pathogen_excluded_among_still_eligible=int(
            (survives_negative & ~exclusion_early & exclusion_day).sum()
        ),
        n_analyzed=len(a), events_7d=int(a.event.sum()),
        events_during_entire_index_hospitalization=int(a.culture_positive_time.notna().sum()),
        early_negative_available_by_landmark=int(a.early_negative_available_by_landmark.sum()),
        post_observed=int(a.post_observed.sum()),
        post_observed_non_events=int((a.post_observed & ~a.event).sum()),
        end_status_counts={key: int(value) for key, value in a.end_status.value_counts().items()},
        horizon_days=int(HORIZON.days),
        baseline_negative_note=(
            "At least one routine or fungal blood-culture specimen collected by the landmark with documented no growth "
            "and no organism on that specimen. Final laboratory results for most such samples occur after the landmark. "
            "This defines a retrospectively verified phenotype cohort, not an eligibility rule available at prediction time."
        ),
        endpoint_note=(
            "First confirmed typed bloodstream pathogen in the conservative exact list, sampled after the landmark, "
            "by day 7, and strictly before index-stay discharge or death. Common commensals or unspecified culture labels "
            "are not counted. The construct does not distinguish primary and secondary BSI or establish infection onset, "
            "hospital attribution, or catheter attribution. No post-landmark test is required for inclusion; no detected "
            "event therefore need not imply absence of infection."
        ),
        official_sources=[MICRO_DOC, NHSN_DOC, NHSN_CC],
    )
    if not a.time_days.gt(0).all() or not a.time_days.le(7).all() or not a.subject_id.is_unique:
        raise ValueError("Invalid follow-up duration or duplicated patient in the outcome cohort")
    if not (a.event == a.end_status.eq("event")).all() or not a.loc[a.event, "post_observed"].all():
        raise ValueError("Event labels or detection support are inconsistent")
    return a.rename(columns={"subject_id": "ID"}).assign(event=a.event.astype(int))[
        ["ID", "event", "time_days", "end_status", "post_observed", "early_negative_available_by_landmark"]
    ].sort_values("ID").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=MIMIC_ARCHIVE)
    parser.add_argument("--formal", type=Path, default=MIMIC / "formal_v3_1")
    parser.add_argument("--index", type=Path, default=HERE / "cache" / "incident_index.csv.gz")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    cohort = load_cohort(args.formal, args.index)
    cultures, audit = load_specimens(args.archive, cohort)
    result = endpoint(cohort, cultures, audit)
    audit.update(
        landmark_index_sha256=file_hash(args.formal / "landmark_index.csv.gz"),
        shared_incident_index_sha256=file_hash(args.index),
        frozen_predictions_sha256=file_hash(args.formal / "predictions.csv.gz"),
    )
    inputs = args.out / "inputs"
    metadata = args.out / "audit"
    inputs.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    path = inputs / "immune_heme.csv.gz"
    provisional = path.with_suffix(".gz.tmp")
    result.to_csv(provisional, index=False, compression={"method": "gzip", "mtime": 0})
    provisional.replace(path)
    audit["input_sha256"] = file_hash(path)
    path = metadata / "immune_heme.json"
    provisional = path.with_suffix(".json.tmp")
    provisional.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8")
    provisional.replace(path)
    print(f"Immune/hematologic endpoint: {len(result):,} people, {int(result.event.sum()):,} day-7 events")


if __name__ == "__main__":
    main()
