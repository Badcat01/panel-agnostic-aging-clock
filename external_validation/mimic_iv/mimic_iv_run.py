"""Prepare the local v3.1 landmark cohort, then score it with the frozen NHANES ensemble."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from study_paths import MIMIC_ARCHIVE
from zipfile import ZipFile

import numpy as np
import pandas as pd

from mimic_iv_validate import SOURCE, TABLES, digest, files_for, lab_values, one_per_patient

HERE = Path(__file__).resolve().parent
OUTCOMES = ["event", "time_yr", "hospital_event", "hospital_time_yr", "death_timing", "sex"]
TIMES = ["admittime", "landmark"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def write_csv(path, frame, index=False):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=index, compression={"method": "gzip", "mtime": 0})
    tmp.replace(path)


def extract(archive, data_root):
    prefix = "mimic-iv-3.1/"
    hosp = data_root / "hosp"
    hosp.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with ZipFile(archive) as source:
        rows = source.read(prefix + "SHA256SUMS.txt").decode("ascii").splitlines()
        manifest = {name.lstrip("*"): sha for sha, name in (row.split(maxsplit=1) for row in rows if row.strip())}
        for name in TABLES:
            path = hosp / (name + ".csv.gz")
            member = "hosp/" + path.name
            expected = manifest[member]
            if path.exists():
                if digest(path) != expected:
                    raise ValueError(f"Existing local table differs from archive manifest: {path}")
            else:
                tmp = path.with_suffix(".gz.tmp")
                checksum = hashlib.sha256()
                with source.open(prefix + member) as stream, tmp.open("wb") as destination:
                    for block in iter(lambda: stream.read(1 << 24), b""):
                        checksum.update(block)
                        destination.write(block)
                if checksum.hexdigest() != expected:
                    raise ValueError(f"Hospital table failed SHA-256 check: {member}")
                tmp.replace(path)
            outputs[name] = {"name": path.name, "bytes": path.stat().st_size, "sha256": expected}
    write_json(data_root / "verified_members.json", outputs)


def prepare(data_root, out):
    files = files_for(data_root)
    checked = read_json(data_root / "verified_members.json")
    for key, path in files.items():
        if checked[key]["bytes"] != path.stat().st_size or checked[key]["sha256"] != digest(path):
            raise ValueError(f"Local MIMIC source differs from archive verification: {key}")
    admissions = pd.read_csv(files["admissions"], parse_dates=["admittime", "dischtime", "deathtime"],
                             usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"])
    patients = pd.read_csv(files["patients"], parse_dates=["dod"],
                           usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
    if (len(patients), len(admissions)) != (364627, 546028):
        raise ValueError("Unexpected patient or admission count for MIMIC-IV v3.1")
    audit = {"release": "official-3.1", "source": SOURCE, "source_data": checked}
    index = one_per_patient(admissions, patients, audit)
    columns = ["subject_id", "hadm_id", "age", *TIMES, *OUTCOMES, "anchor_age"]
    index_path = out / "landmark_index.csv.gz"
    write_csv(index_path, index[columns])
    audit["index_sha256"] = digest(index_path)
    audit["index_hospital_deaths"] = int(index.hospital_event.sum())
    audit["index_365d_deaths"] = int(index.event.sum())
    write_json(out / "landmark_audit.json", audit)
    print(f"Landmark index ready: {len(index):,} people", flush=True)


def read_index(out):
    path = out / "landmark_index.csv.gz"
    audit = read_json(out / "landmark_audit.json")
    if digest(path) != audit["index_sha256"]:
        raise ValueError("Landmark index digest differs")
    return pd.read_csv(path, parse_dates=TIMES), audit


def labs(data_root, out, chunk_size):
    files = files_for(data_root)
    index, index_audit = read_index(out)
    audit = {"index_sha256": index_audit["index_sha256"], "archive_member_sha256": index_audit["source_data"]["labevents"]["sha256"]}
    dictionary = pd.read_csv(files["d_labitems"], usecols=["itemid", "label", "fluid"])
    values = lab_values(files["labevents"], dictionary, index, chunk_size, audit)
    path = out / "lab_panel.csv.gz"
    write_csv(path, values, index=True)
    audit["lab_panel_sha256"] = digest(path)
    audit["n_with_qualifying_lab"] = len(values)
    write_json(out / "lab_audit.json", audit)
    print(f"Lab panel ready: {len(values):,} people", flush=True)


def score(out):
    sys.path.insert(0, str(HERE.parent / "scripts"))
    import torch
    from reference import BUNDLE, REFERENCE, TRAIN_RUN, linear_score, prepare_features, score_frame, verify

    idx, index_audit = read_index(out)
    lab_audit = read_json(out / "lab_audit.json")
    panel = out / "lab_panel.csv.gz"
    if lab_audit["index_sha256"] != index_audit["index_sha256"] or lab_audit["lab_panel_sha256"] != digest(panel):
        raise ValueError("Panel input digests do not match")
    frame = idx.merge(pd.read_csv(panel), on="hadm_id", validate="one_to_one")
    vitals_path = out / "vitals.csv.gz"
    vitals_audit = read_json(out / "vitals_audit.json")
    if vitals_audit["index_sha256"] != index_audit["index_sha256"] or vitals_audit["vitals_sha256"] != digest(vitals_path):
        raise ValueError("ICU panel does not match the indexed cohort or its audit")
    vitals = pd.read_csv(vitals_path)
    if not vitals.hadm_id.is_unique or len(vitals.merge(idx[["hadm_id", "subject_id"]],
            on=["hadm_id", "subject_id"], validate="one_to_one")) != len(vitals):
        raise ValueError("ICU vitals do not map uniquely to the landmark index")
    frame = frame.merge(vitals[["subject_id", "hadm_id", "sbp", "dbp", "pulse", "bmi"]],
                        on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    frame = frame.rename(columns={"subject_id": "ID"})
    if not frame.ID.is_unique or not frame.sex.isin([0, 1]).all():
        raise ValueError("Invalid ID or sex coding")
    bundle = verify()
    ref = read_json(REFERENCE)
    freeze = digest(TRAIN_RUN / "model_freeze.json")
    if ref["freeze_sha256"] != freeze:
        raise ValueError("NHANES comparators do not match the frozen model")
    torch.set_num_threads(6)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    columns = ["ID", "age", *OUTCOMES, *bundle["features"]]
    frame = frame.reindex(columns=columns)
    values = score_frame(frame, BUNDLE, device)
    if not values.ID.equals(frame.ID):
        raise ValueError("Predictions do not align with cohort IDs")
    for col in OUTCOMES:
        values[col] = frame[col]
    x, _, age_z = prepare_features(frame, bundle)
    values["age_sex_lp"] = linear_score(frame[["age", "sex"]].to_numpy(), ref["age_sex"])
    values["ridge_lp"] = linear_score(np.c_[x, age_z], ref["ridge"])
    for name, features in {"full": bundle["features"], **bundle["groups"]}.items():
        expected = frame[features].notna().sum(axis=1)
        if not expected.equals(values[f"{name}_n_observed"]) or not expected.gt(0).equals(values[f"{name}_lp"].notna()):
            raise ValueError(f"Score coverage differs from observed inputs for {name}")
    path = out / "predictions.csv.gz"
    write_csv(path, values)
    write_json(out / "audit.json", {**index_audit, "lab_audit": lab_audit, "formal_validation_run": True,
               "scoring_device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
               "n_scored": len(frame), "deaths_365d": int(frame.event.sum()),
               "deaths_index_hospital": int(frame.hospital_event.sum()),
               "n_age_topcoded": int(idx.set_index("subject_id").loc[frame.ID, "anchor_age"].eq(91).sum()),
               "marker_count_quartiles": values.full_n_observed.quantile([.25, .5, .75]).tolist(),
               "features_with_data": int(frame[bundle["features"]].notna().any().sum()),
               "at_least_22_markers": int(values.full_n_observed.ge(22).sum()),
               "features_coverage": {k:int(v) for k, v in frame[bundle["features"]].count().items()},
               "predictions_sha256": digest(path), "vitals_sha256": vitals_audit["vitals_sha256"],
               "model_freeze_sha256": freeze, "nhanes_comparator_sha256": digest(REFERENCE)})
    print(f"Frozen model scored {len(values):,} people on {device}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "index", "labs", "score"])
    parser.add_argument("--archive", type=Path, default=MIMIC_ARCHIVE)
    parser.add_argument("--data-root", type=Path, default=HERE / "local_3_1")
    parser.add_argument("--out", type=Path, default=HERE / "formal_v3_1")
    parser.add_argument("--chunk-size", type=int, default=750_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "extract":
        extract(args.archive, args.data_root)
    elif args.stage == "index":
        prepare(args.data_root, args.out)
    elif args.stage == "labs":
        labs(args.data_root, args.out, args.chunk_size)
    else:
        score(args.out)
