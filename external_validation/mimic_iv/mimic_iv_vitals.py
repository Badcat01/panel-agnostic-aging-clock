"""Extract ICU-charted features available at the hospital 24-hour landmark."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as arrow_csv


PREFIX = "mimic-iv-3.1/"
ITEMS = {
    220045: ("Heart Rate", "bpm", 10., 300.),
    220179: ("Non Invasive Blood Pressure systolic", "mmhg", 20., 350.),
    220180: ("Non Invasive Blood Pressure diastolic", "mmhg", 5., 250.),
    226512: ("Admission Weight (Kg)", "kg", 20., 450.),
    226707: ("Height", "inch", 120. / 2.54, 230. / 2.54),
    226730: ("Height (cm)", "cm", 120., 230.),
}
PAIR_ITEMS = [220179, 220180]
MEASUREMENT_COLUMNS = ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"]
KEYS = ["subject_id", "hadm_id"]
FEATURES = ["sbp", "dbp", "pulse", "bmi"]


class HashedStream(io.RawIOBase):
    """Hash the nested gzip bytes while passing them to the CSV reader."""

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, destination):
        block = self.source.read(len(destination))
        destination[:len(block)] = block
        self.digest.update(block)
        self.bytes_read += len(block)
        return len(block)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(archive):
    manifest = archive.read(PREFIX + "SHA256SUMS.txt").decode("ascii")
    entries = [line.split(maxsplit=1) for line in manifest.strip().splitlines()]
    return {member.lstrip("*"): digest for digest, member in entries}


def read_verified(archive, manifest, member, **options):
    data = archive.read(PREFIX + member)
    if hashlib.sha256(data).hexdigest() != manifest[member]:
        raise ValueError(f"Internal SHA-256 mismatch in {member}")
    return pd.read_csv(io.BytesIO(gzip.decompress(data)), **options)


def dictionary_check(dictionary):
    labels = dictionary.set_index("itemid", verify_integrity=True).reindex(ITEMS)
    cases = pd.Series({item: spec[0].casefold() for item, spec in ITEMS.items()})
    units = pd.Series({item: spec[1] for item, spec in ITEMS.items()})
    if not labels.label.astype("string").str.casefold().eq(cases).fillna(False).all():
        raise ValueError("An ICU item label differs from the v3.1 phenotype definition")
    if not labels.unitname.astype("string").str.casefold().eq(units).fillna(False).all():
        raise ValueError("An ICU item unit differs from the v3.1 phenotype definition")
    if not labels.linksto.eq("chartevents").all():
        raise ValueError("An ICU item does not originate from chartevents")


def align_stays(stays, index):
    if not stays.stay_id.is_unique:
        raise ValueError("ICU stay_id is not unique")
    match = stays.merge(index, on=KEYS, how="inner", validate="many_to_one")
    keep = (match.intime.lt(match.landmark) & match.outtime.ge(match.admittime) &
            match.outtime.gt(match.intime))
    return match.loc[keep].set_index("stay_id", verify_integrity=True)


def take_measurements(archive, manifest, stays, audit):
    member = "icu/chartevents.csv.gz"
    member_size = archive.getinfo(PREFIX + member).file_size
    stay_ids = pa.array(stays.index, type=pa.int64())
    itemids = pa.array(list(ITEMS), type=pa.int64())
    counters = {"source_rows": 0, "rows_candidate_by_stay_item": 0,
                "rows_within_window": 0, "valid_rows": 0, "invalid_unit_rows": 0, "invalid_value_rows": 0}
    candidates = []
    types = {name: pa.int64() for name in [*KEYS, "stay_id", "itemid"]}
    types.update(charttime=pa.timestamp("s"), storetime=pa.timestamp("s"), valuenum=pa.float64(), valueuom=pa.string())
    expected_unit = {item: spec[1] for item, spec in ITEMS.items()}
    lower = {item: spec[2] for item, spec in ITEMS.items()}
    upper = {item: spec[3] for item, spec in ITEMS.items()}
    unit_failures = {}
    with archive.open(PREFIX + member) as source:
        hashed = HashedStream(source)
        native = pa.PythonFile(io.BufferedReader(hashed, 1 << 20), mode="r")
        compressed = pa.CompressedInputStream(native, "gzip")
        reader = arrow_csv.open_csv(
            compressed,
            read_options=arrow_csv.ReadOptions(block_size=1 << 26),
            parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
            convert_options=arrow_csv.ConvertOptions(include_columns=MEASUREMENT_COLUMNS,
                                                       column_types=types, strings_can_be_null=True),
        )
        for batch_number, records in enumerate(reader, start=1):
            counters["source_rows"] += records.num_rows
            match = pc.and_(pc.is_in(records["itemid"], value_set=itemids),
                            pc.is_in(records["stay_id"], value_set=stay_ids))
            rows = records.filter(match).to_pandas()
            counters["rows_candidate_by_stay_item"] += len(rows)
            rows = rows.merge(stays, on=[*KEYS, "stay_id"], how="inner", validate="many_to_one")
            time_valid = (rows.charttime.ge(rows.admittime) & rows.charttime.lt(rows.landmark) &
                          rows.charttime.ge(rows.intime) & rows.charttime.le(rows.outtime) &
                          rows.storetime.ge(rows.charttime) & rows.storetime.le(rows.landmark))
            rows = rows.loc[time_valid].copy()
            counters["rows_within_window"] += len(rows)
            source_unit = rows.valueuom.astype("string").str.casefold().str.strip()
            source_unit = source_unit.replace({"in": "inch", "inches": "inch", "beats/min": "bpm"})
            unit_valid = source_unit.eq(rows.itemid.map(expected_unit)).fillna(False)
            counters["invalid_unit_rows"] += int((~unit_valid).sum())
            for (item, unit), count in rows.loc[~unit_valid].assign(unit=source_unit.fillna("[missing]")).groupby(
                    ["itemid", "unit"]).size().items():
                key = f"{item}:{unit}"
                unit_failures[key] = unit_failures.get(key, 0) + int(count)
            value_valid = (np.isfinite(rows.valuenum) & rows.valuenum.ge(rows.itemid.map(lower)) &
                           rows.valuenum.le(rows.itemid.map(upper)))
            counters["invalid_value_rows"] += int((unit_valid & ~value_valid).sum())
            ready = rows.loc[unit_valid & value_valid, [*KEYS, "stay_id", "itemid", "charttime", "storetime", "valuenum"]]
            counters["valid_rows"] += len(ready)
            if len(ready):
                ready = ready.sort_values(["charttime", "storetime", "valuenum"], kind="stable")
                cuff = ready.itemid.isin(PAIR_ITEMS)
                # Preserve matching timestamps for both cuff readings across CSV batch boundaries.
                candidates.append(ready.loc[cuff].drop_duplicates(["stay_id", "charttime", "itemid"]))
                candidates.append(ready.loc[~cuff].drop_duplicates(["stay_id", "itemid"]))
            if batch_number == 1 or batch_number % 15 == 0:
                progress = min(hashed.bytes_read / member_size, 1.)
                print(f"ICU chartevents: {progress:.1%} read; {counters['source_rows']:,} rows scanned", flush=True)
        reader.close()
        if hashed.bytes_read != member_size or hashed.digest.hexdigest() != manifest[member]:
            raise ValueError("Internal SHA-256 mismatch in chartevents")
    audit.update(counters)
    audit["unsupported_units"] = unit_failures
    if not candidates:
        raise ValueError("No verified ICU vital or anthropometry rows at the landmark")
    return pd.concat(candidates, ignore_index=True)


def compose_features(rows):
    sort = ["charttime", "storetime", "stay_id", "valuenum"]
    rows = rows.sort_values(sort, kind="stable")
    cuff = rows.loc[rows.itemid.isin(PAIR_ITEMS)].drop_duplicates(["stay_id", "charttime", "itemid"])
    paired = cuff.pivot(index=[*KEYS, "stay_id", "charttime"], columns="itemid", values="valuenum")
    paired = paired.reindex(columns=PAIR_ITEMS).rename(columns={220179: "sbp", 220180: "dbp"}).reset_index()
    paired = paired.loc[paired.sbp.gt(paired.dbp)].sort_values(["charttime", "stay_id"], kind="stable")
    blood_pressure = paired.drop_duplicates(KEYS).loc[:, [*KEYS, "sbp", "dbp"]]
    pulse = rows.loc[rows.itemid.eq(220045)].drop_duplicates(KEYS).rename(columns={"valuenum": "pulse"})
    pulse = pulse.loc[:, [*KEYS, "pulse"]]

    weight = rows.loc[rows.itemid.eq(226512)].drop_duplicates("stay_id")
    weight = weight.loc[:, [*KEYS, "stay_id", "charttime", "valuenum"]].rename(
        columns={"charttime": "weight_time", "valuenum": "weight_kg"}
    )
    height = rows.loc[rows.itemid.isin([226707, 226730])].copy()
    height["height_cm"] = height.valuenum.where(height.itemid.eq(226730), height.valuenum * 2.54)
    height = height.sort_values(["charttime", "storetime", "itemid"], ascending=[True, True, False], kind="stable")
    height = height.drop_duplicates("stay_id").loc[:, [*KEYS, "stay_id", "charttime", "height_cm"]]
    ht_wt = height.merge(weight, on=[*KEYS, "stay_id"], how="inner", validate="one_to_one")
    ht_wt["bmi"] = ht_wt.weight_kg / (ht_wt.height_cm / 100.)**2
    ht_wt["ready_time"] = ht_wt[["charttime", "weight_time"]].max(axis=1)
    ht_wt = ht_wt.loc[ht_wt.bmi.between(8., 120.)].sort_values(["ready_time", "stay_id"], kind="stable")
    bmi = ht_wt.drop_duplicates(KEYS).loc[:, [*KEYS, "bmi"]]
    return blood_pressure.merge(pulse, on=KEYS, how="outer", validate="one_to_one").merge(
        bmi, on=KEYS, how="outer", validate="one_to_one"
    ).loc[:, [*KEYS, *FEATURES]].sort_values("hadm_id").reset_index(drop=True)


def validate_index(index):
    if not index.subject_id.is_unique or not index.hadm_id.is_unique:
        raise ValueError("The index must have one first admission per participant")
    if index[KEYS].isna().any().any() or index[["landmark", "admittime"]].isna().any().any():
        raise ValueError("The index contains missing identifiers or dates")
    if not index.landmark.sub(index.admittime).eq(pd.Timedelta(hours=24)).all():
        raise ValueError("The index must be anchored exactly 24 hours after admission")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = pd.read_csv(args.index, usecols=[*KEYS, "admittime", "landmark"], parse_dates=["admittime", "landmark"])
    validate_index(index)
    audit = {"release": "official-3.1", "index_sha256": file_hash(args.index), "index_rows": len(index)}
    with ZipFile(args.archive) as archive:
        manifest = source_manifest(archive)
        dictionary = read_verified(archive, manifest, "icu/d_items.csv.gz", usecols=["itemid", "label", "unitname", "linksto"])
        dictionary_check(dictionary)
        icu = read_verified(archive, manifest, "icu/icustays.csv.gz", usecols=[*KEYS, "stay_id", "intime", "outtime"],
                            parse_dates=["intime", "outtime"])
        stays = align_stays(icu, index)
        audit["icu_stays_overlapping_landmark_window"] = len(stays)
        audit["people_with_icu_stay_overlapping_landmark_window"] = int(stays.subject_id.nunique())
        measurements = take_measurements(archive, manifest, stays, audit)
        audit["source_member_sha256"] = {path: manifest[path] for path in
                                        ["icu/d_items.csv.gz", "icu/icustays.csv.gz", "icu/chartevents.csv.gz"]}
    result = compose_features(measurements)
    if not result.hadm_id.is_unique or not result.subject_id.is_unique:
        raise ValueError("Vital feature assembly duplicated a participant or admission")
    audit["people_with_any_icu_feature"] = len(result)
    audit["people_with_cardiovascular_features"] = int(result[["sbp", "dbp", "pulse"]].notna().any(axis=1).sum())
    audit["people_with_cardiovascular_complete"] = int(result[["sbp", "dbp", "pulse"]].notna().all(axis=1).sum())
    audit["people_per_feature"] = {name: int(result[name].notna().sum()) for name in FEATURES}
    audit["feature_definition"] = {
        "sbp_dbp": "Earliest paired non-invasive cuff readings with matching ICU stay and charttime; SBP > DBP.",
        "pulse": "Earliest valid charted ICU heart rate; proxy for the original resting pulse.",
        "bmi": "Earliest paired charted admission weight (kg) and height (cm or inch) in the same ICU stay. "
               "Chart time supports availability; the measurement may be a previously documented height.",
        "bounds": {item: {"label": spec[0], "unit": spec[1], "low": spec[2], "high": spec[3]} for item, spec in ITEMS.items()},
    }
    audit["timing"] = ("Chart time at or after hospital admission, before the 24-hour landmark and within the ICU stay. "
                       "Store time between chart time and landmark, inclusive.")
    args.out.mkdir(parents=True, exist_ok=True)
    temp = args.out / "vitals.csv.gz.tmp"
    result.to_csv(temp, index=False, compression={"method": "gzip", "mtime": 0})
    temp.replace(args.out / "vitals.csv.gz")
    audit["vitals_sha256"] = file_hash(args.out / "vitals.csv.gz")
    temp = args.out / "vitals_audit.json.tmp"
    temp.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(args.out / "vitals_audit.json")
    print(f"Verified and wrote ICU features for {len(result):,} people", flush=True)


if __name__ == "__main__":
    main()
