"""Stream timed liver, kidney and metabolic measurements from the verified release."""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as arrow_csv

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
MIMIC = RUN.parent
sys.path.insert(0, str(MIMIC))
from mimic_iv_run import digest, write_csv, write_json

ITEMS = {50861: "Alanine Aminotransferase (ALT)", 50912: "Creatinine", 50931: "Glucose"}
COLS = ["subject_id", "hadm_id", "labevent_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"]
DTYPES = {column: pa.int64() for column in COLS[:4]}
DTYPES.update(charttime=pa.timestamp("s"), storetime=pa.timestamp("s"), valuenum=pa.float64(), valueuom=pa.string())


def make_index(hosp, formal):
    index = pd.read_csv(formal / "landmark_index.csv.gz", parse_dates=["admittime", "landmark"])
    admitted = pd.read_csv(hosp / "admissions.csv.gz", usecols=["hadm_id", "subject_id", "dischtime", "deathtime"],
                          parse_dates=["dischtime", "deathtime"])
    pred = pd.read_csv(formal / "predictions.csv.gz", usecols=["ID", "full_lp"])
    assert pred.ID.is_unique and pred.full_lp.notna().all()
    index = index[index.subject_id.isin(pred.ID)].merge(admitted, on=["subject_id", "hadm_id"], validate="one_to_one")
    index["stop_time"] = index[["dischtime", "deathtime"]].min(axis=1)
    assert len(index) == len(pred) == 130242 and index.subject_id.is_unique and index.hadm_id.is_unique
    assert index.stop_time.gt(index.landmark).all() and index.landmark.sub(index.admittime).eq(pd.Timedelta(days=1)).all()
    index["ID"] = index.subject_id
    return index[["ID", "subject_id", "hadm_id", "admittime", "landmark", "dischtime", "stop_time", "age", "sex", "hospital_event"]]


def run():
    dest = RUN / "cache"
    dest.mkdir(exist_ok=True)
    source = MIMIC / "local_3_1"
    hosp = source / "hosp"
    formal = MIMIC / "formal_v3_1"
    manifest = json.loads((source / "verified_members.json").read_text(encoding="utf8"))
    frozen = json.loads((formal / "audit.json").read_text(encoding="utf8"))
    if digest(formal / "predictions.csv.gz") != frozen["predictions_sha256"]:
        raise ValueError("Frozen prediction file differs from prior audit")
    for key in ["labevents", "admissions", "d_labitems"]:
        if digest(hosp / f"{key}.csv.gz") != manifest[key]["sha256"]:
            raise ValueError(f"Archive member hash differs for {key}")
    dict_rows = pd.read_csv(hosp / "d_labitems.csv.gz").set_index("itemid", verify_integrity=True).loc[list(ITEMS)]
    assert dict_rows["label"].to_dict() == ITEMS and dict_rows.fluid.str.lower().eq("blood").all()
    index = make_index(hosp, formal)
    write_csv(dest / "incident_index.csv.gz", index)
    windows = index[["subject_id", "hadm_id", "admittime", "landmark", "stop_time"]].copy()
    windows["lookback"] = windows.admittime - pd.Timedelta(days=7)
    windows["horizon"] = windows.landmark + pd.Timedelta(days=7)
    itemids = pa.array(list(ITEMS), type=pa.int64())
    subjects = pa.array(windows.subject_id.to_numpy(), type=pa.int64())
    tmp = dest / "serial_labs.csv.gz.tmp"
    out_cols = ["subject_id", "hadm_id", "source_hadm_id", *COLS[2:]]
    counts = {key: 0 for key in ITEMS}
    scanned = 0
    with pa.input_stream(str(hosp / "labevents.csv.gz"), compression="gzip") as native, gzip.open(
            tmp, mode="wt", encoding="utf8", newline="", compresslevel=4) as out:
        reader = arrow_csv.open_csv(native, read_options=arrow_csv.ReadOptions(block_size=1 << 25),
                                    parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
                                    convert_options=arrow_csv.ConvertOptions(include_columns=COLS, column_types=DTYPES,
                                                                               strings_can_be_null=True))
        out.write(",".join(out_cols) + "\n")
        for batch_number, batch in enumerate(reader, start=1):
            scanned += batch.num_rows
            hit = pc.and_(pc.is_in(batch["itemid"], value_set=itemids), pc.is_in(batch["subject_id"], value_set=subjects))
            subset = batch.filter(hit).to_pandas().rename(columns={"hadm_id": "source_hadm_id"})
            mapped = subset.merge(windows, on="subject_id", validate="many_to_one")
            same_stay = mapped.source_hadm_id.eq(mapped.hadm_id) & mapped.charttime.ge(mapped.admittime)
            prior_kidney = mapped.itemid.eq(50912) & mapped.charttime.lt(mapped.admittime) & mapped.charttime.ge(mapped.lookback)
            keep = (same_stay | prior_kidney) & mapped.charttime.lt(mapped.stop_time) & mapped.charttime.le(mapped.horizon)
            admitted = mapped.loc[keep, out_cols].copy()
            admitted["source_hadm_id"] = admitted.source_hadm_id.astype("Int64")
            admitted.to_csv(out, index=False, header=False)
            for item, n in admitted.itemid.value_counts().items():
                counts[int(item)] += int(n)
            if batch_number % 70 == 0:
                print(f"serial labevents: {scanned:,} rows scanned, {sum(counts.values()):,} timed records", flush=True)
    if scanned != 158374764 or not all(counts.values()):
        raise ValueError(f"Unexpected official laboratory coverage: {scanned}, {counts}")
    ready = dest / "serial_labs.csv.gz"
    tmp.replace(ready)
    write_json(dest / "serial_labs_audit.json", {"source_rows": scanned, "retained_rows_by_item": counts,
               "source_sha256": manifest["labevents"]["sha256"], "index_sha256": digest(dest / "incident_index.csv.gz"),
               "serial_labs_sha256": digest(ready), "original_predictions_sha256": frozen["predictions_sha256"]})
    print(f"Serial cache verified: {sum(counts.values()):,} records, {len(index):,} indexed patients", flush=True)


if __name__ == "__main__":
    run()
