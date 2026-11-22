"""Researcher-run input preparation for the HRS 2022 Tracker v2 and Weights v1."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pyreadstat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tracker import derive_intervals


KEYS = ["HHID", "PN"]
OLD_KEYS = ["OVHHID", "OVPN"]
STATUS = ["SALIVE", "RALIVE", "QALIVE"]
ENDPOINT = ["PAGE", "PIWYEAR", "PIWMONTH", "SEX", "BIRTHYR", "BIRTHMO",
            "KNOWNDECEASEDYR", "KNOWNDECEASEDMO", "LASTALIVEYR", "LASTALIVEMO", *STATUS]
DESIGN = ["VBSI16WGTRA", "STRATUM", "SECU"]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def clean_ids(frame, columns, optional=False):
    result = frame.copy()
    for col, width in zip(columns, (6, 3)):
        s = result[col].astype("string").str.strip()
        s = s.mask(s.eq(""))
        valid = s.str.fullmatch(r"[0-9]{1," + str(width) + r"}").fillna(False)
        if not (valid | (optional & s.isna())).all():
            raise ValueError(f"Invalid identifier in {col}")
        result[col] = s.str.zfill(width)
    return result


def baseline_crosswalk(roster, tracking):
    """Prefer direct IDs; resolve a prior ID only where the current ID is absent."""
    current = ["current_hhid", "current_pn"]
    tr = clean_ids(clean_ids(tracking, KEYS), OLD_KEYS, optional=True)
    if tr.duplicated(KEYS).any():
        raise ValueError("Current Tracker identifiers are not unique")
    ids = tr[KEYS].copy()
    ids[current] = tr[KEYS]
    direct = roster[KEYS].merge(ids, how="left", on=KEYS, validate="one_to_one")
    prior = tr.loc[tr.OVYEAR.gt(2016) & tr[OLD_KEYS].notna().all(axis=1),
                   [*KEYS, *OLD_KEYS]].rename(columns=dict(zip(KEYS, current)) | dict(zip(OLD_KEYS, KEYS)))
    unresolved = direct[current[0]].isna()
    recovered = direct.loc[unresolved, KEYS].merge(prior, on=KEYS, how="left", validate="one_to_one")
    direct.loc[unresolved, current] = recovered[current].to_numpy()
    linked = direct[current].notna().all(axis=1)
    if direct.loc[linked].duplicated(current).any():
        raise ValueError("More than one baseline participant maps to the same Tracker person")
    return direct, {
        "direct_matches": int((~unresolved).sum()),
        "documented_former_id_matches": int(recovered[current[0]].notna().sum()),
        "unmatched": int((~linked).sum()),
    }


def build(roster, tracking, weights, cutoff):
    roster = clean_ids(roster, KEYS)
    tracking = clean_ids(tracking, KEYS)
    weights = clean_ids(weights, KEYS)
    if roster.duplicated(KEYS).any() or weights.duplicated(KEYS).any():
        raise ValueError("VBS roster or weights contain duplicate identifiers")
    if not tracking.SEX.dropna().isin([1, 2]).all():
        raise ValueError("Unknown SEX code")
    for name in STATUS:
        if not tracking[name].dropna().isin([1, 2, 5, 6]).all():
            raise ValueError(f"Unknown {name} status")
    if not tracking.KNOWNDECEASEDMO.dropna().isin([*range(1, 13), 98]).all():
        raise ValueError("Unexpected death month code")
    if not weights.SECU.dropna().isin([1, 2, 99]).all():
        raise ValueError("Unexpected sampling half-unit code")
    if not weights.STRATUM.dropna().isin([*range(1, 81), 99]).all():
        raise ValueError("Unexpected sampling stratum code")
    keys, counts = baseline_crosswalk(roster, tracking)
    names = {"HHID": "current_hhid", "PN": "current_pn"}
    linked = roster.merge(keys, how="left", on=KEYS, validate="one_to_one")
    cov = tracking[[*KEYS, *ENDPOINT]].rename(columns=names | {"SEX": "GENDER"})
    work = linked.merge(cov, how="left", on=list(names.values()), validate="many_to_one")
    target, audit = derive_intervals(work, cutoff, STATUS)
    design = keys.merge(weights[[*KEYS, *DESIGN]].rename(columns=names), how="left",
                        on=list(names.values()), validate="many_to_one")
    design = design[[*KEYS, *DESIGN]].rename(columns={
        "VBSI16WGTRA": "weight", "STRATUM": "stratum", "SECU": "psu"})
    design[["stratum", "psu"]] = design[["stratum", "psu"]].replace(99, float("nan"))
    design["weight"] = design.weight.where(design.weight.gt(0))
    audit.update(crosswalk=counts, n_positive_weight=int(design.weight.notna().sum()),
                 n_positive_weight_with_design=int(design[["weight", "stratum", "psu"]].notna().all(axis=1).sum()))
    return target, design, audit


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args):
    tr_file = args.release_dir / "trk2022tr_r.dta"
    wt_file = args.release_dir / "wgts2022tr_r.dta"
    vbs, _ = pyreadstat.read_sav(args.vbs, usecols=[*KEYS, "PVBS_N_DAYS"])
    tracking, _ = pyreadstat.read_dta(tr_file, usecols=[*KEYS, *OLD_KEYS, "OVYEAR", *ENDPOINT])
    weights, _ = pyreadstat.read_dta(wt_file, usecols=[*KEYS, *DESIGN])
    target, design, audit = build(vbs, tracking, weights, args.cutoff)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for role, frame in [("tracker", target), ("design", design)]:
        file = args.out / f"{role}.csv.gz"
        frame.to_csv(file, index=False, compression={"method": "gzip", "mtime": 0})
        outputs.append(dict(role=role, file=str(file.resolve()), sha256=digest(file),
                            columns={c: c for c in frame if c not in KEYS}))
    audit.update(vbs_sha256=digest(args.vbs), tracker_sha256=digest(tr_file),
                 weights_sha256=digest(wt_file), tracker_version="HRS 2022 Tracker v2.0, October 2026",
                 weights_version="HRS 2022 Tracker Weights v1.0, October 2026",
                 coverage_note="Selected administrative upper bound. Tracker does not certify complete death ascertainment through a single day.")
    write_json(args.out / "tracker.csv.gz.json", audit)
    write_json(args.out / "tracker_source_map.json", dict(
        raw_tracker_file=str(tr_file.resolve()), raw_tracker_sha256=audit["tracker_sha256"],
        raw_vbs_file=str(args.vbs.resolve()), raw_vbs_sha256=audit["vbs_sha256"],
        tracker_version=audit["tracker_version"], tracker_cutoff=args.cutoff,
        tracker_derivation="stage_tracker_2022.py plus frozen tracker.py::derive_intervals; HRS-known death, clipped last-known-alive, 2016 baseline household ID",
        sources=outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbs", required=True, type=Path)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="Explicit project administrative upper date YYYY-MM-DD")
    run(parser.parse_args())
