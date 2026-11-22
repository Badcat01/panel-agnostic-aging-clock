"""Build dated 2016–2022 healthspan endpoints on the original 2016 VBS roster."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pyreadstat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from healthspan_input import CONDS, run as aggregate_endpoints
from tracker import month_start, sha
from stage_tracker_2022 import KEYS, OLD_KEYS, baseline_crosswalk, clean_ids

WAVES = {13: 2016, 14: 2018, 15: 2020, 16: 2022}
ADL = ["dressa", "batha", "eata", "beda", "toilta"]
GZIP = {"method": "gzip", "mtime": 0}


def id_pair(frame, hhid, pn):
    household = frame[hhid].astype("string").str.strip()
    person = frame[pn].astype("string").str.strip()
    if not household.str.fullmatch(r"\d{1,6}").fillna(False).all() or not person.str.fullmatch(r"\d{1,3}").fillna(False).all():
        raise ValueError("Unexpected HHID or PN")
    ids = household.str.zfill(6) + "-" + person.str.zfill(3)
    if ids.duplicated().any():
        raise ValueError("Duplicate participants")
    return ids


def dates(year, month):
    return month_start(year, month) + pd.Timedelta(days=14)


def align_product(frame, baseline_ids, current_ids):
    """Keep direct product links first; use the Tracker's documented ID change only for absent rows."""
    if not frame.index.is_unique:
        raise ValueError("Source records must be unique by participant")
    baseline = pd.Series(baseline_ids, index=baseline_ids)
    direct = baseline.isin(frame.index)
    recovered = ~direct & current_ids.isin(frame.index)
    selected = baseline.where(direct, current_ids.where(recovered))
    if selected.dropna().duplicated().any():
        raise ValueError("More than one VBS participant maps to one product record")
    aligned = frame.reindex(selected.tolist())
    aligned.index = baseline_ids
    return aligned, {"direct_matches": int(direct.sum()), "documented_former_id_matches": int(recovered.sum()),
                     "unmatched": int(selected.isna().sum()), "baseline": len(baseline)}


def read_langa(file, baseline_ids, current_ids):
    cols = ["hhid", "pn"] + [f"{stub}{year}" for year in WAVES.values()
                            for stub in ("cogfunction", "proxy", "interview")]
    source, meta = pyreadstat.read_dta(file, usecols=cols)
    source.index = id_pair(source, "hhid", "pn")
    source, counts = align_product(source, baseline_ids, current_ids)
    waves = []
    for wave, year in WAVES.items():
        label = meta.column_names_to_labels[f"cogfunction{year}"]
        if not all(s in label for s in ("1=Normal", "2=CIND", "3=Demented")):
            raise ValueError(f"Verify official cognition codes for {year}")
        seen = source[f"interview{year}"].eq(1)
        category = source[f"cogfunction{year}"].where(seen)
        proxy = source[f"proxy{year}"].where(seen)
        if not category.dropna().isin([1, 2, 3]).all() or not proxy.dropna().isin([1, 2, 5]).all():
            raise ValueError(f"Unknown classification or proxy type for {year}")
        waves.append(pd.DataFrame({"wave": wave, "lw": category.map({1: 0, 2: 1, 3: 1}),
                                   "lw_proxy": proxy.map({1: 1, 2: 1, 5: 0})}))
    return pd.concat(waves).rename_axis("ID").reset_index(), counts


def build(args):
    args.out.mkdir(parents=True, exist_ok=True)
    vbs, _ = pyreadstat.read_sav(args.vbs, usecols=["HHID", "PN", "PVBS_N_DAYS"])
    vbs = clean_ids(vbs, KEYS)
    tr, _ = pyreadstat.read_dta(args.raw_tracker, usecols=[*KEYS, *OLD_KEYS, "OVYEAR", "PIWYEAR", "PIWMONTH"])
    crosswalk, tracker_counts = baseline_crosswalk(vbs, tr)
    vbs.index = id_pair(vbs, "HHID", "PN")
    crosswalk.index = id_pair(crosswalk, "HHID", "PN")
    current_ids = (crosswalk.current_hhid + "-" + crosswalk.current_pn).reindex(vbs.index)
    tr.index = id_pair(tr, "HHID", "PN")
    tr, _ = align_product(tr, vbs.index, current_ids)
    draw = dates(tr.PIWYEAR, tr.PIWMONTH) + pd.to_timedelta(vbs.PVBS_N_DAYS, unit="D")
    fields = ["iwstat", "iwendy", "iwendm", "proxy", *ADL, *CONDS]
    requested = ["hhidpn", "raeduc"] + [f"r{wave}{field}" for wave in WAVES for field in fields]
    rand, meta = pyreadstat.read_dta(args.rand, usecols=requested)
    if set(rand.columns) != set(requested):
        raise ValueError(f"Incomplete RAND release: {set(requested).difference(rand.columns)}")
    digits = pd.to_numeric(rand.hhidpn, errors="raise").astype("int64").astype(str).str.zfill(9)
    rand.index = digits.str[:-3] + "-" + digits.str[-3:]
    if not rand.index.is_unique:
        raise ValueError("RAND key collision")
    rand, rand_counts = align_product(rand, vbs.index, current_ids)
    ep = pd.read_csv(args.tracker, dtype={"HHID": "string", "PN": "string"})
    ep.index = id_pair(ep, "HHID", "PN")
    if set(ep.index) != set(vbs.index):
        raise ValueError("Tracker endpoint roster differs from VBS")
    end = pd.Timestamp(json.loads(Path(str(args.tracker) + ".json").read_text())["cutoff"])
    latest = dates(rand.r16iwendy, rand.r16iwendm).where(rand.r16iwstat.isin([1, 2])).max()
    health_end = min(end, latest + pd.offsets.MonthEnd(0))
    days = pd.Timedelta(days=365.25)
    baseline = pd.DataFrame({"education": rand.raeduc.map({1: 1, 2: 2, 3: 2, 4: 3, 5: 3}),
                             "horizon": (health_end - draw) / days}).rename_axis("ID")
    waves = []
    for w in WAVES:
        z = rand[[f"r{w}{f}" for f in fields]].copy()
        z.columns = fields
        dt = dates(z.iwendy, z.iwendm)
        observed = z.iwstat.isin([1, 2]) & dt.notna() & dt.le(end) & draw.notna()
        t = (dt - draw) / days
        if w == min(WAVES):
            t = pd.Series(0., index=z.index)
        else:
            observed &= t.gt(0)
        adl = z[ADL].apply(lambda v: v.map({0: 0, 1: 1, 2: 1}))
        positive = adl.eq(1).any(axis=1)
        outcome = z[CONDS].where(z[CONDS].isin([0, 1]))
        outcome["adl"] = positive.astype(float).where(positive | adl.notna().all(axis=1))
        outcome["time"] = t
        outcome["proxy"] = z.proxy.where(z.proxy.isin([0, 1]))
        outcome = outcome.where(observed, axis=0)
        outcome["wave"] = w
        waves.append(outcome)
    visits = pd.concat(waves).rename_axis("ID").reset_index()
    lw, langa_counts = read_langa(args.langa, vbs.index, current_ids)
    visits = visits.merge(lw, how="left", on=["ID", "wave"], validate="one_to_one")
    visits.loc[visits.time.isna(), ["lw", "lw_proxy"]] = float("nan")
    if visits.duplicated(["ID","wave"]).any() or len(visits) != len(vbs) * len(WAVES):
        raise ValueError("Visit roster inconsistent")
    baseline_file = args.out / "baseline.csv.gz"
    visits_file = args.out / "visits.csv.gz"
    langa_file = args.out / "langa.csv.gz"
    health_file = args.out / "health.csv.gz"
    baseline.to_csv(baseline_file, compression=GZIP)
    visits.drop(columns=["lw"]).to_csv(visits_file, index=False, compression=GZIP)
    visits[["ID","wave","lw"]].to_csv(langa_file, index=False, compression=GZIP)
    aggregate_endpoints(args.tracker, health_file, langa_file, visits_file, baseline_file)
    audit = {
        "input_sha256": {k: sha(getattr(args, k)) for k in ["vbs","raw_tracker","tracker","langa","rand"]},
        "tracker_administrative_cutoff": str(end.date()), "visit_administrative_cutoff": str(health_end.date()),
        "respondents": len(vbs), "respondents_with_draw_date": int(draw.notna().sum()),
        "respondents_missing_from_rand": int(rand.hhidpn.isna().sum()),
        "ID_linkage": {"raw_tracker": tracker_counts, "RAND": rand_counts, "Langa_Weir": langa_counts},
        "baseline_education_available": int(baseline.education.notna().sum()),
        "ADL_items": ADL, "chronic_conditions": CONDS, "RAND_fields": meta.column_names_to_labels,
        "langa_mapping": "interview=1, cogfunction 1=normal; 2=CIND, 3=dementia; proxy 1,2=proxy,5=self",
        "health_output_sha256": sha(health_file),
        "waves": {year: {"observed": int(s.time.notna().sum()), "ADL": int(s.adl.notna().sum()),
                          "fully_observed_chronic": int(s[CONDS].notna().all(axis=1).sum()),
                          "official_LW": int(s.lw.notna().sum()),
                          "official_LW_proxy": int((s.lw_proxy.eq(1) & s.lw.notna()).sum())}
                  for w, s in visits.groupby("wave") for year in [WAVES[w]]}
    }
    health = pd.read_csv(health_file)
    for endpoint in ("adl", "lw"):
        status = health["st_" + endpoint]
        times = health["ti_" + endpoint]
        if not (times[status.notna()].gt(0).all() and times.notna().equals(status.notna()) and status.dropna().isin([0, 1, 2]).all()):
            raise ValueError(f"Invalid derived {endpoint} endpoint")
        risk_n = int((visits.wave.eq(13) & visits[endpoint].eq(0)).sum())
        audit[endpoint] = {"baseline_at_risk": risk_n, "positive_time": int(status.notna().sum()),
                           "incident": int(status.eq(1).sum()), "competing_death": int(status.eq(2).sum()),
                           "censored_at_last_observation": int(status.eq(0).sum()), "at_risk_without_time": risk_n - int(status.notna().sum())}
    new = health.new_conditions.dropna()
    if not new.between(0,len(CONDS)).all() or not health.years_at_new_conditions.dropna().gt(0).all():
        raise ValueError("Invalid chronic count or observation time")
    audit["chronic_counts"] = {"observed": len(new), "new_diagnoses": int(new.sum()),
                               "respondents_with_at_least_one": int(new.gt(0).sum())}
    (args.out / "health_verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2),encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("vbs", "raw_tracker", "tracker", "rand", "langa", "out"):
        parser.add_argument("--" + arg.replace("_","-"), type=Path, required=True)
    build(parser.parse_args())
