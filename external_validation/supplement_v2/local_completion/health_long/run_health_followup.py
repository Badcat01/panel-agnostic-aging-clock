"""Reconstruct the observed HRS proxy classification and run analyses without exporting IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys

import numpy as np
import pandas as pd
import pyreadstat as pr

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / "healthspan"
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from study_paths import COMMUNITY_ROOT
IADL = ("phonea", "medsa", "mealsa", "moneya", "shopa")


def read_columns(path, columns):
    """Read the handful of columns needed, irrespective of Stata field case."""
    fields = {x.lower(): x for x in pr.read_dta(path, metadataonly=True)[1].column_names}
    frame, metadata = pr.read_dta(path, usecols=[fields[x] for x in columns], user_missing=True)
    frame.columns = frame.columns.str.lower()
    return frame, metadata


def joined_id(frame):
    return frame.hhid.astype(str).str.strip().str.zfill(6) + "-" + frame.pn.astype(str).str.strip().str.zfill(3)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source(data, name):
    return next(path for path in data.rglob("*.dta") if path.name.lower() == name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rscript", default="Rscript")
    args = ap.parse_args()
    data = COMMUNITY_ROOT
    bases = pd.read_csv(PREVIOUS / "hrs_baseline.csv.gz", dtype={"ID": str}).set_index("ID", verify_integrity=True)
    visits = pd.read_csv(PREVIOUS / "hrs_visits.csv.gz", dtype={"ID": str})
    vars_by_wave = {
        wave: [f"r{wave}{name}" for name in ("proxy", "cog27", "prmem", *IADL)]
        for wave in (13, 14, 15)
    }
    physical_vars = [f"r{w}{m}" for w in (12, 13, 14) for m in ("grp", "timwlk", "iwendy", "iwendm", "iwstat")]
    rand_path = source(data, "randhrs1992_2020v2.dta")
    rand, _ = read_columns(rand_path, ["hhid", "pn", *physical_vars, *sum(vars_by_wave.values(), [])])
    rand.index = joined_id(rand)
    assert rand.index.is_unique and bases.index.isin(rand.index).all()
    rand = rand.loc[bases.index]

    checks = {}
    specs = {13: (2016, "p"), 14: (2018, "q"), 15: (2020, "r")}
    cohort_files = {rand_path, PREVIOUS / "hrs_baseline.csv.gz", PREVIOUS / "hrs_visits.csv.gz"}
    results = []
    for wave, (year, prefix) in specs.items():
        field = f"{prefix}a011"
        a_path = source(data, f"h{str(year)[-2:]}a_r.dta")
        assess, _ = read_columns(a_path, ["hhid", "pn", field])
        assess.index = joined_id(assess)
        assert assess.index.is_unique
        cohort_files.add(a_path)
        published = visits.loc[visits.wave.eq(wave)].set_index("ID", verify_integrity=True).reindex(bases.index)
        respondent = rand[f"r{wave}proxy"]
        self_score = pd.to_numeric(rand[f"r{wave}cog27"], errors="coerce")
        memory = rand[f"r{wave}prmem"].map({i: i - 1 for i in range(1, 6)})
        interviewer = assess[field].reindex(rand.index).map({1: 0, 2: 1, 3: 2})
        # RAND .z = does not do the activity but could do it without difficulty.
        # Generic "don't do" and other missing categories have unknown ability.
        iadl = rand[[f"r{wave}{item}" for item in IADL]].apply(lambda col: col.map({0: 0, 1: 1, 2: 1, "z": 0}))
        limitations = iadl.sum(axis=1, min_count=5)
        proxy_score = memory + interviewer + limitations
        self_valid = respondent.eq(0) & self_score.between(0, 27) & published.time.notna()
        self_class = pd.Series(np.select([self_score.le(6), self_score.le(11)], [2., 1.], default=0.), index=rand.index).where(self_valid)
        self_dementia = self_class.eq(2).astype(float).where(self_class.notna())
        pd.testing.assert_series_equal(self_dementia, published.lw_self_dementia, check_names=False, check_dtype=False)
        is_proxy = respondent.isin([1, 2]) & published.time.notna()
        proxy_valid = is_proxy & proxy_score.between(0, 11)
        proxy_class = pd.Series(np.select([proxy_score.ge(6), proxy_score.ge(3)], [2., 1.], default=0.), index=rand.index).where(proxy_valid)
        combined = self_class.combine_first(proxy_class)
        checks[str(year)] = {
            "self_scored": int(self_valid.sum()),
            "dated_proxy": int(is_proxy.sum()),
            "proxy_scored": int(proxy_valid.sum()),
            "proxy_missing_memory": int((is_proxy & memory.isna()).sum()),
            "proxy_missing_interviewer": int((is_proxy & interviewer.isna()).sum()),
            "proxy_missing_any_iadl": int((is_proxy & limitations.isna()).sum()),
            "proxy_dementia": int(proxy_class.eq(2).sum()),
            "proxy_cind": int(proxy_class.eq(1).sum()),
            "previous_self_dementia_exact": True,
        }
        results.append(pd.DataFrame({"time": published.time, "wave": wave, "self_class": self_class,
                                     "combined_class": combined, "proxy_score": proxy_score.where(proxy_valid)}))

    # HRS 2016 VBS spans the alternate physical-measurement half-samples. 2014/2018 pairs
    # are dated against the 2016 visit. Analyze follow-up levels conditional on earlier levels.
    tests = pd.DataFrame(index=bases.index)
    for wave in (12, 13, 14):
        for measure in ("grp", "timwlk"):
            val = pd.to_numeric(rand[f"r{wave}{measure}"], errors="coerce")
            valid = rand[f"r{wave}iwstat"].isin([1, 2]) & val.gt(0)
            tests[f"{measure}_{wave}"] = val.where(valid)
        year = pd.to_numeric(rand[f"r{wave}iwendy"], errors="coerce")
        month = pd.to_numeric(rand[f"r{wave}iwendm"], errors="coerce")
        tests[f"date_{wave}"] = pd.to_datetime(dict(year=year, month=month, day=15), errors="coerce")
    tests["follow_2018_years"] = results[1]["time"]
    tests["interval_2014_2018"] = (tests.date_14 - tests.date_12).dt.days / 365.25
    for measure in ("grp", "timwlk"):
        checks[f"pairs_{measure}"] = {
            f"{a}_{b}": int(tests[[f"{measure}_{a}", f"{measure}_{b}"]].notna().all(axis=1).sum())
            for a, b in ((12, 14), (13, 14))
        }

    with tempfile.TemporaryDirectory(prefix="clock_health_extension_") as temp:
        tmp = Path(temp)
        pd.concat(results).to_csv(tmp / "hrs_cognition.csv", index_label="ID")
        tests.to_csv(tmp / "hrs_physical.csv", index_label="ID")
        subprocess.run([args.rscript, str(HERE / "fit_health_extensions.R"), str(PREVIOUS),
                        str(HERE), str(tmp)], check=True, env={**os.environ, "LC_ALL": "C"})
    speed_files = ["charls_w5_baseline.csv.gz", "charls_w5_visits.csv.gz", "charls_3_pace.csv.gz",
                   "elsa_baseline.csv.gz", "elsa_visits.csv.gz", "elsa_4_pace.csv.gz", "elsa_6_pace.csv.gz",
                   "elsa_visits_w2_to_w9.csv.gz"]
    cohort_files.update(PREVIOUS / name for name in speed_files)
    checks["source_sha256"] = {f.relative_to(COMMUNITY_ROOT).as_posix(): digest(f) for f in sorted(cohort_files)}
    (HERE / "health_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
