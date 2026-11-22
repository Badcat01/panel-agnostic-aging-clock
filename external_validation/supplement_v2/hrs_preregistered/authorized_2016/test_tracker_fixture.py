"""Test the 2022 Tracker adapter with invented records in a temporary directory."""
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from stage_tracker_2022 import KEYS, STATUS, digest, run


def check():
    # These six sequential IDs and every measurement below are invented.
    records = pd.DataFrame({"HHID": [f"{i:06d}" for i in range(1, 7)], "PN": ["001"] * 6})
    v = records.assign(PVBS_N_DAYS=10.0)
    t = records.iloc[:5].assign(OVHHID="", OVPN="", OVYEAR=np.nan, AGE=60.0,
                                 PAGE=65.0, PIWYEAR=2016.0, PIWMONTH=1.0, SEX=1.0,
                                 BIRTHYR=1950.0, BIRTHMO=12.0, LASTALIVEYR=2023.0,
                                 LASTALIVEMO=12.0, KNOWNDECEASEDYR=np.nan, KNOWNDECEASEDMO=np.nan)
    for field in STATUS:
        t[field] = 1.0
    t.loc[1, ["HHID", "OVHHID", "OVPN", "OVYEAR"]] = ["000007", "000002", "001", 2018.0]
    t.loc[1, ["SEX", "LASTALIVEYR", "LASTALIVEMO", "KNOWNDECEASEDYR", "KNOWNDECEASEDMO", "SALIVE"]] = [2., 2021., 3., 2021., 98., 5.]
    t.loc[2, ["KNOWNDECEASEDYR", "KNOWNDECEASEDMO", "SALIVE", "LASTALIVEMO"]] = [2023., 98., 5., 1.]
    t.loc[3, "QALIVE"] = 5.
    t.loc[4, ["BIRTHMO", "PAGE"]] = [0., 66.]
    w = t[KEYS].assign(VBSI16WGTRA=100.0, STRATUM=1., SECU=1.)
    w.loc[4, ["STRATUM", "SECU", "VBSI16WGTRA"]] = [99., 99., np.nan]
    with tempfile.TemporaryDirectory() as tmp:
        where = Path(tmp)
        pyreadstat.write_sav(v, where / "vbs.sav")
        pyreadstat.write_dta(t, where / "trk2022tr_r.dta")
        pyreadstat.write_dta(w, where / "wgts2022tr_r.dta")
        out = where / "derived"
        run(Namespace(release_dir=where, out=out, vbs=where / "vbs.sav", cutoff="2023-07-31"))
        d = pd.read_csv(out / "tracker.csv.gz", dtype={"HHID": str, "PN": str})
        weights = pd.read_csv(out / "design.csv.gz", dtype={"HHID": str, "PN": str})
        audit = json.loads((out / "tracker.csv.gz.json").read_text(encoding="utf-8"))
        mapping = json.loads((out / "tracker_source_map.json").read_text(encoding="utf-8"))
        assert d[KEYS].equals(records) and weights[KEYS].equals(records)
        assert audit["crosswalk"] == dict(direct_matches=4, documented_former_id_matches=1, unmatched=1)
        assert audit["death_interval_crosses_cutoff"] == 1 and audit["death_missing_date"] == 1
        assert d.event.notna().sum() == 3 and d.event.sum() == 1
        assert d.loc[1, "sex"] == 0 and d.loc[1, "time_lower_yr"] < d.loc[1, "time_yr"] < d.loc[1, "time_upper_yr"]
        assert d.loc[0, "time_yr"] == (pd.Timestamp("2023-07-31") - pd.Timestamp("2016-01-25")).days / 365.25
        assert d.loc[4, "age"] == 66 and d.loc[5, ["event", "time_yr", "age", "sex"]].isna().all()
        assert weights.loc[1, "weight"] == 100 and weights.loc[4, ["weight", "stratum", "psu"]].isna().all()
        for source in mapping["sources"]:
            assert digest(Path(source["file"])) == source["sha256"]


if __name__ == "__main__":
    check()
    print("Passed synthetic Tracker 2022 and weights adapter checks.")
