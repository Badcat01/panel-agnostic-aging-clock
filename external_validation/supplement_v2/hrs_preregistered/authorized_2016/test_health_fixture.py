"""Exercise the full healthspan build on invented RAND, Langa–Weir, VBS and Tracker tables."""
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyreadstat

from build_health import ADL, CONDS, WAVES, build


def verify(tmp):
    n = 8
    identifiers = [str(i).zfill(6) for i in range(1, n + 1)]
    keys = pd.DataFrame({"HHID": identifiers, "PN": ["001"] * n})
    vbs = keys.assign(PVBS_N_DAYS=0)
    raw = keys.assign(PIWYEAR=2016, PIWMONTH=3, OVHHID="", OVPN="", OVYEAR=np.nan)
    rand = pd.DataFrame({"hhidpn": [1000*i+1 for i in range(1, n+1)], "raeduc": [1,2,3,4,5,2,4,np.nan]})
    lw = keys.rename(columns={"HHID":"hhid","PN":"pn"}).copy()
    labels = {}
    for wave, year in WAVES.items():
        prefix = f"r{wave}"
        values = {"iwstat":1, "iwendy":year, "iwendm":3, "proxy":0, **dict.fromkeys(ADL + CONDS,0)}
        for field, value in values.items():
            rand[prefix+field] = np.full(n,value, dtype=float)
        lw[f"cogfunction{year}"] = 1
        labels[f"cogfunction{year}"] = "1=Normal, 2=CIND, 3=Demented"
        lw[f"proxy{year}"] = 5
        lw[f"interview{year}"] = 1
        if wave>13:
            rand.loc[[1,4],prefix+"iwstat"] = 5
            lw.loc[[1,4],f"interview{year}"] = 99
            rand.loc[0,[prefix+"batha",prefix+"diabe",prefix+"hearte"]] = 1
            lw.loc[0,f"cogfunction{year}"] = 2
            lw.loc[0,f"proxy{year}"] = 1
        if wave>=15:
            lw.loc[5,f"cogfunction{year}"] = 2
            rand.loc[5,prefix+"dressa"] = 1
    rand.loc[0,"r16cancre"] = 1
    rand.loc[3,["r13dressa","r13diabe"]] = [1,np.nan]
    lw.loc[3,"cogfunction2016"] = 2
    lw.loc[6, "cogfunction2022"] = 3
    rand.loc[6,"r16eata"] = 2
    for i in (0,6):
        raw.loc[i,["OVHHID","OVPN","OVYEAR"]] = [keys.loc[i,"HHID"],keys.loc[i,"PN"],2021.]
        raw.loc[i,"HHID"] = f"{90+i:06d}"
    rand.loc[0,"hhidpn"] = 90001
    lw.loc[6,"hhid"] = "000096"
    events = keys.assign(event=[0,1,0,0,0,1,0,0],time_yr=[7,1,7,7,7,3,7,7],time_upper_yr=[7,1.1,7,7,7,3.1,7,7])
    args = SimpleNamespace(vbs=tmp/"vbs.sav", raw_tracker=tmp/"raw.dta", tracker=tmp/"tracker.csv.gz",
                           langa=tmp/"langa.dta", rand=tmp/"rand.dta", out=tmp/"out")
    pyreadstat.write_sav(vbs,args.vbs)
    pyreadstat.write_dta(raw,args.raw_tracker)
    pyreadstat.write_dta(rand,args.rand)
    pyreadstat.write_dta(lw,args.langa,column_labels=labels)
    events.to_csv(args.tracker,index=False)
    Path(str(args.tracker)+".json").write_text(json.dumps({"cutoff":"2023-12-31"}))
    build(args)
    results = pd.read_csv(args.out/"health.csv.gz")
    for endpoint in ("adl","lw"):
        s = results["st_"+endpoint]
        np.testing.assert_allclose(s, [1,2,0,np.nan,np.nan,2,1,0], equal_nan=True)
        np.testing.assert_allclose(results.loc[[1,5],"ti_"+endpoint],[1,3])
        assert 5.9 < results.loc[6,"ti_"+endpoint] < 6.1
    np.testing.assert_allclose(results.new_conditions, [3,np.nan,0,np.nan,np.nan,0,0,0],equal_nan=True)
    meta = json.loads((args.out/"health_verification.json").read_text())
    assert meta["waves"]["2022"]["official_LW_proxy"]==1
    assert meta["lw"]["baseline_at_risk"]==7 and meta["lw"]["at_risk_without_time"]==1
    assert meta["ID_linkage"]["raw_tracker"] == {"direct_matches":6,"documented_former_id_matches":2,"unmatched":0}
    for name in ("RAND","Langa_Weir"):
        assert meta["ID_linkage"][name] == {"direct_matches":7,"documented_former_id_matches":1,"unmatched":0,"baseline":8}
    return {"type":"synthetic only","participants":n,"planned_visit_waves":list(WAVES.values()),
            "baseline_prevalence_excluded":True,"first_observed_incident":True,"loss_without_followup_excluded":True,
            "death_timing_checked":True,"post_death_observations_excluded":True,"proxy_CIND_included":True,
            "year_2022_dementia_included":True,"repeated_chronic_not_duplicated":True,
            "incomplete_baseline_chronic_excluded":True,"documented_tracker_product_ID_changes_verified":True}


if __name__=="__main__":
    with TemporaryDirectory() as d:
        print(json.dumps(verify(Path(d)),ensure_ascii=False,indent=2))
