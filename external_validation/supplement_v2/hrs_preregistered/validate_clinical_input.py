"""Test raw 2016 VBS and Tracker inputs using dummy clock values; do not fit or inspect HRS-DNAm associations."""
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from prepare import HERE,formal,run,sha
from tracker import derive
from build_hrs_elsa import HRS
from healthspan_input import run as extend_healthspan


def main():
    vbs=next(HRS.rglob("hrs2016vbs.sav"));trk=next(HRS.rglob("trk2020tr_r.dta"))
    endpoint,summary=derive(vbs,trk,"2021-12-31",["RALIVE","QALIVE"])
    _,old=formal("hrs")
    ep=endpoint.set_index(endpoint.HHID.str.strip()+"-"+endpoint.PN.str.strip())
    shared=ep.loc[old.index]
    for col in ["age","sex","time_yr","event","time_lower_yr","time_upper_yr"]:
        np.testing.assert_allclose(shared[col],old[col],atol=1e-8,equal_nan=True)
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        temp=Path(tmp); harmonized=temp/"tracker.csv.gz"
        endpoint.to_csv(harmonized,index=False,compression={"method":"gzip","mtime":0})
        health=temp/"health.csv.gz"
        extend_healthspan(harmonized,health)
        outcomes=json.loads(Path(str(health)+".json").read_text(encoding="utf8"))
        fake=temp/"fake.csv"; t=endpoint.iloc[:40][["HHID","PN"]].copy()
        t["dummy_score"]=np.arange(len(t));t["pass"]=1;t.to_csv(fake,index=False)
        config=dict(tracker_version="public historical clinical input test",tracker_cutoff="2021-12-31",raw_tracker_file=str(trk),
            raw_tracker_sha256=sha(trk),raw_vbs_file=str(vbs),raw_vbs_sha256=sha(vbs),tracker_derivation="tracker.py",
            sources=[dict(role="tracker",file=harmonized.name,sha256=sha(harmonized),
                          columns={c:c.upper() for c in ["age","sex","event","time_yr","time_lower_yr","time_upper_yr"]}),
                     dict(role="dnam",file=fake.name,sha256=sha(fake),columns={"clock_dummy":"dummy_score"},
                          qc_pass={"field":"pass","values":[1]})])
        desc=temp/"config.json";desc.write_text(json.dumps(config),encoding="utf8")
        path=temp/"prepared.csv.gz";run(desc,path)
        final=json.loads(Path(str(path)+".json").read_text(encoding="utf8"))
        out=pd.read_csv(path)
        assert (out.loc[out.age.le(80),"clinical"]==out.loc[out.age.le(80),"clinical_cap80"]).all()
        assert (out.loc[out.age.gt(85),"clinical"]!=out.loc[out.age.gt(85),"clinical_cap85"]).any()
        assert out.organ_cardiovascular.isna().all()
    result=dict(real_HRS_DNAm_used=False,clinical_input_dry_run_only=True,any_mortality_model_fitted_to_dummy_values=False,
        known_tracker="trk2020tr_r.dta",historical_tracker_check_n=len(shared),valid_same_age_frozen_predictions=final["previous_clinical_scores_verified"],
        full_valid_2016_clinical_roster=final["n_clinical"],dummy_rows_linked=final["n_DNAm_QC_matched"],tracker_audit=summary,
        available_ADL_followup_n=outcomes["adl_risk_and_followup_n"],available_chronic_count_n=outcomes["n_chronic_count"],
        official_langa_status=outcomes["official_langa_status"])
    (HERE/"synthetic/clinical_input_verification.json").write_text(json.dumps(result,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
