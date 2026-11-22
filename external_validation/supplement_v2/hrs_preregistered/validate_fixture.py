"""Cross-check the fixture, family-stable folds, bootstrap completion, hypothesis thresholds and input mapping."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from prepare import source

HERE=Path(__file__).resolve().parent
OUT=HERE/"synthetic/tested"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path=HERE/"synthetic/fixture.csv.gz"
    d=pd.read_csv(path,dtype={"ID":str})
    eligible=d.qc_pass.eq(1)&d.age.ge(50)&d.time_yr.gt(0)
    fixture=d.loc[eligible]
    audits=json.loads((OUT/"run_status.json").read_text(encoding="utf8"))
    assert audits["mode"]=="synthetic" and audits["input_hash"]==sha(path)
    assert audits["household_split_violations"]==0 and audits["eligible"]==len(fixture)
    tbl=pd.read_csv(OUT/"discrimination.csv")
    main=tbl.query("analysis=='primary'").iloc[0]
    survey=tbl.query("analysis=='survey_design'").iloc[0]
    for check in [main,survey]:
        assert check.requested==check.successful==2000 and check.lower<check.upper
        assert np.isclose(check.delta,check.C_clinical-check.C_dnam)
        assert check.n==len(fixture) and check.deaths==fixture.event.sum()
    assert (main.H1=="supported")==(main.lower>-.01)
    lrt=pd.read_csv(OUT/"LRT.csv"); h=lrt[lrt.analysis.eq("primary")].iloc[0]
    assert (str(h.H2).lower()=="true")==(max(h.p_add_clinical,h.p_add_dnam)<.01)
    assert tbl[tbl.analysis.eq("one_year_landmark")].n.iloc[0]==fixture.time_yr.gt(1).sum()
    sec=pd.read_csv(OUT/"healthspan.csv")
    assert set(sec.endpoint)=={"adl","lw","new_conditions"} and set(sec.term)=={"a","b"}
    refs=pd.read_csv(OUT/"VBS_reference_matrix.csv")
    assert refs.panel.nunique()==4 and refs.target.nunique()==5
    missing=pd.read_csv(OUT/"VBS_reference_unavailable.csv")
    assert set(missing.marker)=={"ntprobnp"}
    with tempfile.TemporaryDirectory() as folder:
        temp=Path(folder)
        sample=temp/"fixture_mapping.csv"
        pd.DataFrame({"HHID":["012345","000001","012345"],"PN":["001","001","005"],
                      "TELO":[3.2,5.3,999],"QC":[1,0,1]}).to_csv(sample,index=False)
        spec=dict(role="dnam",file=sample.name,sha256=sha(sample),columns={"clock_DNAmTL":"TELO"},
                  missing_codes={"clock_DNAmTL":[999]},qc_pass={"field":"QC","values":[1]},
                  reverse_direction=["clock_DNAmTL"])
        mapped,qc=source(spec,temp)
        assert qc.tolist()==[True,False,True] and mapped.index[0]=="012345-001"
        np.testing.assert_allclose(mapped.clock_DNAmTL,[-3.2,np.nan,np.nan],equal_nan=True)
    summary=dict(status="synthetic_end_to_end_pass",real_HRS_DNAm_data_used=False,synthetic_input_rows=len(d),
        eligible=int(main.n),households=int(main.families),synthetic_deaths=int(main.deaths),
        all_households_in_single_fold=True,household_bootstrap_full_refits=int(main.successful),
        stratified_PSU_bootstrap_full_refits=int(survey.successful),missing_code_and_official_QC_mapping_tested=True,
        H1_threshold_verified=True,H2_both_LRT_thresholds_verified=True,
        secondary_outcome_models=sorted(sec.model.unique()),VBS_reference_markers=int(refs.target.nunique()),
        unavailable_cardiovascular_score_recorded=True,
        verified_outputs={p.name:sha(p) for p in sorted(OUT.glob("*.csv"))})
    (OUT/"verification.json").write_text(json.dumps(summary,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
