"""Independent invariants for cohort sizes, score sums, available-time cutoffs and exported risk maps."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from shared import HERE, TRAIN_RUN, sha, verify

import duckdb

def verify_icu():
    db=duckdb.connect(str(HERE/"icu_cache/time_limited.duckdb"),read_only=True)
    sql=lambda s: db.execute(s).fetchone()[0]
    n=sql("SELECT count(*) FROM cohort")
    assert n==sql("SELECT count(distinct subject_id) FROM cohort")==27005
    assert sql("SELECT count(*) FROM cohort WHERE intime<admittime OR intime>landmark OR outtime<=cutoff OR cutoff!=intime+INTERVAL '1 day'")==0
    assert sql("SELECT count(*) FROM hosp.diagnoses_icd")==0
    report={"unique_patients":n,"concepts":sql("SELECT count(*) FROM information_schema.tables WHERE table_schema='derived'")}
    for table,time in [("icu.inputevents","starttime"),("icu.outputevents","charttime"),("icu.chartevents","charttime")]:
        fields=[time,"storetime"]+(["endtime"] if time=="starttime" else [])
        late=" OR ".join("x."+f+">c.cutoff" for f in fields)
        q=f"SELECT count(*) FROM {table} x JOIN cohort c USING(stay_id) WHERE {late}"
        assert sql(q)==0
        report[table+"_late_rows"]=0
    assert sql("SELECT count(*) FROM hosp.labevents x JOIN cohort c USING(subject_id) WHERE x.storetime>c.cutoff OR x.charttime>c.cutoff")==0
    report["hosp.labevents_late_rows"]=0
    spans={}
    for name,target in [("oasis","oasis"),("sapsii","sapsii"),("first_day_sofa","sofa")]:
        cols=[r[0] for r in db.execute("describe derived."+name).fetchall()]
        components=[s for s in cols if s.endswith("_score")] if target!="sofa" else ["respiration","coagulation","liver","cardiovascular","cns","renal"]
        total="+".join("coalesce("+c+",0)" for c in components)
        assert sql("SELECT count(*) FROM derived."+name+" WHERE "+target+"!=("+total+")")==0
        spans[target]=db.execute(f"SELECT min({target}),max({target}) FROM derived.{name}").fetchone()
    assert sql("SELECT max(coalesce(comorbidity_score,0)) FROM derived.sapsii")==0
    report["score_min_max"]=spans
    db.close()
    patients=pd.read_csv(HERE/"icu_cache/severity_scores.csv.gz")
    rows=pd.read_csv(HERE/"icu_severity.csv")
    for row in rows.itertuples():
        np.testing.assert_allclose(roc_auc_score(patients[row.endpoint],patients[row.comparator]),row.severity_AUC,atol=1e-9)
        np.testing.assert_allclose(roc_auc_score(patients[row.endpoint],patients.full_lp),row.clock_AUC,atol=1e-9)
    return report

def main():
    verify()
    freeze=sha(TRAIN_RUN/"model_freeze.json")
    assert freeze=="a472b93f36c8d49de51d2c55c720aa801fa0e08d65c71575986e3bd995b20ae1"
    results={"freeze_sha256":freeze,"icu":verify_icu()}
    cols=pd.read_csv(HERE/"reference_lp_reproduction.csv")
    assert len(cols)==16 and cols.max_abs_difference.max()<1e-6
    results["linear_projection_max_abs_error"]=cols.max_abs_difference.max()
    s=pd.concat(pd.read_csv(p) for p in (HERE/"survival").glob("*_C.csv") if p.name!="mhas_C.csv")
    assert len(s)==8*3*2 and (s.B==2000).all() and s.C.between(0,1).all()
    for col in ["lower","upper"]:
        assert s[col].between(0,1).all()
    results["survival_model_metric_rows"]=len(s)
    f=pd.read_csv(HERE/"horizons.csv")
    assert len(f)==(7*3+1)*3
    v=f[f.supported]
    assert v.cumulative_dynamic_AUC.between(0,1).all() and v.ipcw_brier.between(0,1).all()
    decision=pd.read_csv(HERE/"horizon_decisions.csv")
    assert not decision.duplicated(["cohort","horizon","model","threshold"]).any()
    assert np.all(decision[decision.model.eq("treat_none")].net_benefit==0)
    results["supported_cohort_horizons"]=len(v)//3
    audit=pd.read_csv(HERE/"assay_audit_documented.csv")
    assert len(audit)==10*43 and audit.loc[audit.present,"raw_mapping"].notna().all()
    results["marker_audit_cells"]=len(audit)
    results["sourced_observed_cells"]=int(audit.present.sum())
    kidney=pd.read_csv(HERE/"readmission_matched.csv").query("endpoint=='kidney' and code_position=='principal' and eligibility=='one_or_more'").iloc[0]
    assert kidney.n==82513 and kidney.events==669
    results["icu"]["derivation_commit"]="303d26c623dcc9c49cc0f204468d4acc2f063797"
    (HERE/"verification.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(results,ensure_ascii=True))

if __name__=="__main__":
    main()
