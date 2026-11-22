"""Compute time-available MIMIC severity comparators from versioned MIT-LCP SQL."""
import gzip
import hashlib
import json
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from study_paths import MIMIC_ARCHIVE
from shared import EXT,HERE,read,sha
RES=HERE.parent/"resources"
import duckdb
import sqlglot
MIM=EXT/"mimic_iv"
sys.path.insert(0,str(MIM))
from mimic_iv_panels import compare_auc,ci95,oof_comparison

CODE=RES/"mimic-code"/"mimic-iv"/"concepts"
WORK=HERE/"icu_cache"
def inputs():
    WORK.mkdir(exist_ok=True)
    extract=WORK/"sources";extract.mkdir(exist_ok=True)
    cs={p.stem:p for p in CODE.rglob("*.sql")}
    deps={}; order=[]; raw=set()
    def visit(name):
        if name in deps:return
        s=cs[name].read_text(encoding="utf8")
        refs=re.findall(r"physionet-data\.mimiciv_(derived|hosp|icu)\.(\w+)",s)
        deps[name]=s
        for t,n in refs:
            if t=="derived":visit(n)
            else:raw.add((t,n))
        order.append(name)
    for n in ["oasis","sapsii","first_day_sofa"]:visit(n)
    paths={}
    with zipfile.ZipFile(MIMIC_ARCHIVE) as z:
        prefix="mimic-iv-3.1/"
        manifest={p.lstrip("*"):s for s,p in (x.split(maxsplit=1) for x in z.read(prefix+"SHA256SUMS.txt").decode().splitlines())}
        for schema,name in sorted(raw):
            member=f"{schema}/{name}.csv.gz"
            path=MIM/"local_3_1"/member
            if not path.exists():path=extract/(name+".csv.gz")
            if not path.exists():
                tmp=path.with_suffix(".part")
                with z.open(prefix+member) as a,tmp.open("wb") as b:shutil.copyfileobj(a,b,4<<20)
                assert sha(tmp)==manifest[member];tmp.replace(path)
            else:assert sha(path)==manifest[member]
            paths[name]=path
    return deps,order,paths

def main():
    deps,order,paths=inputs()
    db=duckdb.connect(str(WORK/"time_limited.duckdb"))
    db.execute("SET threads=4");db.execute("SET memory_limit='8GB'");db.execute("SET preserve_insertion_order=false")
    icu=pd.read_csv(paths["icustays"],parse_dates=["intime","outtime"])
    idx=pd.read_csv(MIM/"formal_v3_1"/"landmark_index.csv.gz",parse_dates=["admittime","landmark"])
    pred_path=MIM/"formal_v3_1"/"predictions.csv.gz"
    assert sha(pred_path)==read(MIM/"formal_v3_1"/"audit.json")["predictions_sha256"]
    pred=pd.read_csv(pred_path).set_index("ID",verify_integrity=True)
    r=icu.merge(idx[["hadm_id","subject_id","admittime","landmark"]],on=["hadm_id","subject_id"],validate="many_to_one")
    r=r[r.subject_id.isin(pred.index)&r.intime.ge(r.admittime)&r.intime.le(r.landmark)].sort_values(["subject_id","intime","stay_id"])
    r=r.drop_duplicates("subject_id");r["cutoff"]=r.intime+pd.Timedelta(days=1)
    r=r[r.outtime>r.cutoff].copy()
    # Recorded exact or linked death may precede administrative discharge.
    since=(r.cutoff-r.landmark)/pd.Timedelta(days=365)
    r=r[(pred.loc[r.subject_id,"hospital_time_yr"].to_numpy()>since)&(pred.loc[r.subject_id,"time_yr"].to_numpy()>since)]
    assert r.subject_id.is_unique
    db.register("selected",r);db.execute("CREATE OR REPLACE TABLE cohort AS SELECT * FROM selected")
    for schema in ["icu","hosp","derived"]:db.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    schema_for=lambda name:"icu" if name in ["icustays","chartevents","inputevents","outputevents"] else "hosp"
    for name in ["icustays","patients","admissions","services","diagnoses_icd","outputevents","inputevents","labevents","chartevents"]:
        schema=schema_for(name)
        exists=db.execute("select count(*) from information_schema.tables where table_schema=? and table_name=?",[schema,name]).fetchone()[0]
        if exists:continue
        types={"labevents":{"value":"VARCHAR"},"chartevents":{"value":"VARCHAR"},"diagnoses_icd":{"icd_code":"VARCHAR"}}
        types_opt=", types="+str(types[name]) if name in types else ""
        p=paths[name].as_posix().replace("'","''")
        raw=f"read_csv('{p}',header=true,sample_size=100000,ignore_errors=false{types_opt})"
        if name=="icustays":join="v.stay_id=c.stay_id"
        elif name=="patients":join="v.subject_id=c.subject_id"
        elif name in ["admissions","services"]:join="v.hadm_id=c.hadm_id"+(" AND v.transfertime<=c.cutoff" if name=="services" else "")
        elif name=="diagnoses_icd":join="false" # all index admissions are the first observed
        elif name=="labevents":join="v.subject_id=c.subject_id AND (v.hadm_id=c.hadm_id OR v.hadm_id is null) AND v.charttime between c.intime - interval '6 hours' and c.cutoff AND v.storetime between v.charttime and c.cutoff"
        elif name=="inputevents":join="v.stay_id=c.stay_id AND v.starttime between c.intime - interval '6 hours' and c.cutoff AND v.storetime<=c.cutoff"
        else:join="v.stay_id=c.stay_id AND v.charttime between c.intime - interval '6 hours' and c.cutoff AND v.storetime between v.charttime and c.cutoff"
        print("raw",name,flush=True)
        # Limit derived notes and dose intervals to the prediction landmark.
        cols="v.* REPLACE (least(v.endtime,c.cutoff) AS endtime)" if name=="inputevents" else "v.*"
        db.execute(f"CREATE TABLE {schema}.{name} AS SELECT {cols} FROM {raw} v JOIN cohort c ON {join}")
    for name in order:
        s=re.sub(r"`physionet-data\.mimiciv_(derived|hosp|icu)\.(\w+)`",r"\1.\2",deps[name])
        translated=sqlglot.transpile(s,read="bigquery",write="duckdb")[0]
        print("concept",name,flush=True)
        db.execute(f"CREATE OR REPLACE TABLE derived.{name} AS "+translated)
    scores=db.execute("SELECT c.subject_id,c.stay_id,c.hadm_id,c.admittime,c.landmark,c.cutoff,o.oasis,s.sapsii,f.sofa FROM cohort c JOIN derived.oasis o USING(stay_id) JOIN derived.sapsii s USING(stay_id) JOIN derived.first_day_sofa f USING(stay_id)").df()
    scores=scores.set_index("subject_id",verify_integrity=True).join(pred[["full_lp","hospital_event","event","time_yr","hospital_time_yr"]],validate="one_to_one")
    assert len(scores)==len(r) and scores[["sapsii","sofa","oasis"]].notna().all().all()
    scores.to_csv(WORK/"severity_scores.csv.gz")
    metrics=[]
    # Fixed 365 days from original hospital landmark; ICU landmark follows it by at most one day.
    for outcome in ["hospital_event","event"]:
        y=scores[outcome].to_numpy(bool)
        for severity in ["oasis","sapsii","sofa"]:
            x=scores[severity].to_numpy(float);clock=scores.full_lp.to_numpy()
            a,cov,_=compare_auc(np.column_stack([x,clock]),y)
            delta=a[1]-a[0];ci=ci95(delta,cov[0,0]+cov[1,1]-2*cov[0,1],(-1,1))
            add=oof_comparison(x[:,None],clock,y,20261121)
            metrics.append(dict(endpoint=outcome,comparator=severity,n=len(y),events=y.sum(),severity_AUC=a[0],clock_AUC=a[1],
                                difference=delta,lo=ci[0],hi=ci[1],**add))
    pd.DataFrame(metrics).to_csv(HERE/"icu_severity.csv",index=False)
    db.close()

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
