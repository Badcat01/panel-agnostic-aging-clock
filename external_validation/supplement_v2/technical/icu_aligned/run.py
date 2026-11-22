"""Align frozen clinical scores to the ICU severity-score landmark in the same patients."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from study_paths import MIMIC_ARCHIVE
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
TECH=OUT.parent
SUP=TECH.parent
EXT=SUP.parent
MIM=EXT/"mimic_iv"
CACHE=OUT/"cache"
CACHE.mkdir(exist_ok=True)
sys.path[:0]=[str(EXT/"scripts"),str(MIM)]
import duckdb
from reference import verify,BUNDLE,TRAIN_RUN,score_frame,sha
from mimic_iv_validate import MAP,DIFF
from mimic_iv_vitals import ITEMS,compose_features,dictionary_check,read_verified,source_manifest
from mimic_iv_panels import oof_comparison,compare_auc,ci95
from mimic_iv_run import write_csv,write_json


def panel(db,cohort,hours):
    db.execute(f"""create or replace temp view eligible_lab as
       select c.subject_id,c.hadm_id,x.labevent_id,x.charttime,x.storetime,c.intime,c.admittime,c.cutoff,m.feature,
              case when x.itemid=50889 then x.valuenum/10 else x.valuenum end as value
       from cohort c join hosp.labevents x using(subject_id,hadm_id) join labmap m on m.itemid=x.itemid
       where x.charttime>=c.intime - interval '{hours} hours' and x.charttime<c.cutoff
       and x.storetime between x.charttime and c.cutoff and isfinite(x.valuenum)
       and (x.valuenum>0 or (x.valuenum=0 and x.itemid in ({','.join(map(str,sorted(DIFF|{50889})))})))
       and (x.itemid not in ({','.join(map(str,sorted(DIFF)))}) or x.valuenum<=100)
       and (lower(trim(x.valueuom))=m.unit or (m.unit='iu/l' and lower(trim(x.valueuom))='u/l')
          or (m.unit='meq/l' and lower(trim(x.valueuom))='mmol/l'))""")
    first=db.execute("""select subject_id,hadm_id,feature,value,charttime,storetime,admittime,intime,cutoff from eligible_lab
       qualify row_number() over(partition by subject_id,hadm_id,feature order by charttime,storetime,labevent_id)=1""").df()
    assert first.storetime.le(first.cutoff).all() and first.charttime.ge(first.intime-pd.Timedelta(hours=hours)).all()
    labs=first.pivot(index=["subject_id","hadm_id"],columns="feature",values="value")
    vit=db.execute(f"""select c.subject_id,c.hadm_id,c.stay_id,x.itemid,x.charttime,x.storetime,x.valuenum
       from cohort c join icu.chartevents x using(subject_id,hadm_id,stay_id) join vitmap m on m.itemid=x.itemid
       where x.charttime>=c.intime-interval '{hours} hours' and x.charttime<c.cutoff
       and x.storetime between x.charttime and c.cutoff and x.valuenum between m.lo and m.hi and lower(trim(x.valueuom))=m.unit""").df()
    vitals=compose_features(vit).set_index(["subject_id","hadm_id"],verify_integrity=True)
    p=cohort.set_index(["subject_id","hadm_id"],verify_integrity=True).join(labs,validate="one_to_one").join(vitals,validate="one_to_one").reset_index()
    p["has_lab"]=p.subject_id.isin(first.subject_id)
    p["n_labs"]=p.reindex(columns=[m[0] for m in MAP.values()]).notna().sum(axis=1)
    p["n_vitals"]=p.reindex(columns=["sbp","dbp","pulse","bmi"]).notna().sum(axis=1)
    raw_null=db.execute(f"""select count(*),count(distinct c.subject_id) from cohort c join hosp.labevents x using(subject_id)
          join labmap m on m.itemid=x.itemid where x.hadm_id is null and x.charttime>=c.intime-interval '{hours} hours'
          and x.charttime<c.cutoff and x.storetime between x.charttime and c.cutoff""").fetchone()
    audit=dict(hours_before_intime=hours,n_target=len(p),n_with_lab=int(p.has_lab.sum()),n_any_marker=int((p.n_labs+p.n_vitals>0).sum()),
               n_vitals_only=int(((~p.has_lab)&(p.n_vitals>0)).sum()),eligible_first_lab_records=len(first),
               n_selected_lab_before_admission=int(first.loc[first.charttime.lt(first.admittime),"subject_id"].nunique()),
               n_selected_lab_before_icu=int(first.loc[first.charttime.lt(first.intime),"subject_id"].nunique()),
               raw_target_lab_rows_with_null_hadm_omitted=int(raw_null[0]),n_with_such_null_hadm_rows=int(raw_null[1]),
               median_n_labs=float(p.loc[p.has_lab,"n_labs"].median()),median_n_markers=float((p.n_labs+p.n_vitals)[p.has_lab].median()))
    return p,audit


def compare(data,window,target):
    y=data[target].to_numpy(dtype=bool)
    severity=["oasis","sapsii","sofa"]
    names=["hospital24","icu_minus6","icu_strict",*severity]
    avail=[n for n in names if data[n].notna().all()]
    score,cov,_=compare_auc(data[avail].to_numpy(),y)
    basic=[]; deltas=[]; inc=[]; j=avail.index("hospital24")
    for i,name in enumerate(avail):
        np.testing.assert_allclose(roc_auc_score(y,data[name]),score[i],atol=1e-12)
        basic.append(dict(population=window,outcome=target,score=name,n=len(data),events=int(y.sum()),AUC=score[i],
             lower=ci95(score[i],cov[i,i],(0,1))[0],upper=ci95(score[i],cov[i,i],(0,1))[1]))
        if name not in ["icu_minus6","icu_strict"]:continue
        for comparator in ["hospital24",*severity]:
            k=avail.index(comparator);diff=score[i]-score[k];lo,hi=ci95(diff,cov[k,k]+cov[i,i]-2*cov[k,i],(-1,1))
            deltas.append(dict(population=window,outcome=target,clock=name,comparator=comparator,n=len(data),events=int(y.sum()),
                               delta_AUC=diff,lower=lo,upper=hi))
    for clock in [x for x in names[:3] if x in avail]:
        for comp in severity:
            paired=oof_comparison(data[[comp]].to_numpy(),data[clock].to_numpy(),y,20261121)
            inc.append(dict(population=window,outcome=target,clock=clock,base_severity=comp,n=len(data),events=int(y.sum()),**paired))
    return basic,deltas,inc


def run():
    dbpath=TECH/"icu_cache/time_limited.duckdb";src=TECH/"icu_cache/severity_scores.csv.gz"
    official=MIM/"formal_v3_1"; op=official/"predictions.csv.gz"
    assert sha(op)==json.loads((official/"audit.json").read_text(encoding="utf8"))["predictions_sha256"]
    archive=MIMIC_ARCHIVE
    with zipfile.ZipFile(archive) as z:
        manifest=source_manifest(z)
        dictionary_check(read_verified(z,manifest,"icu/d_items.csv.gz",usecols=["itemid","label","unitname","linksto"]))
        labdict=read_verified(z,manifest,"hosp/d_labitems.csv.gz").set_index("itemid")
    labmap=pd.DataFrame([(i,*v) for i,v in MAP.items()],columns=["itemid","feature","label","unit"])
    assert labdict.loc[labmap.itemid,"label"].tolist()==labmap.label.tolist() and labdict.loc[labmap.itemid,"fluid"].str.lower().eq("blood").all()
    vitmap=pd.DataFrame([(i,*v[1:]) for i,v in ITEMS.items()],columns=["itemid","unit","lo","hi"])
    db=duckdb.connect(str(dbpath),read_only=True);db.execute("set threads=3")
    db.register("labmap",labmap);db.register("vitmap",vitmap)
    cohort=db.execute("select subject_id,hadm_id,stay_id,admittime,landmark,intime,outtime,cutoff from cohort order by subject_id").df()
    formal=pd.read_csv(op,usecols=["ID","age","full_lp","event","hospital_event"]).rename(columns={"ID":"subject_id","full_lp":"hospital24"})
    early=pd.read_csv(src,usecols=["subject_id","stay_id","hadm_id","oasis","sapsii","sofa","full_lp","event","hospital_event"])
    cohort=cohort.merge(early,on=["subject_id","stay_id","hadm_id"],validate="one_to_one").merge(formal,on=["subject_id","event","hospital_event"],validate="one_to_one")
    assert len(cohort)==27005 and cohort.subject_id.is_unique
    np.testing.assert_allclose(cohort.full_lp,cohort.hospital24)
    assert (cohort.intime+pd.Timedelta(days=1)).equals(cohort.cutoff) and cohort.outtime.gt(cohort.cutoff).all()
    torch.backends.mha.set_fastpath_enabled(False);torch.set_float32_matmul_precision("highest")
    bundle=verify()
    data=cohort[["subject_id","hadm_id","event","hospital_event","hospital24","oasis","sapsii","sofa"]].copy()
    audits=[]; files=[]
    for name,hours in [("icu_minus6",6),("icu_strict",0)]:
        p,a=panel(db,cohort[["subject_id","hadm_id","age"]],hours)
        path=CACHE/(name+"_panel.csv.gz");write_csv(path,p);files.append(path)
        inp=p.rename(columns={"subject_id":"ID"}).reindex(columns=["ID","age",*bundle["features"]])
        pred=score_frame(inp,BUNDLE,"cuda")
        np.testing.assert_array_equal(pred.ID,data.subject_id)
        np.testing.assert_array_equal(pred.full_n_observed,p.n_labs+p.n_vitals)
        data[name]=pred.full_lp.where(p.has_lab);data[name+"_any"]=pred.full_lp
        a.update(n_primary_with_score=int(data[name].notna().sum()),missing_primary=int(data[name].isna().sum()))
        audits.append(a)
    db.close()
    # Preserve the original ICU analysis' patient order and therefore its stratified folds at n=27,005.
    data=data.set_index("subject_id").loc[early.subject_id].reset_index()
    als=CACHE/"aligned_individual_scores.csv.gz";write_csv(als,data);files.append(als)
    both=data.icu_minus6.notna()&data.icu_strict.notna()
    assert (data.icu_minus6.notna()|data.icu_strict.isna()).all()
    any_panel=data.drop(columns=["icu_minus6","icu_strict"]).rename(columns={
        "icu_minus6_any":"icu_minus6","icu_strict_any":"icu_strict"}).dropna(subset=["icu_minus6","icu_strict"])
    basic=[];delta=[];increment=[]
    for population,subset in [("all_27005",data),("minus6_with_lab",data[data.icu_minus6.notna()]),
                              ("both_windows_with_lab",data[both]),("both_windows_any_marker",any_panel)]:
        for ep in ["hospital_event","event"]:
            raw,pair,aug=compare(subset,population,ep);basic+=raw;delta+=pair;increment+=aug
    inc=pd.DataFrame(increment)
    primary=inc.population.eq("both_windows_with_lab")
    inc.loc[primary,"q_BH_18_common"]=multipletests(inc.loc[primary,"incremental_auc_oof_p"],method="fdr_bh")[1]
    paths=[]
    for name,frame in [("window_coverage",pd.DataFrame(audits)),("raw_auc",pd.DataFrame(basic)),("paired_auc_difference",pd.DataFrame(delta)),
                       ("oof_increment",inc)]:
        p=OUT/(name+".csv");frame.to_csv(p,index=False);paths.append(p)
    gap=(cohort.cutoff-cohort.landmark).dt.total_seconds()/3600
    write_json(OUT/"verification.json",dict(index_n=len(data),both_windows_have_lab=int(both.sum()),
        median_landmark_difference_hours=float(gap.median()),IQR_and_max_hours=gap.quantile([.25,.75,1]).tolist(),
        original_index_no_later_than_ICU_landmark=bool(gap.ge(0).all()),n_after_6hours=int(gap.gt(6).sum()),
        raw_auc_checked_with_sklearn=True,requires_matching_subject_hadm_labs=True,requires_matching_subject_hadm_stay_vitals=True,
        model_freeze_sha256=sha(TRAIN_RUN/"model_freeze.json"),database_sha256=sha(dbpath),severity_file_sha256=sha(src),
        results_sha256={p.name:sha(p) for p in files+paths},source_dictionary_sha256={k:manifest[k] for k in ["icu/d_items.csv.gz","hosp/d_labitems.csv.gz"]}))
    print(pd.DataFrame(audits).to_string(index=False))
    print(inc[primary][["outcome","clock","base_severity","n","incremental_auc_oof"]].to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(limits=2):run()
