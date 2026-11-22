"""Link authorized release data and verified frozen HRS scores using a locked JSON field map."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyreadstat
import torch

HERE=Path(__file__).resolve().parent
EXT=HERE.parents[1]
sys.path[:0]=[str(EXT/"hrs_dnam"),str(HERE.parent/"technical")]
from compare_hrs_dnam import read_clocks
from shared import formal
from reference import BUNDLE, TRAIN_RUN, score_frame, sha, verify


def source(spec,base):
    path=(base/spec["file"]).resolve()
    if sha(path)!=spec["sha256"]:
        raise ValueError(f"Source digest mismatch: {path.name}")
    cols=dict(spec["columns"])
    qc=spec.get("qc_pass")
    if qc:
        cols["__qc__"]=qc["field"]
    tab=read_clocks(path,cols).set_index("ID",verify_integrity=True)
    for col,values in spec.get("missing_codes",{}).items():
        if col not in tab:raise ValueError(f"Unknown missingness target {col}")
        tab[col]=tab[col].mask(tab[col].isin(values))
    for col in spec.get("positive_log",[]):
        if not col.startswith("ref_"):raise ValueError("Reference log transform specified for non-assay")
        tab[col]=np.log(tab[col].where(tab[col]>0))
    status=None
    if spec["role"]=="dnam":
        if qc:
            status=tab.pop("__qc__").isin(qc["values"])
        elif spec.get("official_qc_note"):
            # Explicitly identified official release has already removed failed samples.
            status=pd.Series(True,index=tab.index)
        else:
            raise ValueError("Document official DNAm QC status or supplied pass values")
        for col in tab:
            if col.startswith("clock_"):
                tab[col]=tab[col].where(status)
        for col in spec.get("reverse_direction",[]):
            tab[col]=-tab[col]
    return tab,status


def baseline(raw_vbs,age_sex):
    fieldmap={"albumin":"PALB","alp":"PALKP2","ast":"PAST","alt":"PALT","bilirubin":"PBILT",
        "total_protein":"PTP","creatinine":"PCR","bun":"PBUN","potassium":"PK","sodium":"PNA","chloride":"PCL",
        "bicarbonate":"PCO2","calcium":"PCA","glucose":"PGLUFF","triglycerides":"PTGF","cholesterol":"PCHOL",
        "hdl":"PHDLD","wbc":"PWBC","lymph_pct":"PLYMP","mono_pct":"PMONO","neut_pct":"PNEUT","eos_pct":"PEOS",
        "crp":"PCRP","rbc":"PRBC","hgb":"PHGB","mcv":"PMCV","mch":"PMCH","rdw":"PRDW","platelets":"PPLT","mpv":"PMPV"}
    v,_=pyreadstat.read_sav(raw_vbs,usecols=["HHID","PN",*fieldmap.values()])
    ids=v.HHID.str.strip()+"-"+v.PN.str.strip()
    raw=v.rename(columns={v:k for k,v in fieldmap.items()})[list(fieldmap)]
    raw.index=pd.Index(ids,name="ID")
    if not raw.index.is_unique:raise ValueError("Duplicate 2016 VBS participant")
    raw=raw.where(raw.ge(0))
    raw["crp"]/=10
    raw["globulin"]=(raw.total_protein-raw.albumin).where(raw.total_protein>raw.albumin)
    return raw.join(age_sex,validate="one_to_one").dropna(subset=["age","sex"]).loc[lambda x:
        x.drop(columns=["age","sex"]).notna().any(axis=1)&x.age.between(20,120)]


def run(mapping,output):
    spec=json.loads(mapping.read_text(encoding="utf8"))
    if not all(spec.get(k) for k in ["tracker_version","tracker_cutoff","sources","raw_tracker_file","raw_tracker_sha256",
                                     "raw_vbs_file","raw_vbs_sha256","tracker_derivation"]):
        raise ValueError("Provide tracker version, cutoff, verified raw file, and derivation of standardized time bounds")
    raw_tracker=(mapping.parent/spec["raw_tracker_file"]).resolve()
    raw_vbs=(mapping.parent/spec["raw_vbs_file"]).resolve()
    if sha(raw_tracker)!=spec["raw_tracker_sha256"] or sha(raw_vbs)!=spec["raw_vbs_sha256"]:raise ValueError("Tracker/VBS SHA mismatch")
    if not {"dnam","tracker"}.issubset({s["role"] for s in spec["sources"]}):
        raise ValueError("Provide official DNAm source and harmonized latest tracker")
    bundle=verify()
    predfile=EXT/"results/hrs/predictions.csv.gz"
    archived=json.loads((EXT/"results/completion_check.json").read_text(encoding="utf8"))
    if sha(predfile)!=archived["cohort_checks"]["hrs"]["predictions_sha256"]:raise ValueError("Frozen HRS scores changed")
    ts=[s for s in spec["sources"] if s["role"]=="tracker"]
    if len(ts)!=1 or not {"age","sex"}.issubset(ts[0]["columns"]):raise ValueError("One tracker with baseline age and sex is required")
    tracker,_=source(ts[0],mapping.parent)
    raw=baseline(raw_vbs,tracker[["age","sex"]])
    torch.backends.mha.set_fastpath_enabled(False);torch.set_float32_matmul_precision("highest")
    device="cuda" if torch.cuda.is_available() else "cpu"
    prepared=raw.reindex(columns=["age",*bundle["features"]]).reset_index()
    pred=score_frame(prepared,BUNDLE,device).set_index("ID")
    # Check original laboratory mapping and re-inference on every common participant with the same baseline age.
    old_raw,old_pred=formal("hrs")
    common=old_pred.index.intersection(pred.index)
    same=common[old_pred.loc[common,"age"].eq(pred.loc[common,"age"])]
    if not len(same):raise ValueError("No same-age previous clinical reference participants")
    shared_features=[k for k in bundle["features"] if k in raw and k in old_raw]
    np.testing.assert_allclose(raw.loc[same,shared_features],old_raw.loc[same,shared_features],equal_nan=True)
    np.testing.assert_allclose(pred.loc[same,"full_accel"],old_pred.loc[same,"full_accel"],rtol=1e-5,atol=2e-4)
    d=pred[["age","full_accel"]].rename(columns={"full_accel":"clinical"}).join(raw[["sex"]])
    for col in [x for x in pred if x.endswith("_accel") and x!="full_accel"]:
        d["organ_"+col.removesuffix("_accel")]=pred[col]
    for cap in [80,85]:
        values=d.clinical.copy()
        ii=raw.age>cap
        if ii.any():
            sub=raw.loc[ii].assign(age=cap).reset_index()
            recalculated=score_frame(sub,BUNDLE,device).set_index("ID")
            values.loc[ii]=recalculated.full_accel.reindex(values.loc[ii].index)
        d["clinical_cap"+str(cap)]=values
    qcs=[];counts=[];mapped=set()
    for s in spec["sources"]:
        tab,status=source(s,mapping.parent)
        if s["role"]=="tracker":tab=tab.drop(columns=["age","sex"])
        if set(tab)&set(d):raise ValueError(f"Repeated or protected target in {s['file']}")
        if status is not None:qcs.append(status)
        if s["role"]=="tracker" and not {"time_yr","event","time_lower_yr","time_upper_yr"}.issubset(tab):
            raise ValueError("Tracker must supply midpoint and both interval limits from the 2016 draw")
        counts.append({"role":s["role"],"name":Path(s["file"]).name,"sha256":s["sha256"],
                       "n":len(tab),"clinical_linked":len(tab.index.intersection(d.index))})
        mapped.update(tab.columns)
        d=d.join(tab,how="left",validate="one_to_one")
    if not any(x.startswith("clock_") for x in mapped):raise ValueError("No mapped clinical-DNAm pair")
    d["qc_pass"]=pd.concat(qcs,axis=1).eq(True).any(axis=1).reindex(d.index,fill_value=False).astype(int)
    if not d.qc_pass.sum():raise ValueError("No eligible DNAm ID matched clinical scores")
    ev=d.event.dropna()
    if not ev.isin([0,1]).all():raise ValueError("Unexpected mortality coding")
    known=d[["time_lower_yr","time_yr","time_upper_yr"]].dropna()
    if not ((known.time_lower_yr<=known.time_yr)&(known.time_yr<=known.time_upper_yr)).all():
        raise ValueError("Tracker bounds inconsistent")
    output.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(output,compression={"method":"gzip","mtime":0})
    audit=dict(type="HRS-authorized-paired",model_freeze_sha256=sha(TRAIN_RUN/"model_freeze.json"),
               prepared_sha256=sha(output),source=counts,n_clinical=len(d),n_DNAm_QC_matched=int(d.qc_pass.sum()),
               tracker_version=spec["tracker_version"],tracker_cutoff=spec["tracker_cutoff"],
               raw_tracker_sha256=spec["raw_tracker_sha256"],raw_vbs_sha256=spec["raw_vbs_sha256"],
               previous_clinical_scores_verified=len(same),mapping_sha256=sha(mapping),clock_mapping=list(sorted(mapped)))
    Path(str(output)+".json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mapping",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.mapping,a.output)
