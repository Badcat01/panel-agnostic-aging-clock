"""Evaluate strictly prior-coded comorbidity and the frozen clinical clock on identical readmission patients."""
import numpy as np
import pandas as pd
import torch
import sys
from threadpoolctl import threadpool_limits
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

from cohort import OUT,CACHE,MIM,EXT,write_json,write_csv,digest
sys.path.insert(0,str(EXT/"scripts"))
from reference import BUNDLE, TRAIN_RUN, verify, score_frame
from mimic_iv_panels import compare_auc, ci95, oof_comparison


def run():
    import json
    audit=json.loads((OUT/"cohort_audit.json").read_text())
    panel=CACHE/"history_index_lab_panel.csv.gz"
    assert digest(panel)==audit["panel_sha256"] and digest(CACHE/"prior_codes.csv.gz")==audit["prior_codes_sha256"]
    scores=pd.read_csv(CACHE/"prior_scores.csv.gz")
    data=pd.read_csv(panel).merge(scores,on="subject_id",validate="one_to_one").rename(columns={"subject_id":"ID"})
    assert len(data)==audit["n_with_prior_coded_stay"] and data.ID.is_unique
    bundle=verify()
    torch.backends.mha.set_fastpath_enabled(False);torch.set_float32_matmul_precision("highest")
    preds=score_frame(data.reindex(columns=["ID","age",*bundle["features"]]),BUNDLE,"cuda")
    assert np.array_equal(preds.ID,data.ID)
    predictors=preds[["ID","full_lp","full_accel","full_n_observed"]]
    d=data[["ID","hadm_id","age","sex","hospital_event","event","charlson","elixhauser"]].merge(predictors,on="ID",validate="one_to_one")
    assert d.notna().all().all()
    out=CACHE/"all_scores.csv.gz";write_csv(out,d)
    age=(d.age.to_numpy()-60)/10
    demo=np.column_stack([age,age*age,d.sex])
    comp=[]
    for target in ["hospital_event","event"]:
        y=d[target].to_numpy(dtype=bool)
        title="in_hospital" if target=="hospital_event" else "365d"
        raw=d[["full_lp","charlson","elixhauser"]]
        vals,cov,_=compare_auc(raw.to_numpy(),y)
        for i,col in enumerate(raw.columns):
            assert np.isclose(vals[i],roc_auc_score(y,raw[col]))
            comp.append(dict(outcome=title,model="raw_"+col,n=len(d),events=int(y.sum()),auc=float(vals[i]),
                             auc_ci=ci95(vals[i],cov[i,i],(0,1))))
        for metric in ["charlson","elixhauser"]:
            com=d[metric].to_numpy()
            full=d.full_accel.to_numpy()
            for label,base,addition in [
                ("demographics_add_"+metric,demo,com),("demographics_add_clock",demo,full),
                (metric+"_plus_demographics_add_clock",np.column_stack([demo,com]),full),
                ("clock_plus_demographics_add_"+metric,np.column_stack([demo,full]),com)]:
                if label=="demographics_add_clock" and metric!="charlson":continue
                report=oof_comparison(base,addition,y,20261121)
                comp.append(dict(outcome=title,model=label,n=len(d),events=int(y.sum()),**report))
    tab=pd.DataFrame(comp)
    matched=tab.model.str.contains("_plus_demographics_add_")
    tab.loc[matched,"q_BH_eight"]=multipletests(tab.loc[matched,"incremental_auc_oof_p"],method="fdr_bh")[1]
    tab.to_csv(OUT/"history_mortality_comparison.csv",index=False)
    formal=pd.read_csv(MIM/"formal_v3_1/predictions.csv.gz",usecols=["ID"])
    write_json(OUT/"verification.json",dict(n=len(d),hospital_deaths=int(d.hospital_event.sum()),
               year_deaths=int(d.event.sum()),raw_auc_checked_with_sklearn=True,ICD_package="comorbidity 1.1.0",
               original_person_overlap=int(d.ID.isin(formal.ID).sum()),median_baseline_markers=float(d.full_n_observed.median()),
               clock_sha256=digest(TRAIN_RUN/"model_freeze.json"),paired_score_sha256=digest(out),
               comparison_sha256=digest(OUT/"history_mortality_comparison.csv"),
               comorbidity_scores_sha256=digest(CACHE/"prior_scores.csv.gz")))
    print(tab[["outcome","model","n","events","auc","baseline_auc_oof","augmented_auc_oof"]].to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(limits=2):run()

