"""Verify complete coverage, training exclusions, file checksums and independently recomputed exposure estimates."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from score_holdout import HERE,TRAIN_RUN,load_json,write_json,sha


def main():
    roster=load_json(HERE/"roster_audit.json")
    oof=pd.read_csv(HERE/"xwas_oof/development_oof.csv.gz")
    original=pd.read_csv(TRAIN_RUN/"development_ids.csv.gz")
    dn=pd.read_csv(HERE/"dnam_frozen_scores.csv.gz")
    assert len(oof)==19426 and oof.SEQN.is_unique and set(oof.SEQN)==set(original.SEQN)
    assert len(dn)==2525 and dn.SEQN.is_unique and not set(dn.SEQN)&set(oof.SEQN)
    assert sha(HERE/"dnam_frozen_scores.csv.gz")==roster["output_sha256"]
    cv=load_json(TRAIN_RUN/"cross_validation.json")["nested_outer"]
    for f in cv:
        fold=load_json(HERE/f"xwas_oof/audit_{f['year']}.json")
        assert fold["held"] not in fold["fit_cycles"] and fold["n_held"]+fold["n_fit"]==len(original)
        assert fold["prior_nested_settings"]==f and abs(fold["seed0_reconstructed_C"]-f["c"])<1e-6
        assert sha(HERE/f"xwas_oof/held_{f['year']}.csv.gz")==fold["predictions_sha256"]
    ref=pd.read_csv(HERE/"organ_reference_matrix.csv")
    hit=ref[ref.matched & ref.analysis.eq("primary") & ref.q_diagonal_BH.lt(.05)]
    assert list(hit.marker)==["SSCYPC"]
    xs=pd.read_csv(HERE/"xwas/all_stage_results.csv")
    row=xs.beta_accel_years;lo=xs.lower;hi=xs.upper
    near=np.where((lo<=0)&(hi>=0),0,np.where(abs(lo)<abs(hi),lo,hi))
    for effect,name in [(row,"E_value_approx"),(near,"E_value_limit")]:
        # Independently express the same effect as an approximate odds ratio and take its square root.
        rr=np.sqrt(np.exp(1.82*abs(effect)/xs.residual_SD_years))
        want=rr*(1+np.sqrt(1-1/rr))
        np.testing.assert_allclose(want,xs[name],atol=1e-12)
    for _,subset in xs.groupby("stage"):
        np.testing.assert_allclose(multipletests(subset.p,method="fdr_bh")[1],subset.q_BH,atol=1e-12)
    rep=pd.read_csv(HERE/"xwas/replication.csv")
    count=[len(rep),sum(rep.two_stage),sum(rep.stages.eq(3)),sum(rep.three_stage)]
    assert count==[198,74,180,61]
    local_links=[]
    for name in ["NHANES补充实验.md","XWAS分析口径.md"]:
        for link in re.findall(r"\]\(([^)]+)\)",(HERE/name).read_text(encoding="utf8")):
            if "://" not in link:
                assert (HERE/link).is_file(),link
                local_links.append(link)
    manifests={}
    for name in ["manifest","extra_manifest","food_manifest"]:
        records=load_json(HERE/f"sources/{name}.json")
        for record in records:
            assert sha(HERE/"sources"/record["file"])==record["sha256"],record["file"]
        manifests[name]=len(records)
    write_json(HERE/"verification.json",dict(
        n_development_oof=len(oof),n_DNAm_holdout=len(dn),training_overlap=0,
        oof_refit_cycles=len(cv),oof_seed0_archived_concordance_reproduced=True,
        exposure_stage_estimates=len(xs),all_E_values_reconstructed=True,CI_crosses_zero_E_limit_1=int((near==0).sum()),
        E_value_min=float(xs.E_value_approx.min()),E_value_max=float(xs.E_value_approx.max()),all_exposure_FDRs_reconstructed=True,
        exposure_combinations=[int(x) for x in count],source_files_sha256_verified=manifests,local_report_links_verified=len(local_links),
        measured_biomarker_positive_diagonal=hit.marker.tolist(),hashes={s:sha(HERE/s) for s in [
        "dnam_discrimination.csv","organ_reference_matrix.csv","nhanes_cross_sectional_function.csv","nhanes_function_same_people.csv",
        "xwas/all_stage_results.csv","xwas/replication.csv","xwas_oof/development_oof.csv.gz","NHANES补充实验.md"]}))


if __name__=="__main__":main()
