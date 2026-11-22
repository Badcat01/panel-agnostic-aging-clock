"""Reconcile the predictions and prespecified decisions against the frozen run."""
import argparse
import numpy as np
import pandas as pd
from design import EXT, FEATS, OUT, RUN, original, panels, read


def verify(require_complete):
    sources={}
    panel_audit={}
    core=["nhanes3","hrs","mimic","charls","elsa","share"]
    cohort=pd.read_csv(RUN/"temporal_predictions.csv.gz",dtype={"SEQN":str}).set_index("SEQN")
    for label, source, idcol in [
        ("primary_full",RUN/"temporal_predictions.csv.gz","SEQN"),
        ("actual_mimic",EXT/"mimic_iv/formal_v3_1/predictions.csv.gz","ID"),
        ("actual_nhanes3",EXT/"results/nhanes3/predictions.csv.gz","ID"),
        *[("external_"+n,EXT/"results"/n/"predictions.csv.gz","ID") for n in ["hrs","elsa","charls","share","klosa"]]
    ]:
        fp=OUT/"predictions"/(label+".csv.gz")
        if not fp.exists() and not require_complete:
            continue
        d=pd.read_csv(fp,dtype={"ID":str}).set_index("ID")
        ref=pd.read_csv(source,dtype={idcol:str}).set_index(idcol).loc[d.index]
        distance=float(np.abs(d.M0-ref.full_lp).max())
        assert distance<3e-5
        np.testing.assert_array_equal(d.event,ref.event)
        np.testing.assert_allclose(d.time_yr,ref.time_yr)
        sources[label]={"n":len(d),"max_lp_difference_from_prior":distance}
    metrics=pd.read_csv(OUT/"comparisons.csv")
    judgments=[]
    for panel in core:
        p=OUT/"predictions"/f"primary_{panel}.csv.gz"
        if not p.exists() and not require_complete:
            continue
        df=pd.read_csv(p,dtype={"ID":str}).set_index("ID")
        assert sorted(df.index)==sorted(cohort.index) and df.event.sum()==1977 and len(df)==25614
        subset=metrics[(metrics.setting=="primary")&(metrics.panel==panel)]
        for model in ["M1_nomask","M2_panel"]:
            row=subset[subset.model==model].dropna(subset=["ci_low","ci_high"])
            if not len(row):continue
            row=row.iloc[0]
            out={"panel":panel,"model":model,"delta":row.delta_M0_minus_comparator,
                              "ci_low":row.ci_low,"ci_high":row.ci_high,"M0_superior":row.ci_low>0,
                              "M0_noninferior_one_sided_975":row.ci_low>-.003,
                              "both_95_bounds_inside_plus_minus_003":row.ci_low>-.003 and row.ci_high<.003}
            if model=="M2_panel" and np.isfinite(row.get("ci90_low",np.nan)):
                out.update(lower_one_sided_95=row.ci90_low,upper_one_sided_95=row.ci90_high,
                           M0_noninferior_one_sided_95=row.ci90_low>-.003,
                           equivalence_tost_alpha05=row.ci90_low>-.003 and row.ci90_high<.003)
            judgments.append(out)
        if require_complete:
            mandatory={"M0","M1_nomask","M2_panel","M3_XGB_aug","M3_XGB_raw","M4_RSF","M5_MLP","M6_MICE"}
            assert mandatory.issubset(df.columns) and np.isfinite(df[list(mandatory)]).all().all()
            assert mandatory-{"M0"} <= set(subset.loc[subset.bootstrap.eq(2000),"model"])
            fit=OUT/"fits"/("M2_"+panel)
            manifest=read(fit/"freeze.json")["sha256"]
            assert len(list((fit/"cv").glob("*.json")))==30
            assert {f"seed{k}.pt" for k in original.SEEDS}<=manifest.keys()
            assert all(original.sha(fit/k)==v for k,v in manifest.items())
            assert (OUT/"fits"/("M4_rsf_"+panel)/"final.rds").is_file()
            ext=metrics.loc[(metrics.setting=="external")&(metrics.panel==panel)]
            assert mandatory-{"M0"} <= set(ext.loc[ext.bootstrap.eq(2000),"model"])
            target=ext.loc[ext.model.eq("M2_panel")].iloc[0]
            assert np.isfinite([target.ci_low,target.ci_high,target.ci90_low,target.ci90_high]).all()
            panel_audit[panel]={"valid_cv_cells":30,"frozen_ensemble_seeds":len(original.SEEDS),
                                "primary_n":len(df),"external_n":int(target.n),"external_events":int(target.deaths)}
    if judgments:
        pd.DataFrame(judgments).to_csv(OUT/"prespecified_decisions.csv",index=False)
    availability=pd.read_csv(OUT/"scoring_coverage.csv")
    hashes={name:original.sha(OUT/name) for name in ("comparisons.csv","scoring_coverage.csv",
                  "temporal_paired_heatmap.png","external_paired_heatmap.png","community_random_effects.csv","panel_definitions.json")}
    original.save_json(OUT/"result_audit.json",{"final_all_models":require_complete,"prior_matches":sources,
                       "paired_panel_completion":panel_audit,"table_hashes":hashes,"availability_scenarios":int(availability.scenario.nunique()),
                       "primary_n":len(cohort),"primary_events":int(cohort.event.sum())})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--final",action="store_true")
    verify(p.parse_args().final)
