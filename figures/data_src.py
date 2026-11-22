"""Central loader: every number plotted comes from an archived CSV/JSON (or the
derived descriptive tables built by prep_derived.py)."""
import json, os
import pandas as pd

S = os.environ["AC_SRC"]          # .../src/aging_clock
D = os.environ["AC_DERIVED"]      # derived dir
EV = f"{S}/external_validation"
SV = f"{EV}/supplement_v2"
HRS = os.environ.get("AC_HRS_AGGREGATES", f"{SV}/hrs_preregistered/authorized_2016/汇总结果/20261122_154642_897962")
FIG12 = f"{S}/manuscript/figures/fig12"
FIG345 = f"{S}/manuscript/figures/fig345"


def csv(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def js(path):
    return json.load(open(path, encoding="utf-8"))


def fig1b():
    return csv(f"{FIG12}/Fig1b_source.csv")


def fig1c():
    return csv(f"{FIG12}/Fig1c_source.csv")


def panels_sim():
    return csv(f"{FIG12}/Fig2a_source.csv")


def rich():
    return csv(f"{FIG12}/Fig2c_source.csv")


def random_effects():
    return csv(f"{SV}/models/community_random_effects.csv")


def cohorts():
    return csv(f"{EV}/requested_cohorts/cohorts.csv")


def cohort_summary():
    return csv(f"{EV}/results/cohort_summary.csv")


def horizons():
    return csv(f"{EV}/requested_cohorts/horizons.csv")


def subgroups():
    return csv(f"{SV}/models/subgroup_bootstrap.csv")


def loco():
    return csv(f"{SV}/multicohort/eligible_sources_only/leave_one_cohort_out.csv")


def risk_transport():
    return csv(f"{SV}/multicohort/eligible_sources_only/absolute_risk_transport.csv")


def temporal_metrics():
    return js(f"{S}/runs/nhanes_gpu_20261121/temporal_metrics.json")


def phenoage_n3():
    return js(f"{EV}/results/nhanes3/phenoage_comparison.json")


def dnam_nhanes():
    return csv(f"{FIG345}/Fig3_source.csv")


def dnam_hr():
    return csv(f"{SV}/biomarkers/dnam_hr.csv")


def dnam_lrt():
    return csv(f"{SV}/biomarkers/dnam_lrt.csv")


def hrs_disc():
    return csv(f"{HRS}/discrimination.csv")


def hrs_hr():
    return csv(f"{HRS}/HR.csv")


def hrs_lrt():
    return csv(f"{HRS}/LRT.csv")


def hrs_health():
    return csv(f"{HRS}/healthspan.csv")


def hrs_vbs_ref():
    return csv(f"{HRS}/VBS_reference_matrix.csv")


def hrs_proxy():
    return csv(f"{HRS}/dnam_proxy_organ.csv")


def mimic_main():
    return csv(f"{FIG345}/Fig4_source.csv")


def mimic_panels():
    return csv(f"{EV}/mimic_iv/formal_v3_1/panel_metrics.csv")


def icu_raw():
    return csv(f"{SV}/technical/icu_aligned/raw_auc.csv")


def icu_inc():
    return csv(f"{SV}/technical/icu_aligned/oof_increment.csv")


def organ_effects():
    return csv(f"{FIG345}/Fig5_effects_source.csv")


def organ_matrix():
    return csv(f"{FIG345}/Fig5_matrix_source.csv")


def cause_matrix():
    return csv(f"{EV}/nhanes3_organ/cause_score_matrix.csv")


def n3_validation():
    return csv(f"{EV}/nhanes3_organ/validation_summary.csv")


def disease_diag():
    return csv(f"{SV}/healthspan/disease_diagonal.csv")


def healthspan_s1():
    return csv(f"{FIG345}/FigS1_source.csv")


def xwas():
    return csv(f"{SV}/biomarkers/xwas/replication.csv")


def cv():
    return js(f"{S}/runs/nhanes_gpu_20261121/cross_validation.json")


def interval_sens():
    return csv(f"{EV}/results/interval_sensitivity.csv")


def assay_mask():
    return js(f"{EV}/results/nhanes3/assay_mask_sensitivity.json")


def charls_weights():
    return csv(f"{SV}/local_completion/charls_survey/charls_cluster_sensitivity.csv")


def ig():
    return csv(f"{SV}/technical/marker_integrated_gradients.csv")


def shapley():
    return csv(f"{SV}/technical/organ_shapley_by_age.csv")


def mimic_incident():
    return csv(f"{EV}/mimic_iv/incident_v3_1/results/incident_metrics.csv")


def derived(name, **kw):
    return pd.read_csv(f"{D}/{name}", **kw)
