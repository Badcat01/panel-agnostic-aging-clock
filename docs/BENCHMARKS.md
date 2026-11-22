# Panel benchmarks, DNA methylation and functional outcomes

These workflows support Figures 3–4, parts of Figures 2 and 6, and the associated Extended Data analyses. Commands assume a checkout on an analysis machine with the relevant data permissions. Several scripts create participant-level predictions inside their module directory. Run analyses that use controlled data in a private working copy. The scoring example in the main README can run independently of these workflows.

## Panel comparison and model ablation

The simulated primary analysis compares models in the same 25,614 NHANES temporal-holdout participants under the full panel and six restricted panels. The additional BMI-only restriction selects a different analysis set. The real-cohort meta-analysis combines within-cohort differences across NHANES III, HRS, CHARLS and ELSA using Hartung–Knapp intervals.

First generate the NHANES development and holdout files, training protocol, development roster and predictions. Prepare the community input tables and per-feature coverage summaries under `external_validation/inputs/` and `external_validation/results/<cohort>/`. For MIMIC-IV the benchmark reads the local `formal_v3_1/` landmark index, lab panel, vitals and audit. The checked-in `models/panel_definitions.json` contains feature names for the historical fixed panels. Rebuilding them uses the eligible cohorts' measured-feature frequencies. The comparator training code verifies the frozen model digest and development roster.

Use a BF16-capable CUDA device for the neural comparator fits. The study used XGBoost 2.1.4, randomForestSRC 3.10.0 and survey 4.5. Install XGBoost in the active Python environment. The R commands use randomForestSRC, survival, jsonlite, data.table, R.utils, dplyr, survey and BioAge. Install the historical BioAge implementation with `remotes::install_github("dayoonkwon/BioAge@b1f9fc02f086cd4aa74185f2335ab1366082e7fe")`.

From `external_validation/supplement_v2/models/`, run the following in order.

```sh
python design.py
python evaluation_data.py
python train_ablations.py
python train_tree.py
Rscript rsf.R share elsa charls mimic hrs nhanes3
python mice.py
python panel_ridge.py
Rscript bioage.R
python predict.py
python metrics.py --bootstrap
python richness.py
python report_models.py
python audit_results.py --final
```

`comparisons.csv` contains paired estimates within settings. `community_random_effects.csv` contains the across-cohort estimates and `random_panel_richness.csv` the three random panel draws. `subgroup_bootstrap.py` uses the established survival predictions and calibration inputs to estimate discrimination and calibration in the prespecified age and sex strata.

## Clinical and DNA methylation clocks in the same participants

For the NHANES comparison, first generate the formal NHANES inputs and obtain the NCHS DNAm release (`dnmepi.sas7bdat`). Place it in `data/raw/nhanes/`. From `external_validation/supplement_v2/biomarkers/`, run `python score_holdout.py` and then `python compare_dnam.py`. Scoring this historical holdout uses CUDA and checks its overlap against the original development roster. The mortality analysis compares ten DNAm measures in 2,525 eligible participants. It writes discrimination, hazard-ratio and likelihood-ratio summaries.

The HRS 2016 comparison requires the six exact release archives listed in `hrs_preregistered/authorized_2016/researcher_run.py`, plus `hrs2016vbs.sav` and `vbs16aa.dta`. They include the original and supplemental epigenetic clocks, Tracker, weights, the Langa–Weir classification and RAND longitudinal data. Each archive is verified against its recorded SHA-256. Obtain files and passwords through the study's own access process. First generate the HRS community input, frozen predictions and metrics for the cross-check of matching ages. R requires survival, data.table, jsonlite, digest and R.utils. Python readers use pyreadstat and pyzipper.

From `external_validation/supplement_v2/hrs_preregistered/`, supply local file locations and an output directory outside the checkout.

```sh
python authorized_2016/researcher_run.py --downloads "/secure/hrs/archives" --private "/secure/hrs/analysis" --vbs "/secure/hrs/hrs2016vbs.sav" --cytokines "/secure/hrs/vbs16aa.dta"
```

The default follow-up cutoff is 2023-07-31. The HRS model uses household-grouped five-fold evaluation with 2,000 household bootstrap replicates for the primary comparison. HRS participants who were also in the whole-cohort clinical validation must be counted only once when describing the study population. The archived study record describes a prespecified local analysis without an official OSF registration. The historical HRS secondary outputs contained missing uncertainty estimates for chronic-condition count and joint organ regressions. The code contains the corrected standard-error extraction. The corrected intervals still require a fresh authorized analysis.

For Figure 6d, the measured-reference analysis in `references.py` adjusts for linear and quadratic age, sex and the whole-body score. The DNAm surrogate analysis in `authorized_2016/proxy_organ.py` adjusts for linear age, sex and the whole-body score. The manuscript legend describes quadratic age adjustment for both estimates.

### Synthetic HRS check

The following commands exercise the HRS pipeline on generated families without study files. Run them in `hrs_preregistered/`. Twenty refits give a short implementation check. The complete validation in `validate_fixture.py` expects 2,000 refits.

```sh
python make_fixture.py
Rscript run.R synthetic/fixture.csv.gz synthetic/tested synthetic 20
Rscript secondary.R synthetic/fixture.csv.gz synthetic/tested
python references.py synthetic/fixture.csv.gz synthetic/tested
python authorized_2016/test_tracker_fixture.py
python authorized_2016/test_health_fixture.py
python authorized_2016/test_release_fixture.py
python authorized_2016/test_proxy_fixture.py
python authorized_2016/test_aggregate_report.py
```

## Healthspan, organ comparisons and exposures

For HRS, ELSA and CHARLS, point `AGING_CLOCK_COHORTS` to the directory containing the authorized cohort releases. Preserve the original filenames and wave folders. Run the cohort preparation and validation first, then the panel comparator workflow above. The community healthspan model also uses the per-cohort BioAge predictions. In `external_validation/supplement_v2/healthspan/` run `python prepare_healthspan.py`, `Rscript fit_healthspan.R .` and `Rscript check_proportional_hazards.R .`. These analyses use survival, data.table and nlme. Cohort baseline samples, incident events and available interview dates determine the denominator for each endpoint. The registered HRS DNAm healthspan subset has a separate set of same-person inclusion criteria.

For the NHANES exposure analysis in Extended Data Figure 10, use the `biomarkers/` directory. Run `python download_references.py`, `python download_function_exposures.py` and `python healthy_eating.py` to retrieve NCHS components and USDA dietary files. The dietary reader needs bsdtar with 7z support. Run `python rebuild_development_oof.py` to obtain development predictions from held-out survey cycles, then `python exposure_wide.py` to estimate the three-stage associations. The outcome is age acceleration in years. The analysis is observational.
