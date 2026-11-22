# Reproducing the cohort analyses

Run these commands from the repository root in an environment with the analysis dependencies. Each builder harmonizes measurements and derives one baseline per participant under the rules of the corresponding study. The builders write individual records into local work directories. Use an access-controlled working copy for analyses of restricted releases.

## NHANES III mortality

Place the CDC files below in `external_validation/nhanes3_source/`. The SAS programs specify fixed-width field positions. The three PDF codebooks are included in the input audit. Preserve the filenames, including the uppercase letters in the two mortality filenames.

| Release | Files used |
| --- | --- |
| [NHANES III examination, adult and laboratory files](https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx) | `adult.dat`, `exam.dat`, `lab.dat`, their corresponding `.sas` programs, and `adult-acc.pdf`, `exam-acc.pdf`, `lab-acc.pdf` |
| [2019 public-use linked mortality](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/) | `NHANES_III_MORT_2019_PUBLIC.dat`, `SAS_ReadInProgramAllSurveys.sas` |

```sh
python external_validation/scripts/build_nhanes3.py
python external_validation/scripts/validate_external.py nhanes3 --device cpu
python external_validation/scripts/report_requested_cohorts.py --cohorts nhanes3
python external_validation/scripts/nhanes3_phenoage.py
```

The first command produces `external_validation/inputs/nhanes3.csv.gz` and an audit in `external_validation/audit/`. The second uses the released ensemble, comparator coefficients and NHANES baseline risks. The third checks identity alignment, counts and file hashes, and records the audit used by the organ analyses. It writes a new summary for the requested set of cohorts. The expected NHANES III evaluation has 16,288 adults and 7,101 deaths. The earlier community protocol used 500 bootstrap draws. The manuscript benchmark comparisons use the paired analyses described in [BENCHMARKS.md](BENCHMARKS.md).

For NHANES III system-specific validation, run `organ_score.py export-nhanes3`, then `mortality.py`, `kidney_cystatin.py`, `diabetes_without_known.py` and `sensitivity.py` in `external_validation/nhanes3_organ/` using Python. The cystatin C script accepts `--xpt` for a local CDC file and can also fetch the official release. For the liver ultrasound endpoint, supply the CDC XPT as `external_validation/nhanes3_organ/HGUHS.xpt`, then run `liver_ultrasound.py`. Pass `--rscript Rscript` to `sensitivity.py` for the complex-survey analysis. Install the R packages `survey` and `survival` for that step. [CDC cystatin C](https://wwwn.cdc.gov/Nchs/data/nhanes3/27a/sscystat.xpt) · [CDC ultrasound](https://wwwn.cdc.gov/nchs/data/nhanes3/34a/HGUHS.htm)

## Community cohorts

Obtain the releases from the applicable study under its access terms. Set `AGING_CLOCK_COHORTS` to their common parent directory. Its default is `data/cohorts/` under this repository. For instance, on PowerShell use `$env:AGING_CLOCK_COHORTS = 'E:\study-data\cohorts'`. Keep the original variable names and filenames. The builders check one-to-one joins, follow-up dates and available measurements.

| Cohort | Local release layout and primary inputs |
| --- | --- |
| HRS | Under `HRS/`, stage the 2016 venous blood release (`hrs2016vbs.sav`) and 2020 Tracker (`trk2020tr_r.dta`). The near-visit and longitudinal analyses also use wave-specific Core files, RAND and harmonized releases. Sensitive DNAm releases require separate access. [HRS data products](https://hrsdata.isr.umich.edu/data-products) |
| ELSA | Place `wave_2_nurse_data_v2.dta` in `ELSA/Raw_data/wave2/`. Provide `h_elsa_g3.dta` and `index_file_wave_0-wave_5_v2.dta` under `ELSA/`. Later waves supply additional function and repeat-marker measurements. [ELSA data](https://www.elsa-project.ac.uk/accessing-elsa-data) |
| CHARLS | Under `CHARLS/Raw_data/`, keep `Harmonized_CHARLS_D/H_CHARLS_D_Data.dta`, `2011charls/Blood_20140429.dta`, `2020charls/Exit_Module.dta`, and the interview, biomarker and tracking files for the waves used. The builder lists each module and field. [CHARLS](https://charls.pku.edu.cn/en/) |
| SHARE, MHAS, KLoSA and LASI-DAD | Place each study's original releases under the common parent. `build_other.py` identifies files recursively by exact study filename. Its four builder functions specify the required wave, variables and tracking records. LASI-DAD supplies score coverage only. |

```sh
python external_validation/scripts/build_hrs_elsa.py
python external_validation/scripts/build_charls.py
python external_validation/scripts/build_other.py share mhas klosa lasi
python external_validation/scripts/validate_external.py hrs elsa charls share mhas klosa lasi --device cpu
python external_validation/scripts/validate_external.py charls_short --sensitivity --device cpu
python external_validation/scripts/report_requested_cohorts.py --cohorts nhanes3 charls hrs elsa
python external_validation/scripts/interval_sensitivity.py
python external_validation/scripts/describe_panels.py
python external_validation/scripts/check_measurement_shift.py
python external_validation/scripts/summarize.py
```

`build_hrs_elsa.py` consumes both study releases. Use the cohort names available to you for validation and reporting. The pooled summary and later sensitivity scripts require all of their component studies. The frozen reference uses the byte-identical original `external_validation/验证方案.md` and model freeze record.

## MIMIC-IV v3.1

Access the v3.1 archive through PhysioNet with the required credentialing, training and data use agreement. Set `AGING_CLOCK_MIMIC_ARCHIVE` to the archive's local path. The default is `data/mimic-iv-3.1.zip`. Keep the original archive prefix and internal checksum manifest. [PhysioNet MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)

The following example uses PowerShell. It builds the first-admission 24-hour landmark, streams the laboratory and vital-sign tables, applies the frozen model and compares predictions for in-hospital and 365-day mortality.

```powershell
$env:AGING_CLOCK_MIMIC_ARCHIVE = 'E:\study-data\mimic-iv-3.1.zip'
$mimic = 'external_validation/mimic_iv'
$analysis = "$mimic/formal_v3_1"
python "$mimic/mimic_iv_run.py" extract
python "$mimic/mimic_iv_run.py" index
python "$mimic/mimic_iv_run.py" labs
python "$mimic/mimic_iv_vitals.py" --archive $env:AGING_CLOCK_MIMIC_ARCHIVE --index "$analysis/landmark_index.csv.gz" --out $analysis
python "$mimic/mimic_iv_run.py" score
python "$mimic/mimic_iv_panels.py" --predictions "$analysis/predictions.csv.gz" --out $analysis
```

The target denominator is 130,242 adults, with 4,116 in-hospital and 17,469 one-year deaths. Individual panels, landmarks and predictions produced by these commands are restricted derived data.

For the ICU comparisons, install Python `duckdb==1.5.6` and `sqlglot==30.33.0`. Obtain MIT-LCP's source in the following layout and check out the study revision. The severity scripts read the official ICU concepts and compute local time-limited inputs. [MIT-LCP concepts at the study revision](https://github.com/MIT-LCP/mimic-code/tree/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iv/concepts)

```sh
git clone https://github.com/MIT-LCP/mimic-code external_validation/supplement_v2/resources/mimic-code
git -C external_validation/supplement_v2/resources/mimic-code checkout 303d26c623dcc9c49cc0f204468d4acc2f063797
python external_validation/supplement_v2/technical/icu_comparators.py
python external_validation/supplement_v2/technical/icu_aligned/run.py
python external_validation/supplement_v2/technical/icu_aligned/verify_aligned.py
```

For the distinct prior-admission cohort, change into `external_validation/supplement_v2/technical/prior_comorbidity/` and run `python cohort.py`, `Rscript prior_scores.R`, `python compare.py`, `Rscript verify_scores.R`, then `python subgroups.py`. The R stages require `comorbidity` version 1.1.0, `data.table` and `jsonlite`. Return to the repository root and run `python external_validation/supplement_v2/local_completion/clinical_tech/sapsii_prior_icu.py`. The prior-history ICU comparison uses 9,213 participants.

For landmarked incident endpoints (Supplementary Table 15), run the Python scripts below from `external_validation/mimic_iv/incident_v3_1/scripts/` after the hospital scores have been generated. Each stage uses the same formal cohort and keeps its intermediate records locally.

1. `collect_serial_labs.py` prepares the landmark roster and scans serial measurements.
2. `build_liver_metabolic.py`, `build_kidney.py`, `build_infection.py` and `build_cardio.py` construct the organ-specific risk sets and observed events.
3. `evaluate_incident.py` compares the scores. Run `kidney_sensitivity.py` and `verify_incident.py` for the additional renal analysis and independent checks. The `diagnose_*.py` scripts examine monitoring, overlap and sensitivity. The [analysis plan](../external_validation/mimic_iv/incident_v3_1/分析方案.md) specifies how observation ends and defines each outcome.

## Calibration, organ outcomes and longitudinal analyses

* **Transport and calibration.** After preparing the development, temporal and external cohorts, run `supplement_v2/technical/calibration.py`; `survival_standard.py temporal nhanes3 charls hrs elsa share klosa mimic`; then `horizon_audit.py`. Paths in this section are relative to `external_validation/`. `interpretability.py` computes marker gradients and system contributions. `survey_and_fairness.py` and `supplement_v2/local_completion/charls_survey/run.py` use the original survey design files. `supplement_v2/multicohort/loco_survival.py`, `transport_risk.py` and `verify_loco.py`, in that order, separately fit and verify the leave-one-cohort-out comparison.
* **Organ-system comparisons.** `supplement_v2/technical/hrs_references.py` computes the matched HRS reference analyses. `mimic_readmit.py` in the same directory generates the readmission outcomes. `supplement_v2/local_completion/organ_local/run.py` uses the NHANES III cause-of-death, ultrasound, MIMIC readmission, in-hospital incident and biomarker-reference intermediates to compare the five scores on the same outcomes.
* **Community onset and function.** After building the three community cohorts and the `supplement_v2/healthspan/` baseline/visit files, run `supplement_v2/local_completion/organ_community/run_community.py` and `charls_lab_thresholds.py` in that directory for incident diagnoses and measured threshold changes. `supplement_v2/local_completion/health_long/run_health_followup.py` adds proxy-inclusive and functional follow-up. Its `fit_health_extensions.R` and the organ analyses' `fit_community.R` require `data.table` and `survival`; both Python launchers accept `--rscript`.
