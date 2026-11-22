# Scoring with the frozen FT-Mask model

Run these commands from the repository root. Inference needs Python, NumPy, pandas and PyTorch. The development run used Python 3.10.19, NumPy 2.2.6, pandas 2.3.3 and PyTorch 2.9.1. A CPU installation can score new examinations using the released weights.

```bash
python scripts/verify_frozen.py
python scripts/score_standardized.py --input examples/synthetic_panel.csv --output outputs/synthetic_scores.csv --device cpu
```

The example contains three invented examinations. On the reference CPU run the first two whole-body scores were **48.196** and **58.706**. The second examination contains one kidney measurement. Its kidney score is **61.243** and its other system scores are missing. The third examination contains only age, so all six scores are missing. Floating-point results can vary slightly across PyTorch builds.

## Inputs

Use one row per examination. Supply age at examination in years in the `age` column. Optional `ID` and `SEQN` columns pass through to the output. Input units below are the units before model transformation. Supply observed numeric values and leave missing entries empty, or omit a whole biomarker column. An actual zero is an observed value. The scorer performs the frozen log transforms, standardization, clipping and masking.

| System | Measurements and units |
| --- | --- |
| Liver | `albumin`, `total_protein`, `globulin` (g/dL), `alp`, `ast`, `alt`, `ggt`, `ldh` (U/L), `bilirubin` (mg/dL) |
| Kidney | `creatinine`, `bun`, `uric_acid`, `phosphorus`, `calcium` (mg/dL), `uacr` (mg albumin/g creatinine), `potassium`, `sodium`, `chloride`, `bicarbonate` (mmol/L) |
| Metabolic | `glucose`, `triglycerides`, `cholesterol`, `hdl` (mg/dL), `hba1c` (%), `bmi` (kg/m²), `waist` (cm) |
| Immune and hematologic | `wbc`, `platelets` (10⁹/L), `lymph_pct`, `mono_pct`, `neut_pct`, `eos_pct`, `rdw` (%), `crp` (mg/dL), `rbc` (10¹²/L), `hgb` (g/dL), `mcv`, `mpv` (fL), `mch` (pg), `iron` (µg/dL) |
| Cardiovascular | `sbp`, `dbp` (mmHg), `pulse` (beats/min) |

Enter a percentage such as 28% as `28`. Convert CRP from mg/L to mg/dL by dividing by ten. For the study's NHANES inputs the data preparation also applies cycle-specific assay harmonization. When scoring data from another source, align units and assay methods before passing the measurements to the scorer. The NHANES development cohort included adults aged at least 20 with at least 22 of 43 biomarkers. The scorer accepts sparser rows. The output count records the number used for each panel.

## Outputs

The scoring command produces four columns per panel for `full`, `liver`, `kidney`, `metabolic`, `immune_heme` and `cardiovascular`.

| Suffix | Interpretation |
| --- | --- |
| `_score` | Age acceleration standardized to the panel's NHANES development reference. A score of 50 is the reference mean. Ten points equal one reference SD. |
| `_risk_equivalent_age` | Ensemble log relative risk mapped onto the development cohort's age scale, in years. |
| `_accel_years` | Difference from the risk-equivalent age expected at the person's chronological age, in years. |
| `_n_observed` | Number of biomarker values available to the panel. The count covers biomarkers only. |

The coefficients for the age scale are in `runs/nhanes_gpu_20261121/bundle/clock_bundle.json`. The development means and SDs for the 50/10 scores are in `external_validation/nhanes3_organ/organ_score_reference.json`. Each system score uses that system's measured inputs together with age. A score reflects mortality-risk ranking under the model. The model provides no estimate of an organ's histologic age.

For the raw ensemble log relative risk, use the original frozen scorer. It also outputs the age-scale quantities, coverage and no-observation flags.

```bash
python scripts/score_clock.py --bundle runs/nhanes_gpu_20261121/bundle --input examples/synthetic_panel.csv --output outputs/synthetic_raw.csv --device cpu
```

## Reconstructing the NHANES analyses

The frozen bundle already supports scoring. To reconstruct its development and temporal holdout datasets, use the study environment (`lifelines==0.30.0` in addition to the packages above). The download step validates the size and SHA-256 of 99 public files against the input manifest. It includes the public epigenetic biomarker file for defining the DNAm holdout. [NHANES DNAm data](https://wwwn.cdc.gov/nchs/nhanes/dnam/) · [NCHS public mortality files](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/)

```bash
python scripts/01_download.py
python scripts/02_build_dataset.py
python scripts/rebuild_release_roster.py
python scripts/evaluate_clock.py --run runs/nhanes_gpu_20261121
```

The build writes the two analysis tables into `data/processed/`. The roster step checks their original SHA-256 digests and reconstructs the 19,426-person development roster locally. Evaluation writes participant predictions and the summary for the 25,614-person temporal holdout under the specified run. Its original 300-resample paired interval is an intermediate analysis. Use the manuscript comparison code for the reported 2,000-resample intervals. Generated participant-level files stay in your working copy.

For a fresh training run, first prepare the NHANES tables, then select a new output directory. The training implementation requires CUDA with BF16 support and uses five development cycles and five final initialization seeds. The released run's hardware, package versions, seeds and search grid are recorded in `runs/nhanes_gpu_20261121/protocol.json`.

```bash
python scripts/train_clock_gpu.py --run runs/my_nhanes_refit
python scripts/evaluate_clock.py --run runs/my_nhanes_refit
```
