# FT-Mask clinical aging clock

Research code and frozen model for *A panel-agnostic clinical aging clock transfers across populations and care settings*.

FT-Mask uses chronological age and up to 43 routine clinical measurements to rank all-cause mortality risk. The five-model ensemble produces a whole-body score and five system scores. It was developed in 19,426 NHANES participants. Each score reports how many measurements contributed. A system with no observed measurements receives a missing score.

## Score three synthetic examinations on a CPU

From the repository root, in a Python 3.10 environment, run

```bash
python -m pip install -r requirements.txt
python scripts/verify_frozen.py
python scripts/score_standardized.py --input examples/synthetic_panel.csv --output outputs/synthetic_scores.csv --device cpu
```

The whole-body scores of the first two synthetic examinations round to 48.196 and 58.706. The third contains age alone and receives no score. The standardized scale has a development mean of 50 and a standard deviation of 10. Risk-equivalent age and acceleration in years are also available. See the [scoring guide](docs/SCORING.md) and [43-variable input dictionary](docs/input_dictionary.csv).

## Reproduce a manuscript analysis

Use the broader Python environment for development, evaluation and plotting with `python -m pip install -r requirements-analysis.txt`. Selected analyses also use R and the external resources listed in the study guides. CPU scoring works with the released weights. Training the original implementation requires a CUDA GPU with BF16 support.

| Manuscript result | Where to begin |
| --- | --- |
| Development, temporal holdout and nested cross-validation | [NHANES data preparation, training and evaluation](docs/SCORING.md) |
| Community validation, transport, hospital and ICU endpoints, organ-system comparisons | [Cohorts and analyses](docs/COHORTS.md) |
| Masking ablations, panel-specific models and survival benchmarks | [Model comparisons](docs/BENCHMARKS.md) |
| NHANES and HRS DNA methylation comparisons, disability, function and exposures | [DNAm and healthspan workflows](docs/BENCHMARKS.md) |
| Figures 1–6, Extended Data Figures 1–10, Tables 1–2 and Supplementary Tables 1–20 | [Figure and table reproduction](figures/README.md) |

Study data and individual predictions are obtained or regenerated on the researcher's machine. Dataset access conditions apply separately to each cohort. The synthetic example and released model can be used immediately without cohort access. The hash manifest covers the frozen artifacts used for scoring. [SHA-256 manifest](runs/nhanes_gpu_20261121/SHA256SUMS)

## Release checks

Using the archived NHANES public inputs, the release reproduced all 39 prediction columns for the 25,614 temporal-holdout participants (1,977 deaths, Harrell's C 0.874705). The NHANES III reconstruction yielded 16,288 participants, 7,101 deaths and Harrell's C 0.858411. The release check used the frozen weights. The HRS workflow was exercised with synthetic households. The figure and table check used the study's existing analysis results. [Figure and table checks](figures/README.md)

Figure 6d uses two covariate specifications. The measured cystatin C analysis includes linear and quadratic age, sex and whole-body acceleration. The DNAm HbA1c surrogate analysis includes linear age, sex and the whole-body score. The figure legend groups both analyses under the first specification. [HRS analysis details](docs/BENCHMARKS.md)
