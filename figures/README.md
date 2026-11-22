# Rebuilding the figures and tables

The scripts in this directory implement the manuscript's final figure numbering and layouts. Run them after the relevant cohort analyses. `AC_SRC` points to the populated study project, with `data/`, `runs/`, `external_validation/` and `manuscript/` underneath. The checkout has the code and frozen model. Researchers regenerate the cohort inputs and intermediate results locally.

`manuscript/figures/fig12/make_fig12.py` and `manuscript/figures/fig345/build_figures.py` produce intermediate source tables from those results. The final plotting scripts reuse these tables. The intermediate filenames follow an earlier numbering scheme.

Run the following from the repository root in a shell with Python and the analysis dependencies installed. In this example the current directory contains the populated analysis tree.

```bash
export AC_SRC="$PWD"
export AC_DERIVED="$PWD/outputs/figure_inputs"
export AC_OUT="$PWD/outputs/manuscript"
python manuscript/figures/fig12/make_fig12.py
python manuscript/figures/fig345/build_figures.py
python figures/prep_derived.py "$AC_SRC" "$AC_DERIVED"
python figures/fig_main_v2.py 1 2 3 4 5 6
python figures/fig_ed.py 1 2 3 4 5 6 7 8 9 10
python figures/build_tables.py
```

In PowerShell, set `$env:AC_SRC = (Get-Location).Path`, `$env:AC_DERIVED = Join-Path $env:AC_SRC 'outputs/figure_inputs'` and `$env:AC_OUT = Join-Path $env:AC_SRC 'outputs/manuscript'`. Use `$env:AC_SRC` and `$env:AC_DERIVED` for the two arguments to `prep_derived.py`. To use another verified HRS aggregate export, set `AC_HRS_AGGREGATES` to its directory. A single figure number can be passed to either figure script.

The figure programs write composites, panels and source workbooks. Figure 1c's workbook contains a pair of values per temporal-holdout participant. Keep generated workbooks and intermediate inputs in the local analysis environment. The table program writes Tables 1–2 and one workbook containing the 20 supplementary tables.

For this release, the plotting and table scripts rebuilt all 16 figures and the tables from the study's existing local analysis results. All 19 generated workbooks (91 sheets) matched the accompanying submission workbooks in sheet names, dimensions and cell values, with a numerical tolerance of 1e-11. This comparison checks the plotting and tabulation stage.
