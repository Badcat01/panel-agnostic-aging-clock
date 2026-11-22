"""Sensitivity analyses of matched cause-specific associations in NHANES III."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import warnings

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import proportional_hazard_test
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent
sys.path.insert(0, str(EXTERNAL / "scripts"))
from build_nhanes3 import DEATH_FILE, MORT_VARS, load_layout, read_fwf

CAUSES = {
    "cardiovascular": (1, 5),
    "kidney": (9,),
    "metabolic": (7,),
    "immune_heme": (8,),
}
PANELS = ["full", "liver", "kidney", "metabolic", "immune_heme", "cardiovascular"]
CORE = ["age_linear", "age_quadratic", "sex", "full_sd"]


def read_data():
    predictions = EXTERNAL / "results" / "nhanes3" / "predictions.csv.gz"
    checks = json.loads((EXTERNAL / "requested_cohorts" / "verification.json").read_text(encoding="utf-8"))
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != checks["cohorts"]["nhanes3"]["predictions_sha256"]:
        raise ValueError("Frozen prediction digest does not match the prior analysis")
    p = pd.read_csv(predictions).set_index("ID", verify_integrity=True)
    raw_fields = [*MORT_VARS, "UCOD_LEADING"]
    layout = load_layout("SAS_ReadInProgramAllSurveys.sas", raw_fields)
    mort, _ = read_fwf(DEATH_FILE, layout, {46, 47, 48}, 33994, missing_codes=False)
    data = p.loc[p[[f"{g}_ba" for g in PANELS]].notna().all(axis=1)].copy()
    if len(data) != 14807 or not data.index.isin(mort.index).all():
        raise ValueError("Complete panel cohort does not match prior scoring")
    m = mort.loc[data.index]
    np.testing.assert_array_equal(m.MORTSTAT, data.event)
    np.testing.assert_allclose(m.PERMTH_EXM, data.time_yr * 12)
    if not m.ELIGSTAT.eq(1).all():
        raise ValueError("Mortality link ineligible")
    data["cause"] = m.UCOD_LEADING
    # Preserve the same score SD in each analysis. The six missing causes are
    # retained for scale estimation only, with neither event nor cause imputed.
    for panel in PANELS:
        values = data[f"{panel}_accel"]
        data[f"{panel}_sd"] = (values - values.mean()) / values.std()
    a = data.age.to_numpy(dtype=float)
    a = (a - a.mean()) / a.std()
    b = a**2
    b = b - b.mean() - np.dot(a, b) / np.dot(a, a) * a
    data["age_linear"] = a
    data["age_quadratic"] = b / b.std()
    return data


def fit_unweighted(data):
    known = data.loc[data.event.eq(0) | data.cause.notna()].copy()
    early = known.assign(
        cause=known.cause.where(known.time_yr.le(10)), time_yr=known.time_yr.clip(upper=10)
    )
    strata = {
        "same_people_main": known,
        "age_under_80": known.loc[known.age.lt(80)],
        "alive_at_two_year_landmark": known.loc[known.time_yr.gt(2)].assign(time_yr=lambda x: x.time_yr - 2),
        "first_10_years": early,
        "after_10_years": known.loc[known.time_yr.gt(10)].assign(time_yr=lambda x: x.time_yr - 10),
    }
    rows = []
    for analysis, d in strata.items():
        for panel, codes in CAUSES.items():
            col = f"{panel}_sd"
            rhs = [*CORE, col]
            df = d[["time_yr", *rhs]].assign(target=d.cause.isin(codes).astype(int))
            model = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(df, "time_yr", "target", fit_options={"step_size": 0.25})
            ci = model.confidence_intervals_.loc[col]
            row = {
                "analysis": analysis, "panel": panel, "n": len(d), "events": int(df.target.sum()),
                "HR_per_SD": np.exp(model.params_[col]), "CI95_lower": np.exp(ci.iloc[0]),
                "CI95_upper": np.exp(ci.iloc[1]), "p": model.summary.loc[col, "p"],
                "standard_error": model.standard_errors_[col], "design_df": np.nan,
            }
            if analysis == "same_people_main":
                ph = proportional_hazard_test(model, df, time_transform="rank").summary.p
                row["schoenfeld_rank_p_panel"] = ph.loc[col]
                row["schoenfeld_rank_p_full"] = ph.loc["full_sd"]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_survey(data, rscript):
    fields = ["SEQN", "WTPFEX6", "SDPPSU6", "SDPSTRA6"]
    layout = load_layout("exam.sas", fields)
    mec, _ = read_fwf("exam.dat", layout, {6229, 6233}, 31311)
    mec = mec.loc[mec.WTPFEX6.gt(0)]
    if not data.index.isin(mec.index).all() or mec[["SDPPSU6", "SDPSTRA6"]].isna().any().any():
        raise ValueError("Complete panel cohort must be a domain of the MEC design")
    periods = data.loc[data.event.eq(0) | data.cause.notna()].copy()
    periods["in_domain"] = True
    fields = ["in_domain", "time_yr", *CORE]
    for panel, codes in CAUSES.items():
        periods[f"{panel}_death"] = periods.cause.isin(codes).astype(int)
        fields.extend([f"{panel}_death", f"{panel}_sd"])
    mec = mec.join(periods[fields], how="left", validate="one_to_one")
    mec["in_domain"] = mec["in_domain"].eq(True)
    counts = mec.groupby("SDPSTRA6").SDPPSU6.nunique()
    if len(counts) != 49 or not counts.eq(2).all():
        raise ValueError("Full MEC sample does not contain 49 paired pseudo-strata")
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "design.csv"
        result = Path(tmp) / "survey.csv"
        mec.to_csv(source, index=False)
        subprocess.run(
            [str(rscript), "--vanilla", str(HERE / "survey_sensitivity.R"), str(source), str(result)], check=True,
        )
        return pd.read_csv(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", type=Path, help="Add complex survey sensitivity using R's survey package")
    args = parser.parse_args()
    data = read_data()
    result = fit_unweighted(data)
    if args.rscript:
        result = pd.concat([result, fit_survey(data, args.rscript)], ignore_index=True)
    result.to_csv(HERE / "death_sensitivity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
