"""Compare linked HRS 2016 DNA-methylation clocks with frozen clinical scores."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from reference import ROOT, TRAIN_RUN, sha, save_json, verify
from validate_external import discrimination

# Higher age, mortality risk, and pace are used as fixed risk directions.
# The release's methylation-level proxies are examined as associations.
BASE_CLOCKS = {
    "Horvath": ("HORVATH_DNAMAGE", True),
    "Hannum": ("HANNUM_DNAMAGE", True),
    "DNAmPhenoAge": ("LEVINE_DNAMAGE", True),
    "SkinBlood": ("HORVATHSKIN_DNAMAGE", True),
    "Lin": ("LIN_DNAMAGE", True),
    "Weidner": ("WEIDNER_DNAMAGE", True),
    "VidalBralo": ("VIDALBRALO_DNAMAGE", True),
    "Yang": ("YANG_DNAMAGE", False),
    "Zhang_2017": ("ZHANG_DNAMAGE", True),
    "Bocklandt": ("BOCKLANDT_DNAMAGE", False),
    "Garagnani": ("GARAGNANI_DNAMAGE", False),
    "GrimAge": ("DNAMGRIMAGE", True),
    "DunedinPoAm": ("MPOA", True),
}


def numeric_id(values, width, field):
    s = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    if not s.str.fullmatch(rf"[0-9]{{1,{width}}}").fillna(False).all():
        raise ValueError(f"Missing or malformed {field}; expected up to {width} digits")
    return s.str.zfill(width)


def read_clocks(path, columns):
    suf = path.suffix.lower()
    if suf == ".dta":
        table = pd.read_stata(path, convert_categoricals=False)
    elif suf == ".sav":
        table = pd.read_spss(path, convert_categoricals=False)
    elif suf == ".sas7bdat":
        table = pd.read_sas(path, encoding="utf-8")
    elif suf == ".csv" or path.name.lower().endswith(".csv.gz"):
        table = pd.read_csv(path, dtype="string")
    else:
        raise ValueError("Use a .dta, .sav, .sas7bdat, .csv or .csv.gz file")
    table.columns = table.columns.str.strip().str.upper()
    if not table.columns.is_unique:
        raise ValueError("Duplicate column names after case normalization")
    hhid = numeric_id(table["HHID"], 6, "HHID")
    pn = numeric_id(table["PN"], 3, "PN")
    ids = hhid + "-" + pn
    if ids.duplicated().any():
        raise ValueError("HRS methylation file contains duplicate HHID-PN")
    result = pd.DataFrame({"ID": ids})
    for label, source_column in columns.items():
        values = pd.to_numeric(table[source_column.upper()], errors="raise")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Infinite values in {source_column}")
        result[label] = values.to_numpy(dtype=float, na_value=np.nan)
    return result


def fit_association(data):
    columns = ["age", "sex", "time_yr", "event"]
    fits = {}
    design = data[columns + ["household"]].copy()
    fields = {"HRS_DNAm_sd": "dnam", "frozen_clinical_accel_sd": "full_accel"}
    sd = {}
    for label, source in fields.items():
        sd[label] = float(data[source].std())
        if sd[label] <= 0:
            return {"status": "constant_score"}
        design[label] = (data[source] - data[source].mean()) / sd[label]
    for key, term_list in (("clinical", ["frozen_clinical_accel_sd"]),
                           ("dnam", ["HRS_DNAm_sd"]), ("joint", list(fields))):
        model = CoxPHFitter().fit(
            design[columns + term_list + ["household"]],
            duration_col="time_yr", event_col="event", cluster_col="household"
        )
        fits[key] = model
    effects = {}
    for key, model in fits.items():
        effects[key] = {
            label: {
                "sd": sd[label],
                "hr": float(np.exp(model.params_[label])),
                "CI95": np.exp(model.confidence_intervals_.loc[label]).tolist()
            }
            for label in fields if label in model.params_
        }
    tests = {}
    for key in ("clinical", "dnam"):
        delta = 2 * (fits["joint"].log_likelihood_ - fits[key].log_likelihood_)
        tests["joint_vs_" + key] = {"LR": float(delta), "p_independent_people": float(chi2.sf(delta, 1))}
    return {"age_sex_adjusted_HR_per_paired_sd": effects, "exploratory_LR": tests}


def compare(performance_data, methylation, name, use_ranking):
    d = performance_data.merge(methylation[["ID", name]], on="ID", how="inner", validate="one_to_one")
    linked = len(d)
    d = d.dropna(subset=["sex", "time_yr", "event", "full_lp", "full_accel", name]).copy()
    d = d.loc[d.time_yr.gt(0)].rename(columns={name: "dnam"})
    d["household"] = d.ID.str.split("-").str[0]
    out = dict(linked=linked, n=len(d), deaths=int(d.event.sum()),
               lost_after_link=linked-len(d), n_households=int(d.household.nunique()))
    if d.event.sum() < 20 or d.sex.nunique() < 2:
        out["status"] = "insufficient_outcomes_or_covariate_variation"
        return out
    out.update(fit_association(d))
    if use_ranking:
        out["raw_score_pairwise_discrimination"] = discrimination(
            d, ["full_lp", "dnam", "age_sex_lp", "ridge_lp"]
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, help="Authorized HRS Epigenetic Clocks v1.0")
    parser.add_argument("--supplement", type=Path, help="Authorized HRS Epigenetic Clocks Supplement v1.0")
    parser.add_argument("--grim2-column", default="GRIM2")
    parser.add_argument("--pace-column", default="PACE")
    parser.add_argument("--output", type=Path, required=True, help="Aggregate JSON output path")
    args = parser.parse_args()
    if args.original is None and args.supplement is None:
        parser.error("Specify at least one authorized HRS data file")
    paths_for_comparison = []
    if args.original is not None:
        paths_for_comparison.append((args.original, BASE_CLOCKS))
    if args.supplement is not None:
        paths_for_comparison.append((args.supplement, {
            "GrimAge2": (args.grim2_column, True), "DunedinPACE": (args.pace_column, True)
        }))

    verify()
    predictions = ROOT / "results" / "hrs" / "predictions.csv.gz"
    completion = json.loads((ROOT / "results" / "completion_check.json").read_text(encoding="utf-8"))
    pred_hash = sha(predictions)
    if pred_hash != completion["cohort_checks"]["hrs"]["predictions_sha256"]:
        raise ValueError("HRS prediction file differs from the verified frozen run")
    hrs = pd.read_csv(predictions, dtype={"ID": "string"})
    if hrs.ID.duplicated().any() or not hrs.ID.str.fullmatch(r"[0-9]{6}-[0-9]{3}").fillna(False).all():
        raise ValueError("HRS score file has invalid or duplicate HHID-PN identifiers")
    results = {"clinical_prediction_sha256": pred_hash, "model_freeze_sha256": sha(TRAIN_RUN / "model_freeze.json"),
               "n_clinical": len(hrs), "sources": [], "clocks": {}}
    for path, clock_specs in paths_for_comparison:
        clocks = read_clocks(path, {name: spec[0] for name, spec in clock_specs.items()})
        linked = int(clocks.ID.isin(hrs.ID).sum())
        if linked == 0:
            raise ValueError(f"No participant in {path.name} matched frozen HRS predictions")
        results["sources"].append({"file": path.name, "sha256": sha(path), "n": len(clocks), "linked": linked})
        for name, (_, use_ranking) in clock_specs.items():
            results["clocks"][name] = compare(hrs, clocks, name, use_ranking)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, results)
    print(f"Saved aggregate HRS comparisons for {len(results['clocks'])} clocks")


if __name__ == "__main__":
    main()
