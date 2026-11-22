"""Build NHANES III 1988–94 first-examination data linked to 2019 mortality."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nhanes3_source"
CDC = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/"
CDC_MORT = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
DEATH_FILE = "NHANES_III_MORT_2019_PUBLIC.dat"

LABS = {
    "albumin": ("AMP", "g/dL"), "alp": ("APPSI", "U/L"), "ast": ("ASPSI", "U/L"),
    "alt": ("ATPSI", "U/L"), "ggt": ("GGPSI", "U/L"), "bilirubin": ("TBP", "mg/dL"),
    "total_protein": ("TPP", "g/dL"), "globulin": ("GBP", "g/dL"), "ldh": ("LDPSI", "U/L"),
    "creatinine": ("CEP", "mg/dL"), "bun": ("BUP", "mg/dL"), "uric_acid": ("UAP", "mg/dL"),
    "potassium": ("SKPSI", "mmol/L"), "sodium": ("NAPSI", "mmol/L"),
    "chloride": ("CLPSI", "mmol/L"), "bicarbonate": ("C3PSI", "mmol/L"),
    "phosphorus": ("PSP", "mg/dL"), "calcium": ("SCP", "mg/dL"),
    "glucose": ("SGP", "mg/dL"), "hba1c": ("GHP", "%"),
    "triglycerides": ("TRP", "mg/dL"), "cholesterol": ("CHP", "mg/dL"), "hdl": ("HDP", "mg/dL"),
    "wbc": ("WCPSI", "10^9/L"), "lymph_pct": ("LMPPCNT", "%"), "mono_pct": ("MOPDIF", "%"),
    "neut_pct": ("GRPDIF", "%"), "eos_pct": ("EOP", "%"), "crp": ("CRP", "mg/dL"),
    "rbc": ("RCPSI", "10^12/L"), "hgb": ("HGP", "g/dL"), "mcv": ("MVPSI", "fL"),
    "mch": ("MCPSI", "pg"), "rdw": ("RWP", "%"), "platelets": ("PLPSI", "10^9/L"),
    "mpv": ("PVPSI", "fL"), "iron": ("FEP", "µg/dL"),
}
SBP = ["PEP6G1", "PEP6H1", "PEP6I1"]
DBP = ["PEP6G3", "PEP6H3", "PEP6I3"]
EXAM_VARS = [
    "SEQN", "DMPSTAT", "HSSEX", "MXPAXTMR", "SDPPHASE", "WTPFEX6", "SDPPSU6", "SDPSTRA6",
    "MAPF12R", "BMPBMI", "BMPWAIST", "PEP6DR", *SBP, *DBP,
]
MORT_VARS = ["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_EXM"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_layout(filename, variables):
    """Parse 1-based, inclusive fields in the official SAS INPUT section."""
    content = (SOURCE / filename).read_text(encoding="latin-1")
    tail = re.split(r"^\s*INPUT\s*$", content, flags=re.MULTILINE)[-1].split(";")[0]
    all_fields = {}
    for match in re.finditer(r"^\s*(\w+)\s+\$?(\d+)(?:-(\d+))?\s*$", tail, re.MULTILINE):
        name, start, end = match.groups()
        all_fields[name] = (int(start) - 1, int(end or start))
    return {name: all_fields[name] for name in variables}


def read_fwf(filename, layout, widths, expected_rows, missing_codes=True):
    """Read declared coordinates, checking record widths and unique participant IDs."""
    records = {name: [] for name in layout}
    with (SOURCE / filename).open("rb") as stream:
        for line in stream:
            if len(line.rstrip(b"\r\n")) not in widths:
                raise ValueError(f"Unexpected fixed width in {filename}: {len(line)}")
            for name, (start, end) in layout.items():
                records[name].append(line[start:end].strip())
    missing = []
    for name, (start, end) in layout.items():
        raw = pd.Series(records[name], dtype=object)
        masked = raw.isin([b"", b"."])
        if missing_codes and name != "SEQN" and name not in {"SDPPSU6", "SDPSTRA6", "WTPFEX6"}:
            masked |= raw.eq(b"8" * (end - start))
        raw = raw.mask(masked)
        records[name] = pd.to_numeric(raw, errors="raise")
        missing.append((name, int(masked.sum())))
    data = pd.DataFrame(records)
    if len(data) != expected_rows or data.SEQN.isna().any():
        raise ValueError(f"Unexpected record count or missing ID in {filename}")
    return data.set_index("SEQN", verify_integrity=True), dict(missing)


def build():
    adult_cols = ["SEQN", "DMPSTAT", "HSSEX", "HSAGEIR"]
    lab_cols = ["SEQN", "DMPSTAT", "HSSEX", "MXPAXTMR", "URP", "UBP", *[v for v, _ in LABS.values()]]
    adult_layout = load_layout("adult.sas", adult_cols)
    exam_layout = load_layout("exam.sas", EXAM_VARS)
    lab_layout = load_layout("lab.sas", lab_cols)
    mort_layout = load_layout("SAS_ReadInProgramAllSurveys.sas", MORT_VARS)
    adult, adult_codes = read_fwf("adult.dat", adult_layout, {3345, 3346}, 20050)
    exam, exam_codes = read_fwf("exam.dat", exam_layout, {6229, 6233}, 31311)
    lab, lab_codes = read_fwf("lab.dat", lab_layout, {1971, 1977}, 29314)
    mortality, _ = read_fwf(DEATH_FILE, mort_layout, {46, 47, 48}, 33994, missing_codes=False)
    if not exam.index.isin(mortality.index).all() or not lab.index.isin(exam.index).all() or not adult.index.isin(mortality.index).all():
        raise ValueError("CDC survey components and mortality identifiers do not reconcile")
    for frame in [adult, lab]:
        shared = frame.index.intersection(exam.index)
        for field in ["DMPSTAT", "HSSEX"]:
            if not frame.loc[shared, field].equals(exam.loc[shared, field]):
                raise ValueError(f"Core records disagree on {field}")
    # Use the actual MEC age even for residents interviewed just before a birthday.
    for name in ["MXPAXTMR"]:
        if not lab[name].equals(exam.loc[lab.index, name]):
            raise ValueError(f"Laboratory and examination age disagree: {name}")
    population = exam.join(mortality, validate="one_to_one")
    subset = population.loc[population.DMPSTAT.eq(2)].copy()
    age_months = subset.MXPAXTMR
    if age_months.isna().any() or not age_months.between(0, 1080).all():
        raise ValueError("Invalid public-use MEC age")
    subset["age"] = np.floor(age_months / 12)
    progression = {"examination_all": len(population), "MEC_only": len(subset)}
    subset = subset.loc[subset.age.ge(20)].copy()
    progression["age_20_plus"] = len(subset)
    subset = subset.loc[~subset.MAPF12R.eq(1)].copy()
    progression["exclude_known_pregnancy"] = len(subset)
    subset = subset.loc[subset.ELIGSTAT.eq(1) & subset.PERMTH_EXM.gt(0) & subset.MORTSTAT.isin([0, 1])].copy()
    progression["eligible_positive_MEC_followup"] = len(subset)
    subset["sex"] = subset.HSSEX.map({1: 1, 2: 0})
    subset["time_yr"] = subset.PERMTH_EXM / 12
    subset["event"] = subset.MORTSTAT.astype(int)
    for feature, (variable, _) in LABS.items():
        subset[feature] = lab[variable].reindex(subset.index)
    # Standardized calibration reported in the CDC public laboratory codebook.
    subset["creatinine_original"] = subset["creatinine"]
    subset["creatinine"] = (0.960 * subset["creatinine"] - 0.184).where(subset.creatinine.gt(0.184 / .960))
    urinary = lab.loc[:, ["UBP", "URP"]].reindex(subset.index)
    subset["uacr"] = 100 * urinary.UBP / urinary.URP.where(urinary.URP.gt(0))
    subset["bmi"] = subset.BMPBMI
    subset["waist"] = subset.BMPWAIST
    subset["pulse"] = subset.PEP6DR
    subset["sbp"] = subset[SBP].where(subset[SBP].gt(0)).mean(axis=1)
    subset["dbp"] = subset[DBP].where(subset[DBP].gt(0)).mean(axis=1)
    features = [*LABS, "uacr", "bmi", "waist", "sbp", "dbp", "pulse"]
    out = subset.loc[subset[features].notna().any(axis=1),
        ["age", "sex", "time_yr", "event", *features, "creatinine_original", "SDPPHASE", "WTPFEX6", "SDPPSU6", "SDPSTRA6"]]
    out = out.rename_axis("ID").reset_index()
    if len(features) != 43 or not np.isfinite(out["time_yr"]).all() or out.sex.isna().any():
        raise ValueError("Invalid outcome, sex or feature schema")
    if ((out[features] < 0).any()).any():
        raise ValueError("Negative concentration or examination measure")
    n_features = out[features].notna().sum(axis=1)
    progression["one_or_more_verified_inputs"] = len(out)
    pheno9 = ["albumin", "creatinine", "glucose", "crp", "lymph_pct", "mcv", "rdw", "alp", "wbc"]
    sources = {
        name: {"url": (CDC_MORT if name in {DEATH_FILE, "SAS_ReadInProgramAllSurveys.sas"} else CDC) + name,
               "bytes": (SOURCE / name).stat().st_size, "sha256": sha256(SOURCE / name)}
        for name in ["adult.dat", "exam.dat", "lab.dat", "adult.sas", "exam.sas", "lab.sas",
                     "adult-acc.pdf", "exam-acc.pdf", "lab-acc.pdf", DEATH_FILE, "SAS_ReadInProgramAllSurveys.sas"]
    }
    mapping = {k: {"variable": v, "unit": u} for k, (v, u) in LABS.items()}
    mapping.update({
        "uacr": {"variable": "100 × UBP (µg/mL) ÷ URP (mg/dL)", "unit": "mg/g"},
        "bmi": {"variable": "BMPBMI", "unit": "kg/m²"},
        "waist": {"variable": "BMPWAIST", "unit": "cm"},
        "pulse": {"variable": "PEP6DR", "unit": "beats/min"},
        "sbp": {"variable": SBP, "unit": "mmHg", "summary": "mean of nonzero same-MEC readings"},
        "dbp": {"variable": DBP, "unit": "mmHg", "summary": "mean of nonzero same-MEC readings"},
    })
    audit = {
        "cohort": "NHANES III (1988–1994)", "baseline": "first NHANES III mobile examination center visit",
        "sources": sources,
        "documentation": [
            "https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx",
            "https://www.cdc.gov/nchs/linked-data/mortality-files/",
            "https://pubmed.ncbi.nlm.nih.gov/18037092/",
        ],
        "eligibility": progression, "n_baseline_scored": len(out), "n_survival": len(out), "deaths": int(out.event.sum()),
        "followup_years_median_observed": float(out.time_yr.median()),
        "n_age_topcoded_90plus": int(out.age.eq(90).sum()),
        "age": "floor(MXPAXTMR/12); public ages 90 and over are top coded here to 90",
        "endpoint": "First examination to linked all-cause death or administrative censoring at end of 2019. "
                    "NCHS public-use PERMTH_EXM was used as reported. Event is MORTSTAT. For some participants "
                    "public-use files substitute synthetic follow-up time or cause of death. Vital status is preserved.",
        "feature_mapping": mapping,
        "n_features_q25_q50_q75": n_features.quantile([.25,.50,.75]).tolist(),
        "n_ge22_model_features": int(n_features.ge(22).sum()),
        "n_pheno9_complete": int(out[pheno9].notna().all(axis=1).sum()),
        "n_crp_at_cdc_below_detection_code_0_21_mg_dl": int(out.crp.eq(.21).sum()),
        "n_crp_measured": int(out.crp.notna().sum()),
        "n_creatinine_calibrated": int(out.creatinine.notna().sum()),
        "creatinine_calibration": "0.960 × CEP − 0.184 mg/dL (IDMS-traceable, CDC NHANES III lab codebook and Selvin et al. 2007).",
        "n_by_feature": {k:int(out[k].notna().sum()) for k in features},
        "notes": [
            "Pregnancy status (MAPF12R=1) excluded. Unknown/not applicable retained, matching the frozen development cohort.",
            "Selected blank and repeated-8 fields were masked using CDC fixed-width coding. Other numeric observations retained.",
            "Lymphocytes are Coulter automated percentages. Monocytes, segmented neutrophils and eosinophils use a manual differential "
            "from a random 10% QC subset plus abnormal hematology. The automated mononuclear/granulocyte fractions are different analytes.",
            "CRP 0.21 mg/dL denotes below-level-of-detection values in this release. Original published concentrations were retained.",
            "Iron uses whole-cohort FEP, the separately reported serum iron analyte. HDL uses the dedicated specimen concentration HDP.",
            "Complex survey weights and strata retained for possible sensitivity analysis. Main evaluation uses unweighted statistics.",
            "A change of era and laboratory platform is expected. No external outcomes were used for unit mapping or calibration.",
        ],
        "codebook_missing_counts": {"adult":adult_codes,"exam":exam_codes,"lab":lab_codes},
    }
    (ROOT/"inputs").mkdir(exist_ok=True)
    (ROOT/"audit").mkdir(exist_ok=True)
    out.to_csv(ROOT/"inputs"/"nhanes3.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    (ROOT/"audit"/"nhanes3.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    print(progression, "events", audit["deaths"], "features", audit["n_features_q25_q50_q75"], flush=True)


if __name__ == "__main__":
    build()
