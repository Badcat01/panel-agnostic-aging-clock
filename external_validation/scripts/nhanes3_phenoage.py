"""Compare frozen FT-Mask to clinical PhenoAge on the latter's source cohort."""

import json
import sys

import pandas as pd
import torch

from reference import BUNDLE, ROOT, REFERENCE, PROJECT, save_json, score_frame, sha, verify
from validate_external import discrimination

sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_clock import phenoage_years

PANEL = ["albumin", "creatinine", "glucose", "crp", "lymph_pct", "mcv", "rdw", "alp", "wbc"]


def main():
    verify()
    directory = ROOT / "results" / "nhanes3"
    source = ROOT / "inputs" / "nhanes3.csv.gz"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf8"))
    if metrics["source_sha256"] != sha(source) or metrics["reference_sha256"] != sha(REFERENCE):
        raise ValueError("NHANES III scored data do not match the recorded inputs")
    raw = pd.read_csv(source, dtype={"ID": "string"})
    scores = pd.read_csv(directory / "predictions.csv.gz", dtype={"ID": "string"})
    raw = raw.set_index("ID", verify_integrity=True)
    scores = scores.set_index("ID", verify_integrity=True)
    common = raw.loc[scores.index]
    phenoage = phenoage_years(common.assign(creatinine=common["creatinine_original"]))
    evaluation = scores.loc[phenoage.notna() & scores.time_yr.gt(0) & scores.event.notna()].copy()
    evaluation["phenoage"] = phenoage.loc[evaluation.index]
    evaluation["phenoage_calibrated_creatinine"] = phenoage_years(common).loc[evaluation.index]
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    masked = score_frame(common.loc[evaluation.index, ["age", *PANEL]], BUNDLE, "cuda")
    evaluation["clock_same_nine"] = masked.full_lp
    scores_all = discrimination(evaluation, ["full_lp", "phenoage", "clock_same_nine", "ridge_lp", "age_sex_lp",
                                               "phenoage_calibrated_creatinine"])
    scores_nine = discrimination(evaluation, ["clock_same_nine", "phenoage"])
    save_json(directory / "phenoage_comparison.json", {
        "n": len(evaluation),
        "deaths": int(evaluation.event.sum()),
        "panel": PANEL,
        "all_available_features": scores_all,
        "same_nine_features": scores_nine,
        "reference": "Levine et al. 2018, doi:10.18632/aging.101414",
        "creatinine": "The primary PhenoAge uses the original NHANES III measurement scale. An additional PhenoAge score uses calibrated creatinine. The frozen clock uses its harmonized inputs.",
        "interpretation": "Clinical PhenoAge was developed in NHANES III; this is a descriptive comparison on its source cohort.",
    })
    print("NHANES III PhenoAge comparison", len(evaluation), int(evaluation.event.sum()))


if __name__ == "__main__":
    main()
