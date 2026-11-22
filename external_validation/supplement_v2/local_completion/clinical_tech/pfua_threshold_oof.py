"""Repeat exposure associations using the same quantifiable PFUA threshold across periods."""
import json
from pathlib import Path
import sys

import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
BIOMARKERS = OUT.parents[1] / "biomarkers"
sys.path.insert(0, str(BIOMARKERS))
from exposure_wide import OUTCOMES, STAGES, analyse, get_sample, read_xpt
from score_holdout import PROJECT, TRAIN_RUN, load_json, sha

# A round threshold above the highest cycle-specific lower detection limit.
THRESHOLD = 0.4
SOURCES = {
    2003: ("_C", "L24PFC_C", "WTSA2YR"),
    2005: ("_D", "PFC_D", "WTSA2YR"),
    2007: ("_E", "PFC_E", "WTSC2YR"),
    2009: ("_F", "PFC_F", "WTSC2YR"),
    2011: ("_G", "PFC_G", "WTSA2YR"),
    2013: ("_H", "PFAS_H", "WTSB2YR"),
    2015: ("_I", "PFAS_I", "WTSB2YR"),
    2017: ("_J", "PFAS_J", "WTSB2YR"),
}


def main():
    reference = get_sample()
    manifest = {r["file"][:-4]: r["sha256"] for r in load_json(BIOMARKERS / "sources/extra_manifest.json")}
    records = []
    audit = []
    for cycle, (suffix, name, weight) in SOURCES.items():
        demographics = PROJECT / "data/raw/nhanes" / f"DEMO{suffix}.xpt"
        d = pd.read_sas(demographics, format="xport").set_index("SEQN", verify_integrity=True)
        covariates = {"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH1": "race",
                      "SDMVSTRA": "strata", "SDMVPSU": "psu"}
        d = d[list(covariates)].rename(columns=covariates)
        p = read_xpt(name, manifest)[["LBXPFUA", "LBDPFUAL", weight]].rename(columns={weight: "w"})
        p = p.dropna(subset=["LBXPFUA", "LBDPFUAL", "w"]).loc[lambda x: x.w.gt(1e-12)]
        assert p[p.LBDPFUAL.round().eq(1)].LBXPFUA.lt(THRESHOLD).all()
        p["exposure_z"] = p.LBXPFUA.round(3).ge(THRESHOLD).astype(float)
        frame = reference[reference.cycle.eq(cycle)].join(d, how="inner", validate="one_to_one")
        frame = frame.join(p, how="inner", validate="one_to_one")
        audit.append(dict(
            cycle=cycle, assay_file=name, weight=weight, n=len(frame), n_above_threshold=int(frame.exposure_z.sum()),
            n_below_detection=int(frame.LBDPFUAL.round().eq(1).sum()),
            assay_sha256=manifest[name], demography_sha256=sha(demographics),
        ))
        records.append(frame)
    data = pd.concat(records, verify_integrity=True)
    results = []
    coverage = []
    windows = (data.cycle.between(2009, 2013), data.cycle.between(2015, 2017), data.cycle.lt(2009))
    for stage, window in zip(STAGES, windows):
        for outcome in OUTCOMES:
            subset = data.loc[window].dropna(subset=[outcome, "sex", "race", "age", "strata", "psu"])
            n_high = int(subset.exposure_z.sum())
            n_low = len(subset) - n_high
            coverage.append(dict(stage=stage, score=outcome, n=len(subset), n_high=n_high, n_low=n_low))
            if min(n_high, n_low) >= 25:
                fit = analyse(subset, outcome)
                if fit:
                    results.append(dict(stage=stage, score=outcome, n_high=n_high, n_low=n_low, **fit))
    result = pd.DataFrame(results)
    result["q_BH_all_tests"] = multipletests(result.p, method="fdr_bh")[1]
    result.to_csv(OUT / "xwas_pfua_threshold_oof.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "xwas_pfua_threshold_oof_coverage.csv", index=False)
    status = dict(
        threshold_ng_ml=THRESHOLD, tested_both_high_and_low_with_minimum_count=25,
        development_holdout_predictions_audited_against_fold_hashes=True,
        development_manifest_sha256=sha(TRAIN_RUN / "development_ids.csv.gz"),
        analysis_sha256=sha(OUT / "xwas_pfua_threshold_oof.csv"), source_cycles=audit,
    )
    (OUT / "xwas_pfua_threshold_oof_audit.json").write_text(json.dumps(status, indent=2), encoding="utf8")
    print(result[["stage", "score", "n", "n_high", "beta_accel_years", "lower", "upper", "q_BH_all_tests"]].to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
