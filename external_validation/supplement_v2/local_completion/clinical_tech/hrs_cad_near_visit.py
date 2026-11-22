"""Test HRS coronary disease PGS against the frozen cardiovascular group in dated prior visits."""
import json
from pathlib import Path
import sys

import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
SUPPLEMENT = OUT.parents[1]
sys.path[:0] = [str(SUPPLEMENT / "technical"), str(OUT.parent / "organ_community")]

from hrs_references import PANELS, SOURCES, matrix, source
from hrs_near_visit import assemble


def main():
    scores = assemble().copy()
    scores["age_c"] = (scores.age - 65) / 10
    scores["age_sq"] = scores.age_c ** 2
    pcs = [f"PC{i}{letter}" for i in ("1_5", "6_10") for letter in "ABCDE"]
    controls = ["age_c", "age_sq", "sex", *pcs, "full_accel"]
    result = []
    coverage = []
    for suffix, group in [("e", "EA"), ("a", "AA")]:
        genotypes = source(f"pgenscore4{suffix}_r.dta")
        cad = f"{suffix.upper()}4_CD_CARDIOGRAM11"
        joined = scores.join(genotypes[[cad, *pcs]], how="inner", validate="one_to_one")
        for days in (0, 7, 30, 90, 180):
            frame = joined[joined.lag_days.between(0, days) & joined.cardiovascular_accel.notna()]
            complete = frame[[cad, *controls, *[p + "_accel" for p in PANELS]]].dropna()
            assert complete.index.is_unique and len(complete) <= len(frame)
            coverage.append(dict(
                ancestry=group, max_days_before_blood_draw=days, n_id_overlap=len(joined),
                n_with_timed_bp_pulse=len(frame), n_complete_all_five=len(complete),
                median_days_before_blood_draw=float(frame.loc[complete.index, "lag_days"].median()),
                estimable=len(complete) >= 40,
            ))
            result.extend(dict(max_days_before_blood_draw=days, **row) for row in matrix(
                complete, cad, "CD_CARDIOGRAM11", "cardiovascular", group, controls, PANELS,
            ))
    df = pd.DataFrame(result)
    match = df.panel.eq("cardiovascular")
    df.loc[match, "dominance_q_BH_all_windows"] = multipletests(
        df.loc[match, "dominance_p_wild_IUT"], method="fdr_bh"
    )[1]
    df["association_q_BH_all_windows"] = multipletests(df.p, method="fdr_bh")[1]
    assert (df.groupby(["group", "max_days_before_blood_draw"]).panel.nunique() == 5).all()
    coverage = pd.DataFrame(coverage)
    assert coverage.query("max_days_before_blood_draw==0").n_complete_all_five.sum() == 0
    df.to_csv(OUT / "hrs_cad_pgs_near_visit.csv", index=False)
    coverage.to_csv(OUT / "hrs_cad_pgs_near_visit_coverage.csv", index=False)
    audit = {
        "same_day_pair_count": 0,
        "window_limits_days_before_draw": [0, 7, 30, 90, 180],
        "min_sample_size_for_existing_matrix_function": 40,
        "primary_90_day_window_chosen_before_genetic_association": True,
        "dominance_test_family_n": int(match.sum()),
        "household_wild_sign_draws": 2000,
        "score_inputs_and_freeze_sha256": scores.attrs,
        "genotype_sources": SOURCES,
    }
    (OUT / "hrs_cad_pgs_near_visit_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(coverage.to_string(index=False))
    print(df.loc[match, ["group", "max_days_before_blood_draw", "n", "partial_r", "ci95_low",
                        "ci95_high", "dominance_q_BH_all_windows"]].to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
