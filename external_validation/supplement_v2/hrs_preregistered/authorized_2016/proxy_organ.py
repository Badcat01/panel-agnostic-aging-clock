"""Paired HRS GrimAge2 surrogate correlations using the NHANES organ hypotheses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits


PANELS = ("liver", "kidney", "metabolic", "immune_heme", "cardiovascular")
# The corresponding NHANES markers and organ hypotheses are in biomarkers/score_holdout.py.
MATCHES = {
    "proxy_cystatin": ("kidney",),
    "proxy_b2m": ("kidney", "immune_heme"),
    "proxy_leptin": ("metabolic",),
    "proxy_loga1c": ("metabolic",),
    "proxy_logcrp": ("immune_heme",),
    "proxy_gdf15": ("immune_heme",),
    "proxy_pai1": ("cardiovascular",),
    "proxy_adm": ("cardiovascular",),
}
SEED = 20261121
RESULT_COLUMNS = [
    "marker", "panel", "matched", "n", "households", "partial_r", "ci95_low",
    "ci95_high", "association_p", "association_q_BH", "nonmatched_panels",
    "min_matched_advantage", "dominance_p_wild_IUT", "dominance_q_BH",
]


def dominance(r, cluster_scores, matched_index, others):
    contrasts = [(j, sign) for j in others for sign in (1, -1)]
    estimates = np.array([r[matched_index] - sign*r[j] for j, sign in contrasts])
    effects = np.column_stack([
        cluster_scores[:, matched_index] - sign*cluster_scores[:, j] for j, sign in contrasts
    ])
    rng = np.random.default_rng(SEED)
    exceed = np.ones(len(contrasts))
    for _ in range(20):
        signs = rng.choice([-1, 1], size=(100, len(cluster_scores)))
        exceed += np.sum(signs @ effects >= estimates, axis=0)
    return estimates.min(), (exceed / 2001).max()


def run(path, output):
    status = json.loads((output / "run_status.json").read_text(encoding="utf8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != status["input_hash"]:
        raise ValueError("Input differs from the paired mortality run")
    data = pd.read_csv(path, dtype={"ID": "string"}).set_index("ID", verify_integrity=True)
    if data.index.hasnans or not data.index.str.fullmatch(r"[0-9]{6}-[0-9]{3}").all():
        raise ValueError("Expected a 6-digit household and 3-digit person identifier")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.loc[data.qc_pass.eq(1) & data.age.ge(50) & data.time_yr.gt(0)
                    & data.event.isin([0, 1]) & data.sex.isin([0, 1]) & data.clinical.notna()]
    if "clock_GrimAge2" in data:
        data = data.loc[data.clock_GrimAge2.notna()]

    rows, skipped = [], []
    panels = [p for p in PANELS if "organ_"+p in data and data["organ_"+p].notna().any()]
    for marker, matches in MATCHES.items():
        for p in set(matches).difference(panels):
            skipped.append((marker, p, "matched organ score unavailable"))
        if marker not in data or not data[marker].notna().any() or not panels:
            skipped.append((marker, "", "surrogate or all organ scores unavailable"))
            continue
        columns = [marker, *["organ_"+p for p in panels]]
        same = data[[*columns, "age", "sex", "clinical"]].dropna()
        family = same.index.str.slice(0, 6)
        n, g = len(same), family.nunique()
        if n < 40 or g < 2:
            skipped.append((marker, "", "fewer than 40 paired participants or two households"))
            continue
        controls = np.column_stack([np.ones(n), same[["age", "sex", "clinical"]]])
        values = same[columns].to_numpy(dtype=float)
        if np.linalg.matrix_rank(controls) < controls.shape[1]:
            skipped.append((marker, "", "rank-deficient age, sex and whole-body adjustment"))
            continue
        residual = values - controls @ np.linalg.lstsq(controls, values, rcond=None)[0]
        sd = np.sqrt(np.mean(residual**2, axis=0))
        scale = np.maximum(1, np.abs(values).max(axis=0))
        if np.any(sd <= np.finfo(float).eps * scale * max(controls.shape)):
            skipped.append((marker, "", "constant adjusted surrogate or organ; common comparison unavailable"))
            continue
        std = residual / sd
        y, x = std[:, :1], std[:, 1:]
        r = (x*y).mean(axis=0)
        influence = x*y - .5*r*(x**2 + y**2)
        scores = pd.DataFrame(influence).groupby(family.to_numpy()).sum().to_numpy()/n
        se = np.sqrt((scores**2).sum(axis=0) * g/(g-1))
        others = [j for j, p in enumerate(panels) if p not in matches]
        for j, p in enumerate(panels):
            row = dict(marker=marker, panel=p, matched=p in matches, n=n, households=g,
                       partial_r=r[j], ci95_low=max(-1., r[j]-1.96*se[j]),
                       ci95_high=min(1., r[j]+1.96*se[j]),
                       association_p=2*norm.sf(abs(r[j])/se[j]) if se[j]>0 else np.nan)
            if p in matches:
                row["nonmatched_panels"] = "|".join(panels[k] for k in others)
                if others:
                    row["min_matched_advantage"], row["dominance_p_wild_IUT"] = dominance(r,scores,j,others)
                else:
                    skipped.append((marker, p, "no nonmatched organ available"))
            rows.append(row)
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    for source, dest in [("association_p","association_q_BH"), ("dominance_p_wild_IUT","dominance_q_BH")]:
        obs = result[source].notna()
        if obs.any():
            result.loc[obs, dest] = multipletests(result.loc[obs, source], method="fdr_bh")[1]
    result.to_csv(output / "dnam_proxy_organ.csv", index=False)
    pd.DataFrame(skipped, columns=["marker","panel","reason"]).to_csv(
        output / "dnam_proxy_unavailable.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.input, args.output)
