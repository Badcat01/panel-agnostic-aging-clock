"""Compare variance from three CHARLS 2011 survey design approximations."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyreadstat
from lifelines import KaplanMeierFitter
from numba import njit

HERE = Path(__file__).resolve().parent
TECH = HERE.parents[1] / "technical"
sys.path.insert(0, str(TECH))
from shared import EXT, REFERENCE, read, sha, verify
from survival_standard import load, uno, weighted_c
from survey_and_fairness import indicators

SCORES = ["full_lp", "ridge_lp", "age_sex_lp"]
N_BOOT = 2000
SEED = 20261121


@njit
def event_risk(t, e, w, horizon):
    """Product-limit incidence, evaluated inclusive of the horizon."""
    total = w.sum()
    survive = 1.0
    i = len(t) - 1
    while i >= 0 and t[i] <= horizon:
        j = i
        exits = 0.0
        deaths = 0.0
        while j >= 0 and t[j] == t[i]:
            exits += w[j]
            deaths += w[j] * e[j]
            j -= 1
        if total > 0:
            survive *= 1.0 - deaths / total
        total -= exits
        i = j
    return 1 - survive


def main():
    verify()
    d = load("charls").dropna(subset=[*SCORES, "sex", "event", "time_yr"])
    design, _ = indicators("charls", d)
    from study_paths import COMMUNITY_ROOT
    root = COMMUNITY_ROOT
    paths = [p for p in root.rglob("psu.dta") if p.parent.name == "2011charls"]
    if len(paths) != 1:
        raise ValueError("2011 community geography source is ambiguous")
    geo_path = paths[0]
    geo, meta = pyreadstat.read_dta(geo_path, usecols=["communityID", "province", "city"])
    geo = geo.set_index("communityID", verify_integrity=True)
    design = design.join(geo, on="psu", validate="many_to_one")
    base = design.loc[design.wt.gt(0)].dropna(subset=["psu"])
    if not base[["province", "city"]].notna().all().all():
        raise ValueError("Some positive-weight community codes lack geography")
    z = d.join(base, how="inner", validate="one_to_one").sort_values("time_yr", ascending=False)
    z["all"] = "all"
    z["city_nested"] = z.province.astype(str) + "|" + z.city.astype(str)
    t, e = z.time_yr.to_numpy(), z.event.to_numpy(dtype=int)
    ranks = [np.unique(z[c], return_inverse=True)[1] + 1 for c in SCORES]
    w = z.wt.to_numpy(dtype=float, copy=True)
    w /= w.mean()
    ref = read(REFERENCE)
    lp = ref["risk_lp_slope"] * (z.full_lp.to_numpy() - ref["risk_lp_mean"])
    prediction = -np.expm1(-ref["baseline_hazard"]["clock"]["3"] * np.exp(lp))

    def evaluate(weight):
        c = [weighted_c(t, e, r, weight) for r in ranks]
        u = [uno(t, e, r, weight, 3.) for r in ranks]
        return np.array([*c, *u, np.average(prediction, weights=weight), event_risk(t,e,weight,3.)])

    point = evaluate(w)
    np.testing.assert_allclose(point[-1], 1 - KaplanMeierFitter().fit(t, e, weights=w).predict(3.))
    reported = pd.read_csv(TECH / "survey_validation.csv").query("cohort == 'charls'").set_index("model")
    np.testing.assert_allclose(point[:3], reported.loc[SCORES, "survey_C"], atol=1e-12)
    assert len(z) == reported.n.iloc[0] and e.sum() == reported.events.iloc[0]
    all_results = []
    for label in ("all", "province", "city_nested"):
        clusters = z[[label, "psu"]].drop_duplicates().reset_index(drop=True).reset_index(names="cluster")
        cluster_id = z.reset_index().merge(clusters, on=[label, "psu"],
                                            validate="many_to_one", sort=False).cluster.to_numpy()
        blocks = list(clusters.groupby(label).cluster.apply(list))
        estimated = []
        rng = np.random.default_rng(SEED)
        for _ in range(N_BOOT):
            rep = np.ones(len(clusters))
            for block in blocks:
                n = len(block)
                if n > 1:
                    rep[block] = np.bincount(rng.integers(0, n, n - 1), minlength=n) * n / (n - 1)
            estimated.append(evaluate(w * rep[cluster_id]))
        simulation = np.array(estimated)
        assert np.isfinite(simulation).all()
        for j, name in enumerate(SCORES):
            for k, metric in enumerate(("Harrell", "Uno_3y")):
                i = k * len(SCORES) + j
                anchor = k * len(SCORES)
                low, high = np.quantile(simulation[:, i], [.025, .975])
                dlow, dhigh = np.quantile(simulation[:, anchor] - simulation[:, i], [.025, .975])
                obslo, obshi = np.quantile(simulation[:, -1], [.025, .975])
                all_results.append(dict(
                    strata_approximation=label, n=len(z), events=int(e.sum()), clusters=len(clusters),
                    strata=len(blocks), singleton_strata=sum(len(b) == 1 for b in blocks),
                    bootstrap_reps=N_BOOT, model=name, metric=metric, estimate=point[i],
                    lo=low, hi=high, clock_minus_model=point[anchor]-point[i], delta_lo=dlow,
                    delta_hi=dhigh, clock_pred_3y=point[-2], observed_3y=point[-1],
                    observed_lo=obslo, observed_hi=obshi))
        print(f"{label}: {len(z)} people, {len(clusters)} communities, {len(blocks)} blocks", flush=True)
    pd.DataFrame(all_results).to_csv(HERE / "charls_cluster_sensitivity.csv", index=False)
    (HERE / "audit.json").write_text(json.dumps({
        "source": geo_path.name, "sha256": sha(geo_path), "unit": "community",
        "weights": "2011 baseline bloodweight", "first_stage_design": "not available",
        "bootstrap": N_BOOT, "seed": SEED, "point_checks": ["prior weighted Harrell C",
                                                            "lifelines weighted Kaplan-Meier"]
    }, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
