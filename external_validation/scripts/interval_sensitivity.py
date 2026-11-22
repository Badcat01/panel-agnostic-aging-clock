"""Evaluate guaranteed follow-up order and deterministic endpoint timing scenarios."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from reference import HORIZONS, REFERENCE, ROOT, sha
from validate_external import censor_probabilities


MODELS = ["full_lp", "age_sex_lp", "ridge_lp"]
ENDPOINTS = ["event", "time_yr", "time_lower_yr", "time_upper_yr"]


def cohort_data(name):
    folder = ROOT / "results" / name
    inputs = ROOT / ("sensitivity_inputs" if name == "charls_short" else "inputs") / (name + ".csv.gz")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf8"))
    if sha(inputs) != metrics["source_sha256"] or sha(REFERENCE) != metrics["reference_sha256"]:
        raise ValueError(f"{name}: data identity changed since the model assessment")
    x = pd.read_csv(inputs, dtype={"ID":str}, usecols=["ID", *ENDPOINTS])
    scores = pd.read_csv(folder / "predictions.csv.gz", dtype={"ID":str},
                         usecols=["ID", *ENDPOINTS, *MODELS])
    d = scores.merge(x, on="ID", how="left", suffixes=("","_original"), validate="one_to_one", indicator=True)
    if not d._merge.eq("both").all():
        raise ValueError("Missing external participant ID in source inputs")
    for var in ENDPOINTS:
        np.testing.assert_allclose(d[var], d[var+"_original"], atol=1e-12,rtol=1e-12, equal_nan=True)
    d = d.loc[d.time_yr.gt(0)].dropna(subset=MODELS+ENDPOINTS)
    if len(d) != metrics["n_sex"] or int(d.event.sum()) != metrics["deaths_sex"]:
        raise ValueError(f"{name}: complete shared evaluation population changed")
    assert (d.time_lower_yr >= 0).all() and (d.time_lower_yr <= d.time_yr+1e-12).all()
    assert (d.time_upper_yr >= d.time_yr-1e-12).all()
    np.testing.assert_allclose(d.loc[d.event==0,"time_lower_yr"],d.loc[d.event==0,"time_upper_yr"])
    return d, metrics


def certain_concordance(d):
    # For a subject with a death upper bound U_i, any subject with L_j>U_i
    # survived past i's possible death. Neither endpoint equality nor pair
    # ambiguity is resolved by imposing an artificial order.
    event = d.event.to_numpy(bool)
    lo, hi = d.time_lower_yr.to_numpy(), d.time_upper_yr.to_numpy()
    lp = d[MODELS].to_numpy(float)
    pairs = events_with_pairs = 0
    wins = np.zeros(len(MODELS))
    for i in np.flatnonzero(event):
        comp = lo > hi[i]
        k = int(comp.sum())
        if k:
            other = lp[comp]
            wins += (other < lp[i]).sum(axis=0) + 0.5 * (other == lp[i]).sum(axis=0)
            pairs += k
            events_with_pairs += 1
    c = wins/pairs if pairs else wins*np.nan
    return {"n":len(d),"deaths":int(event.sum()),"deaths_with_identifiable_pairs":events_with_pairs,
            "n_pairs":pairs, "C_clock":c[0], "C_age_sex":c[1], "C_ridge_cox":c[2],
            "delta_C_clock_minus_age_sex":c[0]-c[1],"delta_C_clock_minus_ridge_cox":c[0]-c[2],
            "rule":"death upper < comparator lower; half credit for tied risk scores"}


def time_scenarios(name, d, ref, metrics):
    event = d.event.to_numpy(bool)
    lo, hi = d.time_lower_yr.to_numpy(), d.time_upper_yr.to_numpy()
    rows = []
    # A two-year follow-up sensitivity population additionally has descriptive
    # one- and two-year rows but the frozen risk map only covers 3, 5, 10 years.
    horizons = [1,2,*HORIZONS] if name=="charls_short" else HORIZONS
    for label, t in (("all_deaths_at_lower", lo), ("all_deaths_at_upper", hi)):
        km = KaplanMeierFitter().fit(t,event)
        levels, g_after, _ = censor_probabilities(t,event)
        for h in horizons:
            idx = np.searchsorted(levels,h,side="right")-1
            g = g_after[idx] if idx>=0 else 1.
            cases, controls = int((event & (t<=h)).sum()), int((t>h).sum())
            supported = min(cases,controls)>=20 and g>0.05
            certain_dead = int((event & (hi<=h)).sum())
            ambiguous = int((event & (lo<=h) & (hi>h)).sum())
            followed = int((lo>h).sum())
            censored_before = int((~event & (lo<=h)).sum())
            assert certain_dead+ambiguous+followed+censored_before==len(d)
            row = dict(cohort=name, horizon_years=h, timing_scenario=label,n=len(d),
                       observed_deaths_in_scenario_by_h=cases,observed_beyond_h_in_scenario=controls,
                       known_death_upper_by_h=certain_dead, death_interval_straddles_h=ambiguous,
                       lower_followup_beyond_h=followed, censored_by_or_at_h=censored_before,
                       censor_survival_at_h=float(g),supported_by_main_rule=bool(supported),
                       km_descriptive_if_h_in_range=float(1-km.predict(h)) if h<=t.max() else np.nan,
                       km_if_supported=float(1-km.predict(h)) if supported else np.nan,
                       note="Deterministic dating scenarios; not confidence bounds. Unsupported KM is descriptive only.")
            if str(h) in ref["baseline_hazard"]["clock"]:
                lp = ref["risk_lp_slope"]*(d.full_lp.to_numpy()-ref["risk_lp_mean"])
                row["frozen_clock_mean_risk"] = float(np.mean(-np.expm1(-ref["baseline_hazard"]["clock"][str(h)]*np.exp(lp))))
                if metrics["fixed_times"][str(h)].get("supported"):
                    np.testing.assert_allclose(row["frozen_clock_mean_risk"],metrics["fixed_times"][str(h)]["clock"]["predicted_mean"],rtol=1.e-10)
            rows.append(row)
    return rows


if __name__=="__main__":
    ref = json.loads(REFERENCE.read_text(encoding="utf8"))
    pairs, scenarios = [], []
    for cohort in ("charls","charls_short","elsa","hrs"):
        data, summary = cohort_data(cohort)
        pairs.append({"cohort":cohort, **certain_concordance(data)})
        scenarios.extend(time_scenarios(cohort,data,ref,summary))
    destination = ROOT/"results"
    pd.DataFrame(scenarios).to_csv(destination/"interval_calibration_scenarios.csv",index=False)
    table = pd.DataFrame(pairs)
    table.to_csv(destination/"interval_sensitivity.csv",index=False)
    print(table.iloc[:,:9].to_string(index=False))
