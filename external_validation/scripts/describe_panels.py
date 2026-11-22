"""Descriptive, unadjusted subgroup comparisons for primary external baselines.

Each panel requires at least one observed biomarker in that panel. Every C-index
comparison uses identical participants and follow-up. One-feature organ scores
describe the restricted model input; they do not independently establish tissue
age or organ specificity. The country-stratified C uses only within-country pairs.
"""
import json

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from reference import REFERENCE, ROOT, sha, verify


# Value labels in the local SHARE wave 6 DBS release and all-waves coverscreen.
COUNTRIES = {12:"Germany", 13:"Sweden", 15:"Spain", 16:"Italy", 17:"France",
             18:"Denmark", 19:"Greece", 20:"Switzerland", 23:"Belgium",
             25:"Israel", 34:"Slovenia", 35:"Estonia"}


def c_stats(frame, columns):
    if not len(frame):
        return 0, {c:np.nan for c in columns}
    t = frame.time_yr.to_numpy()
    e = frame.event.to_numpy(dtype=int)
    levels, ix, counts = np.unique(t, return_inverse=True, return_counts=True)
    deaths = np.bincount(ix, weights=e, minlength=len(levels)).astype(np.int64)
    # Strictly later people and same-time censors are admissible comparisons.
    pairs = int(np.sum(deaths * (len(t)-np.cumsum(counts) + counts-deaths)))
    scores = {c:concordance_index(t, -frame[c].to_numpy(), e) if pairs else np.nan
              for c in columns}
    return pairs, scores


def load_data(name, bundle):
    folder = ROOT/"results"/name
    path = ROOT/"inputs"/(name+".csv.gz")
    met = json.loads((folder/"metrics.json").read_text(encoding="utf8"))
    if sha(path)!=met["source_sha256"] or sha(REFERENCE)!=met["reference_sha256"]:
        raise ValueError(f"{name}: frozen evaluation data hashes differ")
    raw = pd.read_csv(path, dtype={"ID":str})
    d = pd.read_csv(folder/"predictions.csv.gz", dtype={"ID":str})
    raw = raw.loc[np.isfinite(raw.age) & raw.age.ge(20)].reset_index(drop=True)
    pd.testing.assert_series_equal(raw.ID, d.ID)
    assert raw.ID.is_unique
    for c in ["time_yr","event","age","sex"]:
        np.testing.assert_allclose(raw[c],d[c],equal_nan=True,rtol=0,atol=1e-12)
    for p, features in {"full":bundle["features"], **bundle["groups"]}.items():
        np.testing.assert_array_equal(raw.reindex(columns=features).notna().sum(axis=1),d[p+"_n_observed"])
        assert d[p+"_lp"].notna().equals(d[p+"_n_observed"].gt(0))
    if name=="share":
        d["country"] = raw.country
    return raw,d,met


def describe(name, kind, label, panel, group, types):
    can_score = group[panel+"_n_observed"].gt(0)
    measured = group.loc[can_score]
    surv = measured.loc[measured.time_yr.gt(0) & measured.event.isin([0,1])].dropna(subset=["age",panel+"_lp"])
    pairs, c = c_stats(surv,[panel+"_lp","age"])
    return dict(cohort=name,stratum=kind,group=label,panel=panel,available_feature_types=types,
        baseline_denominator=len(group),scored_baseline=len(measured),
        baseline_coverage_fraction=len(measured)/len(group) if len(group) else np.nan,
        median_features_when_scored=measured[panel+"_n_observed"].median() if len(measured) else np.nan,
        n_survival=len(surv),deaths=int(surv.event.sum()),comparable_pairs=pairs,
        C_clock=c[panel+"_lp"],C_age_same_people=c["age"],delta_C=c[panel+"_lp"]-c["age"],
        fewer_than_20_deaths=len(surv)>0 and surv.event.sum()<20)


def by_country(d, met):
    rows = []
    for code, sub in d.groupby("country",dropna=False):
        country = COUNTRIES[int(code)]
        sub = sub.loc[sub.full_lp.notna()]
        f = sub.loc[sub.time_yr.gt(0) & sub.event.isin([0,1])].dropna(subset=["age_sex_lp","full_lp"])
        pairs, c = c_stats(f,["age_sex_lp","full_lp"])
        rows.append(dict(country_code=int(code),country=country,baseline_scored=len(sub),
            n_survival=len(f),deaths=int(f.event.sum()),comparable_pairs=pairs,
            C_clock=c["full_lp"],C_age_sex=c["age_sex_lp"],delta_C=c["full_lp"]-c["age_sex_lp"]))
    total = pd.DataFrame(rows)
    assert len(total)==12 and total.n_survival.sum()==met["n_sex"] and total.deaths.sum()==met["deaths_sex"]
    valid = total.comparable_pairs.gt(0)
    w = total.loc[valid,"comparable_pairs"]
    dl = np.average(total.loc[valid,"C_clock"],weights=w)
    std = np.average(total.loc[valid,"C_age_sex"],weights=w)
    rows.append(dict(country_code="",country="Within-country pairs, weighted by admissible pair count",
        baseline_scored=int(total.baseline_scored.sum()),n_survival=int(total.n_survival.sum()),
        deaths=int(total.deaths.sum()),comparable_pairs=int(w.sum()),C_clock=dl,C_age_sex=std,delta_C=dl-std))
    return pd.DataFrame(rows)


def main():
    bundle = verify()
    rows = []
    for name in ["charls","hrs","elsa","mhas","share","klosa","lasi"]:
        raw,d,met = load_data(name,bundle)
        panels = {"full":bundle["features"], **bundle["groups"]}
        for p, features in panels.items():
            feature_count = int(raw.reindex(columns=features).notna().any().sum())
            row = describe(name,"panel",p,p,d,feature_count)
            if p=="full":
                assert row["n_survival"]==met["n"] and row["deaths"]==met["deaths"]
                if met.get("all"):
                    np.testing.assert_allclose(row["C_clock"],met["all"]["full_lp"]["C"],atol=2e-7)
            rows.append(row)
        groups = [("sex","female",d.sex.eq(0)),("sex","male",d.sex.eq(1)),
            ("age","20-49",d.age.between(20,50,inclusive="left")),
            ("age","50-69",d.age.between(50,70,inclusive="left")),("age","70+",d.age.ge(70))]
        for kind,label,take in groups:
            ntypes = int(raw.loc[take].reindex(columns=bundle["features"]).notna().any().sum())
            rows.append(describe(name,kind,label,"full",d.loc[take],ntypes))
        if name=="share":
            countries = by_country(d,met)
    pd.DataFrame(rows).to_csv(ROOT/"results"/"panel_subgroups.csv",index=False)
    countries.to_csv(ROOT/"results"/"share_country.csv",index=False)


if __name__=="__main__":
    main()
