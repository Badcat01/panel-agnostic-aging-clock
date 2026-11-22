"""Evaluate frozen risk scores in one external cohort, once per participant."""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from numba import njit
from sklearn.metrics import roc_auc_score

from reference import ROOT, TRAIN_RUN, BUNDLE, REFERENCE, HORIZONS, linear_score, prepare_features, score_frame, sha, save_json, verify


@njit
def weighted_c(times, events, ranks, weights):
    # Descending follow-up. At ties, censors can be compared to deaths;
    # two deaths at the same instant cannot be ordered.
    tree = np.zeros(len(times) + 2)
    count = wins = pairs = 0.
    i = 0
    while i < len(times):
        end = i+1
        while end < len(times) and times[end] == times[i]:
            end += 1
        for kind in (0, 1):
            if kind == 1:
                for j in range(i, end):
                    if events[j] == 0 or weights[j] == 0:
                        continue
                    rank = ranks[j]
                    prefix = 0.
                    k = rank-1
                    while k:
                        prefix += tree[k]
                        k -= k & -k
                    inclusive = 0.
                    k = rank
                    while k:
                        inclusive += tree[k]
                        k -= k & -k
                    wins += weights[j] * (prefix + inclusive) / 2.
                    pairs += weights[j] * count
            for j in range(i, end):
                if events[j] != kind or weights[j] == 0:
                    continue
                count += weights[j]
                k = ranks[j]
                while k < len(tree):
                    tree[k] += weights[j]
                    k += k & -k
        i = end
    return wins/pairs if pairs else np.nan


def discrimination(d, columns, bootstrap=500):
    order = np.argsort(-d.time_yr.to_numpy())
    t = d.time_yr.to_numpy(dtype=float)[order]
    e = d.event.to_numpy(dtype=int)[order]
    ranks = [np.unique(d[c].to_numpy()[order], return_inverse=True)[1]+1 for c in columns]
    w = np.ones(len(t),dtype=np.int64)
    C = np.array([weighted_c(t,e,r,w) for r in ranks])
    if not np.isfinite(C).all():
        return {}
    for k,col in enumerate(columns):
        np.testing.assert_allclose(C[k],concordance_index(t,-d[col].to_numpy()[order],e),atol=1e-10)
    out = {c:{"C":float(v)} for c,v in zip(columns,C)}
    if bootstrap:
        rng = np.random.default_rng(20261121)
        sims = []
        for _ in range(bootstrap):
            w = np.bincount(rng.integers(0,len(t),len(t)),minlength=len(t))
            scores = np.array([weighted_c(t,e,r,w) for r in ranks])
            if np.isfinite(scores).all():
                sims.append(scores)
        sims = np.array(sims)
        for k,c in enumerate(columns):
            out[c]["CI95"] = np.percentile(sims[:,k],[2.5,97.5]).tolist()
            if k:
                out[c]["delta_clock_minus_comparator"] = float(C[0]-C[k])
                out[c]["delta_CI95"] = np.percentile(sims[:,0]-sims[:,k],[2.5,97.5]).tolist()
        out["bootstrap_valid"] = len(sims)
    return out


def summary(d):
    info = {"n":len(d),"deaths":int(d.event.sum())}
    if len(d) and d.event.sum() >= 2:
        info.update(discrimination(d,["full_lp","age"],0))
    return info


def censor_probabilities(t,e):
    levels,ix,n = np.unique(t,return_inverse=True,return_counts=True)
    events = np.bincount(ix,weights=e)
    at_risk = n[::-1].cumsum()[::-1]
    # Events are handled before withdrawals on the same date. G(T-) weights deaths.
    step = 1 - np.divide(n-events,at_risk-events,out=np.zeros(len(n)),where=at_risk>events)
    after = np.cumprod(step)
    before = np.r_[1.,after[:-1]][ix]
    return levels,after,before


def fixed_time(d,ref):
    result, calibration = {}, []
    t,e = d.time_yr.to_numpy(),d.event.to_numpy()
    levels,after,before = censor_probabilities(t,e)
    km = KaplanMeierFitter().fit(t,e)
    lps = {"clock":ref["risk_lp_slope"]*(d.full_lp.to_numpy()-ref["risk_lp_mean"]),
           "age_sex":d.age_sex_lp.to_numpy(),"ridge":d.ridge_lp.to_numpy()}
    for h in HORIZONS:
        case = (e==1)&(t<=h)
        control = t>h
        j = np.searchsorted(levels,h,side="right")-1
        g = float(after[j]) if j>=0 else 1.
        row = {"horizon":h,"n":len(d),"cases":int(case.sum()),"controls":int(control.sum()),"G":g}
        result[str(h)] = row
        if min(case.sum(),control.sum()) < 20 or g <= 0.05:
            row["supported"] = False
            continue
        row.update(supported=True, observed_km=float(1-km.predict(h)))
        w = np.zeros(len(t))
        w[case] = 1/before[case]
        w[control] = 1/g
        known = case|control
        risks = {}
        for key,lp in lps.items():
            risk = -np.expm1(-ref["baseline_hazard"][key][str(h)] * np.exp(lp))
            risks[key] = risk
            row[key] = dict(auc=float(roc_auc_score(case[known],lp[known], sample_weight=w[known])),
                            brier=float(np.mean(w*(case-risk)**2)), predicted_mean=float(risk.mean()))
        change = risks["clock"]-risks["age_sex"]
        for label,value in (("continuous_nri",np.sign(change)),("idi",change)):
            row[label+"_vs_age_sex"] = float(np.average(value[case],weights=w[case])-np.average(value[control],weights=w[control]))
        bins = pd.qcut(risks["clock"],5,labels=False,duplicates="drop")
        for b in np.unique(bins):
            mask = bins==b
            n_risk = int((mask&control).sum())
            observed = float(1-KaplanMeierFitter().fit(t[mask],e[mask]).predict(h)) if n_risk else None
            calibration.append(dict(horizon=h,quintile=int(b)+1,n=int(mask.sum()),events_by_horizon=int((mask&case).sum()),
                                    remaining_after=n_risk,predicted=float(risks["clock"][mask].mean()),observed_km=observed))
    return result,calibration


def evaluate(cohort, sensitivity=False, device="cuda"):
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    bundle = verify()
    ref = json.loads(REFERENCE.read_text(encoding="utf8"))
    assert ref["freeze_sha256"] == sha(TRAIN_RUN/"model_freeze.json") and ref["protocol_sha256"] == sha(ROOT/"验证方案.md")
    path = ROOT/("sensitivity_inputs" if sensitivity else "inputs")/(cohort+".csv.gz")
    raw = pd.read_csv(path,dtype={"ID":"string"})
    if not raw.ID.is_unique or raw.ID.isna().any():
        raise ValueError("External baseline must have one nonmissing ID per person")
    feats = bundle["features"]
    x = raw.loc[np.isfinite(raw.age)&(raw.age>=20)].copy().reset_index(drop=True)
    scores = score_frame(x,BUNDLE,device)
    for c in ("sex","time_yr","event","time_lower_yr","time_upper_yr","date_quality"):
        if c in x:
            scores[c] = x[c]
    for c in ("sex","event","time_yr"):
        if c not in scores:
            scores[c] = np.nan
    if not scores.event.dropna().isin([0,1]).all() or not scores.sex.dropna().isin([0,1]).all():
        raise ValueError("Invalid sex or event coding")
    if not scores.ID.equals(x.ID) or not scores.full_n_observed.equals(x.reindex(columns=feats).notna().sum(axis=1)):
        raise ValueError("Scores do not align with original IDs or observed features")
    z,m,age = prepare_features(x,bundle)
    scores["age_sex_lp"] = linear_score(scores[["age","sex"]].to_numpy(), ref["age_sex"])
    scores["ridge_lp"] = linear_score(np.c_[z,age],ref["ridge"])
    surv = scores[scores.full_lp.notna() & scores.time_yr.gt(0) & np.isfinite(scores.time_yr) & scores.event.notna()].copy()
    complete = surv.dropna(subset=["sex"]).copy()
    output = ROOT/"results"/cohort
    output.mkdir(parents=True,exist_ok=True)
    scores.to_csv(output/"predictions.csv.gz",index=False)
    cov = pd.DataFrame({"feature":feats,"baseline_n":m.sum(axis=0),
                        "survival_n":x.loc[surv.index].reindex(columns=feats).notna().sum().to_numpy()})
    cov.to_csv(output/"coverage.csv",index=False)
    measured = scores[scores.full_lp.notna()]
    metrics = dict(cohort=cohort, input_rows=len(raw),age_eligible=len(x),scored=len(measured),
                   source_sha256=sha(path), reference_sha256=sha(REFERENCE), device=torch.cuda.get_device_name() if device == "cuda" else "cpu",
                   feature_types=int((cov.baseline_n>0).sum()), median_features=float(measured.full_n_observed.median()) if len(measured) else None,
                   feature_q25_q75=measured.full_n_observed.quantile([.25,.75]).tolist() if len(measured) else [],
                   n=len(surv),deaths=int(surv.event.sum()),n_sex=len(complete),deaths_sex=int(complete.event.sum()))
    if len(surv) and surv.event.sum() >= 2:
        metrics["observed_followup_median_years"] = float(surv.time_yr.median())
        metrics["all"] = discrimination(surv,["full_lp","age","ridge_lp"])
        metrics["sensitivities"] = {"age_under80":summary(surv[surv.age<80]),
                                    "landmark_2y":summary(surv[surv.time_yr>2]),
                                    "at_least22":summary(surv[surv.full_n_observed>=22])}
        for name in ("time_lower_yr","time_upper_yr"):
            if name in surv:
                tmp = surv.copy()
                tmp["time_yr"] = np.where(tmp.event.eq(1)&tmp[name].notna(),tmp[name],tmp.time_yr)
                if (tmp.time_yr<0).any():
                    raise ValueError("Negative interval boundary")
                metrics["sensitivities"][name] = summary(tmp)
        metrics["coverage_groups"] = {}
        for low,high in [(1,4),(5,10),(11,21),(22,43)]:
            metrics["coverage_groups"][f"{low}-{high}"] = summary(surv[surv.full_n_observed.between(low,high)])
    if len(complete) and complete.event.sum() >= 2:
        metrics["same_sex_available"] = discrimination(complete,["full_lp","age_sex_lp","ridge_lp","age"])
        if complete.sex.nunique()==2 and complete.full_accel.std()>0 and complete.event.sum()>=20:
            design = complete[["time_yr","event","age","sex"]].copy()
            sd = float(complete.full_accel.std())
            design["accel_sd"] = (complete.full_accel-complete.full_accel.mean())/sd
            fit = CoxPHFitter().fit(design,"time_yr","event")
            metrics["accel_association"] = dict(HR=float(np.exp(fit.params_["accel_sd"])), SD_years=sd,
                    CI95=np.exp(fit.confidence_intervals_.loc["accel_sd"]).tolist())
        metrics["fixed_times"], rows = fixed_time(complete,ref)
        if rows:
            pd.DataFrame(rows).to_csv(output/"calibration.csv",index=False)
    metrics["predictions_sha256"] = sha(output / "predictions.csv.gz")
    save_json(output/"metrics.json",metrics)
    print(cohort,metrics["scored"],metrics["n"],metrics["deaths"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohorts",nargs="+")
    parser.add_argument("--sensitivity",action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    for cohort in args.cohorts:
        evaluate(cohort,args.sensitivity,args.device)
