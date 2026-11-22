"""Reconstruct three NHANES risk maps; validate supported horizons in the same subjects."""
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from shared import Ensemble, HERE, REFERENCE, TRAIN_RUN, formal, get_cohorts, h0, prepare_features, read
from reference import linear_score
from calibration import calibrate
from validate_external import censor_probabilities


def project(raw, clock, net, ref):
    x, _, age = prepare_features(raw, net.bundle)
    sex = raw["sex"] if "sex" in raw else raw.male
    return {
        "clock": (clock - ref["risk_lp_mean"]) * ref["risk_lp_slope"],
        "age_sex": linear_score(np.column_stack([raw.age, sex]), ref["age_sex"]),
        "ridge": linear_score(np.column_stack([x, age]), ref["ridge"]),
    }


def main():
    net = Ensemble()
    dev, test, _ = get_cohorts()
    ref = read(REFERENCE)
    train_lp = project(dev, net.score(dev), net, ref)
    base = {(model, h): h0(dev.time_yr.to_numpy(), dev.event.to_numpy(), v, h)
            for model, v in train_lp.items() for h in [1, 3, 5, 10]}
    for model in train_lp:
        for h in [3, 5, 10]:
            np.testing.assert_allclose(base[model, h], ref["baseline_hazard"][model][str(h)], rtol=1e-6)
    metrics, decision, sensitivity, reproduction = [], [], [], []
    for co in ["temporal", "nhanes3", "hrs", "charls", "elsa", "share", "klosa", "mimic"]:
        if co == "temporal":
            raw = test.rename(columns={"male": "sex"}).set_index("SEQN")
            p = pd.read_csv(TRAIN_RUN / "temporal_predictions.csv.gz").rename(
                columns={"male":"sex", "ridge_cox_lp":"ridge_lp"}).set_index("SEQN", verify_integrity=True)
        else:
            raw, p = formal(co)
        p = p.dropna(subset=["full_lp", "age_sex_lp", "ridge_lp", "sex", "time_yr", "event"])
        p = p.loc[p.time_yr.gt(0)]
        raw = raw.loc[p.index]
        t, e = p.time_yr.to_numpy(), p.event.to_numpy(int)
        lp = project(raw, p.full_lp.to_numpy(), net, ref)
        for name in ["age_sex", "ridge"]:
            saved = p[name + "_lp"].to_numpy()
            reproduction.append(dict(cohort=co, model=name, max_abs_difference=np.abs(lp[name]-saved).max(),
                                     correlation=np.corrcoef(lp[name], saved)[0,1]))
        levels, after, before = censor_probabilities(t,e)
        for h in ([1] if co == "mimic" else [3,5,10]):
            case = (e == 1) & (t <= h)
            fixed = np.all((t >= h) | (e == 1))
            ctrl = (~case) & ((t >= h) if fixed else (t > h))
            j = np.searchsorted(levels, h, side="right") - 1
            g = after[j] if j >= 0 else 1.
            support = min(case.sum(), ctrl.sum()) >= 20 and (fixed or g > .05)
            w = np.zeros(len(t))
            w[case] = 1/before[case]
            if support:
                w[ctrl] = 1. if fixed else 1/g
            prob = {"treat_none": np.zeros(len(t)), "treat_all": np.ones(len(t))}
            for m in lp:
                row = dict(cohort=co, horizon=h, model=m, n=len(t), cases=int(case.sum()),
                           controls=int(ctrl.sum()), G_at_h=g, supported=bool(support))
                if support:
                    row.update(calibrate(t,e,lp[m],base[m,h],h))
                    prob[m] = -np.expm1(-base[m,h]*np.exp(lp[m]))
                    take = case | ctrl
                    row["cumulative_dynamic_AUC"] = roc_auc_score(case[take],lp[m][take],sample_weight=w[take])
                metrics.append(row)
            if support:
                for name, q in prob.items():
                    for threshold in [.01,.025,.05,.075,.1,.15,.2,.3]:
                        pos = q >= threshold
                        net_benefit = (w[pos & case].sum()-w[pos & ctrl].sum()*threshold/(1-threshold))/len(t)
                        decision.append(dict(cohort=co,horizon=h,model=name,threshold=threshold,net_benefit=net_benefit))
        # Use identical eligible people per contrast. A sole CRP value may leave no valid score.
        old = p.full_lp.to_numpy(copy=True)
        for label in ["original", "age_cap_80", "age_cap_85", "mask_crp", "survived_observed_2y"]:
            score = old.copy()
            eligible = np.ones(len(t),bool)
            changed = np.zeros(len(t),bool)
            alter = raw
            if label.startswith("age_cap"):
                cap=int(label[-2:])
                changed=raw.age.gt(cap).to_numpy()
                alter=raw.loc[changed].assign(age=raw.loc[changed,"age"].clip(upper=cap))
            if label=="mask_crp" and "crp" in raw:
                changed=raw.crp.notna().to_numpy()
                alter=raw.loc[changed].drop(columns="crp")
            if changed.any():
                score[changed]=net.score(alter)
            if label=="survived_observed_2y":
                eligible=t>2
            eligible &= np.isfinite(score)
            row=dict(cohort=co,analysis=label,n=int(eligible.sum()),events=int(e[eligible].sum()),
                     individuals_affected=int(changed.sum()))
            if e[eligible].sum()>5 and np.unique(t[eligible]).size>1:
                row["C"]=concordance_index(t[eligible],-score[eligible],e[eligible])
                row["unchanged_C_same_people"]=concordance_index(t[eligible],-old[eligible],e[eligible])
            sensitivity.append(row)
        for filename, records in [("horizons",metrics),("horizon_decisions",decision),
                                  ("sensitivity_same_people",sensitivity),("reference_lp_reproduction",reproduction)]:
            pd.DataFrame(records).to_csv(HERE/(filename+".csv"),index=False)
        print(co, len(t), flush=True)
    # Supply same-horizon comparator curves to the original cohort plots, using verified risk maps.
    from survival_standard import plots
    dca=pd.DataFrame(decision)
    for co,g in dca.groupby("cohort"):
        g[g.horizon.eq(1 if co=="mimic" else 3)].to_csv(HERE/"survival"/f"{co}_DCA.csv",index=False)
    plots()


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
