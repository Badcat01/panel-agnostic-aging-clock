"""Compare frozen whole-body and matched organ risk scores on identical patients."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
MIMIC = RUN.parent
sys.path.insert(0, str(MIMIC))
from mimic_iv_panels import ORGAN_REFERENCE, TRAIN_RUN, association, ci95, compare_auc, digest, flatten, oof_comparison, verify
from mimic_iv_run import write_json

ENDPOINTS = {"liver": "new ALT >=200 U/L", "kidney": "first creatinine-defined AKI",
             "metabolic": "new level-2 laboratory hypoglycemia",
             "immune_heme": "first culture-confirmed bloodstream pathogen",
             "cardiovascular": "newly recorded ICU atrial fibrillation/flutter"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def evaluate(d, panel, ref, analysis):
    y = d.event.to_numpy(dtype=bool)
    if min(y.sum(), (~y).sum()) < 10:
        return {"analysis": analysis, "n": len(d), "events": int(y.sum()), "status": "insufficient_cases_or_controls"}
    age = (d.age.to_numpy() - 60)/10
    base = [age, age**2, d.sex.to_numpy(), d.full_accel.to_numpy()/ref["panels"]["full"]["accel_sd_years"]]
    if analysis == "adjust_baseline_analyte":
        available = d.baseline_value.to_numpy(dtype=float)
        if not (np.isfinite(available) & (available > 0)).all():
            raise ValueError("Invalid baseline analyte values")
        base.append(np.log(available))
    organ = d[f"{panel}_accel"].to_numpy()/ref["panels"][panel]["accel_sd_years"]
    base = np.column_stack(base)
    pair = d[["full_lp", f"{panel}_lp"]].to_numpy()
    auc, cov, _ = compare_auc(pair, y)
    for j in range(2):
        np.testing.assert_allclose(auc[j], roc_auc_score(y, pair[:,j]), atol=1e-12)
    return {"analysis": analysis, "n":len(d), "events":int(y.sum()), "recorded_event_fraction":float(y.mean()),
            "status":"evaluated", "full_raw_auc":float(auc[0]), "organ_raw_auc":float(auc[1]),
            "full_raw_auc_ci95":ci95(auc[0], cov[0,0], (0,1)),
            "organ_raw_auc_ci95":ci95(auc[1], cov[1,1], (0,1)),
            **oof_comparison(base, organ, y, seed=20261121),
            **association(base, organ, y)}


def assess(panel, scores, ref):
    inp = RUN / "inputs" / f"{panel}.csv.gz"
    audit = read_json(RUN / "audit" / f"{panel}.json")
    f = pd.read_csv(inp, dtype={"ID":str})
    if audit.get("input_sha256") and digest(inp) != audit["input_sha256"]:
        raise ValueError(f"Input audit hash mismatch for {panel}")
    if f.ID.isna().any() or not f.ID.is_unique or not f.event.isin([0,1]).all():
        raise ValueError(f"Invalid participant ID or outcome in {panel}")
    if not f.time_days.gt(0).all() or not f.time_days.le(7.0000001).all():
        raise ValueError(f"Invalid landmark time in {panel}")
    if not f.event.eq(f.end_status.eq("event")).all():
        raise ValueError(f"End status disagrees with the event in {panel}")
    columns = ["ID","age","sex","full_lp","full_accel",f"{panel}_lp",f"{panel}_accel",f"{panel}_n_observed"]
    d = f.merge(scores[columns], on="ID", validate="one_to_one")
    if len(d) != len(f) or not d.sex.isin([0,1]).all() or not np.isfinite(d[columns[1:]]).all().all():
        raise ValueError(f"Missing or unscored participant in the {panel} outcomes")
    if not d[f"{panel}_n_observed"].gt(0).all():
        raise ValueError(f"Missing relevant features for {panel}")
    out = evaluate(d, panel, ref, "primary")
    out["end_status_counts"] = {str(k): int(v) for k,v in d.end_status.value_counts().items()}
    out["median_panel_markers"] = float(d[f"{panel}_n_observed"].median())
    out["n_with_one_panel_marker"] = int(d[f"{panel}_n_observed"].eq(1).sum())
    analyses = []
    monitor = [c for c in ["post_observed","post_landmark_tested","postlab_observed"] if c in d]
    if monitor:
        yes = d[monitor[0]].astype("string").str.casefold().isin(["true","1","1.0"])
        out["post_observed_n"] = int(yes.sum())
        analyses.append(evaluate(d.loc[yes], panel, ref, "observed_again_after_landmark"))
    if panel == "kidney" and "baseline_creatinine_mg_dl" in d:
        d = d.assign(baseline_value=d.baseline_creatinine_mg_dl)
    if "baseline_value" in d:
        analyses.append(evaluate(d, panel, ref, "adjust_baseline_analyte"))
    if "confirmed_event" in d:
        confirmed = d.assign(event=d.confirmed_event)
        analyses.append(evaluate(confirmed, panel, ref, "confirmed_on_two_days"))
    if panel == "cardiovascular":
        shorter = d.assign(event=d.event.eq(1) & d.time_days.le(2))
        analyses.append(evaluate(shorter, panel, ref, "documented_within_48h"))
    meta = {"panel":panel, "endpoint":ENDPOINTS[panel], "input_sha256":digest(inp)}
    return {**meta, **out}, [{**meta, **x} for x in analyses]


def main():
    verify()
    old = MIMIC / "formal_v3_1"
    pred = old / "predictions.csv.gz"
    frozen = read_json(old / "audit.json")
    model_hash = digest(TRAIN_RUN / "model_freeze.json")
    ref = read_json(ORGAN_REFERENCE)
    if digest(pred) != frozen["predictions_sha256"] or model_hash != ref["model_freeze_sha256"]:
        raise ValueError("Prior score or reference digest differs")
    scores = pd.read_csv(pred, dtype={"ID":str})
    assert scores.ID.is_unique and len(scores) == frozen["n_scored"]
    results, checks = [], []
    with threadpool_limits(limits=2):
        for panel in ENDPOINTS:
            row, sensitivity = assess(panel, scores, ref)
            results.append(row)
            checks.extend(sensitivity)
            print(f"{panel}: {row['n']:,} participants, {row['events']:,} events, {row['status']}", flush=True)
    p = [row.get("incremental_auc_oof_p",1) for row in results]
    for row, value in zip(results, multipletests(p, method="fdr_bh")[1]):
        if row["status"] == "evaluated":
            row["incremental_auc_q_BH_five"] = float(value)
    output = RUN / "results"
    output.mkdir(exist_ok=True)
    save = lambda rows: [flatten({key: value for key, value in row.items() if key != "end_status_counts"}) for row in rows]
    pd.DataFrame(save(results)).to_csv(output/"incident_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(save(checks)).to_csv(output/"incident_sensitivity.csv",index=False,encoding="utf-8-sig")
    write_json(output/"incident_evaluation.json",{"n_parent_cohort": len(scores), "days_after_24h":7,
                 "frozen_predictions_sha256": frozen["predictions_sha256"], "model_freeze_sha256": model_hash,
                 "primary":results, "sensitivity":checks})


if __name__ == "__main__":
    main()
