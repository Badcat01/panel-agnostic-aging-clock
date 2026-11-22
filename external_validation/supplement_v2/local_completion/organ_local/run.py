"""Five-score comparisons on existing frozen NHANES III and MIMIC outcomes."""

import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from study_paths import MIMIC_ARCHIVE
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SUP = HERE.parents[1]
EXT = SUP.parent
MIM = EXT / "mimic_iv"
for folder in (EXT / "scripts", MIM, SUP / "technical", EXT / "nhanes3_organ",
               SUP / "biomarkers"):
    sys.path.insert(0, str(folder))
from build_nhanes3 import load_layout, read_fwf
from mimic_iv_panels import compare_auc
from mimic_readmit import CODES
from mortality import cohort, outcomes
from organ_references import matrix
from shared import formal, read, sha, verify

PANELS = list(CODES)
SCORES = [f"{p}_accel" for p in PANELS]
SEED = 20261121


def fit_auc(base, organ, outcome):
    """Fit all six models on identical training and held-out people."""
    fold_stats = []
    for train, val in StratifiedKFold(5, shuffle=True, random_state=SEED).split(base, outcome):
        models = [base, *(np.column_stack([base, col]) for col in organ.T)]
        predictions = []
        for features in models:
            center = features[train].mean(axis=0)
            scale = features[train].std(axis=0)
            scale[scale == 0] = 1
            fit = LogisticRegression(penalty=None, max_iter=1000, tol=1e-8)
            fit.fit((features[train] - center) / scale, outcome[train])
            if fit.n_iter_.max() >= fit.max_iter:
                raise RuntimeError("Logistic regression did not converge")
            predictions.append(fit.predict_proba((features[val] - center) / scale)[:, 1])
        risk = np.column_stack(predictions)
        stats = compare_auc(risk, outcome[val])
        independent = [roc_auc_score(outcome[val], series) for series in risk.T]
        np.testing.assert_allclose(stats[0], independent, atol=1e-12)
        fold_stats.append(stats)
    weights = np.array([pair_count for _, _, pair_count in fold_stats], dtype=float)
    weights /= weights.sum()
    auc = sum(w * a for w, (a, _, _) in zip(weights, fold_stats))
    cov = sum(w**2 * v for w, (_, v, _) in zip(weights, fold_stats))
    return auc, cov


def comparisons(d, event, cohort_name, endpoint, target, variant, anchor=None):
    required = [*SCORES, "age", "sex", "full_accel"]
    good = d[required].notna().all(axis=1)
    if anchor:
        good &= pd.to_numeric(d[anchor]).gt(0)
    cases = event.loc[good].to_numpy(dtype=bool)
    n, nevent = len(cases), int(cases.sum())
    main = dict(cohort=cohort_name, endpoint=endpoint, matched=target, analysis=variant,
                original_n=len(d), original_events=int(event.sum()), n=n, events=nevent)
    if min(nevent, n - nevent) < 10:
        return [{**main, "status": "insufficient_cases_or_controls", "comparator": "baseline"}]
    use = d.loc[good]
    age = (use.age - 60) / 10
    base = [age, age**2, use.sex, use.full_accel]
    if anchor:
        base.append(np.log(pd.to_numeric(use[anchor])))
    auc, cov = fit_auc(np.column_stack(base), use[SCORES].to_numpy(), cases)
    labels = ["baseline", *PANELS]
    i = labels.index(target)
    rows = []
    for j, other in enumerate(labels):
        if i == j:
            continue
        delta = auc[i] - auc[j]
        se = np.sqrt(max(0, cov[i, i] + cov[j, j] - 2 * cov[i, j]))
        stat = delta / se if se else 0
        rows.append({**main, "status": "evaluated", "comparator": other, "target_auc": auc[i],
                     "comparator_auc": auc[j], "delta": delta, "lo": delta - 1.96 * se,
                     "hi": delta + 1.96 * se, "p_gt": norm.sf(stat), "p_two": 2 * norm.sf(abs(stat))})
    return rows


def mimic_incident(scores):
    rows = []
    folder = MIM / "incident_v3_1"
    for name in PANELS:
        inp = folder / "inputs" / (name + ".csv.gz")
        assert sha(inp) == read(folder / "audit" / (name + ".json"))["input_sha256"]
        source = pd.read_csv(inp, dtype={"ID": str}).set_index("ID", verify_integrity=True)
        assert source.event.isin([0, 1]).all() and source.time_days.between(0, 7.0000001, inclusive="right").all()
        assert source.event.eq(source.end_status.eq("event")).all() and source.index.isin(scores.index).all()
        d = source.join(scores[["age", "sex", "full_accel", *SCORES]], validate="one_to_one")
        rows.extend(comparisons(d, d.event, "MIMIC", name, name, "7day"))
        field = "postlab_observed" if name == "kidney" else "post_observed"
        monitored = d.loc[d[field].astype("string").str.casefold().isin(["true", "1", "1.0"])]
        assert int(monitored.event.sum()) == int(d.event.sum())
        rows.extend(comparisons(monitored, monitored.event, "MIMIC", name, name, "7day_observed"))
        if "baseline_value" in d:
            rows.extend(comparisons(d, d.event, "MIMIC", name, name, "7day_plus_analyte", "baseline_value"))
    return rows


def mimic_readmission(scores):
    src = MIM / "local_3_1" / "hosp"
    archive = MIMIC_ARCHIVE
    prefix = "mimic-iv-3.1/"
    member = "hosp/diagnoses_icd.csv.gz"
    with ZipFile(archive) as z:
        checksums = {path.lstrip("*"): digest for digest, path in
                     (line.split(maxsplit=1) for line in z.read(prefix+"SHA256SUMS.txt").decode().splitlines())}
        for filename in ("admissions.csv.gz", "patients.csv.gz"):
            assert sha(src / filename) == checksums["hosp/" + filename]
        content = z.read(prefix + member)
    assert hashlib.sha256(content).hexdigest() == checksums[member]
    dx = pd.read_csv(io.BytesIO(content), compression="gzip", dtype={"icd_code": "string"})
    a = pd.read_csv(src / "admissions.csv.gz", parse_dates=["admittime", "dischtime"],
                    usecols=["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"])
    patients = pd.read_csv(src / "patients.csv.gz",
                           usecols=["subject_id", "anchor_year", "anchor_year_group"])
    idx = pd.read_csv(MIM / "formal_v3_1" / "landmark_index.csv.gz",
                      usecols=["subject_id", "hadm_id", "landmark"], parse_dates=["landmark"])
    idx = idx[idx.subject_id.astype(str).isin(scores.index)].merge(
        a, on=["subject_id", "hadm_id"], validate="one_to_one").merge(
        patients, on="subject_id", validate="one_to_one")
    idx["horizon"] = idx.landmark + pd.Timedelta(days=365)
    last = idx.anchor_year_group.str.extract(r"(\d{4})\s*$")[0].astype(int)
    covered = last + idx.horizon.dt.year - idx.anchor_year <= 2021
    risk = idx.loc[covered & idx.hospital_expire_flag.eq(0) & idx.dischtime.lt(idx.horizon)].copy()
    future = a.merge(risk[["subject_id", "hadm_id", "dischtime", "horizon"]],
                     on="subject_id", suffixes=("", "_index"))
    future = future.loc[future.hadm_id.ne(future.hadm_id_index)
                        & future.admittime.gt(future.dischtime_index)
                        & future.admittime.le(future.horizon)]
    assert risk.subject_id.is_unique and future.hadm_id.is_unique
    assert len(risk) == read(SUP / "technical" / "readmission_design.json")["survived_discharge_and_within_year"]
    obs = dx.loc[dx.hadm_id.isin(set(future.hadm_id) | set(risk.hadm_id))]
    code = obs.icd_code.str.upper().str.strip()
    returned = set(future.subject_id)
    rows = []
    prior_flow = pd.read_csv(SUP / "technical" / "readmission_flow.csv")
    for panel, (icd9, icd10) in CODES.items():
        capture = obs.loc[(obs.icd_version.eq(9) & code.str.match(icd9))
                          | (obs.icd_version.eq(10) & code.str.match(icd10))]
        valid = risk.loc[~risk.hadm_id.isin(capture.hadm_id), "subject_id"]
        cases = set(future.loc[future.hadm_id.isin(capture.hadm_id), "subject_id"])
        prior = prior_flow.query("panel == @panel and code_position == 'any'").iloc[0]
        assert len(valid) == prior.eligible_before_scores and valid.isin(cases).sum() == prior.any_target
        d = scores.loc[valid.astype(str)]
        positive = pd.Series(d.index.isin(map(str, cases)), index=d.index)
        rows.extend(comparisons(d, positive, "MIMIC", panel, panel, "1year_any_diagnosis"))
        inside = d.loc[d.index.isin(map(str, returned))]
        rows.extend(comparisons(inside, positive.loc[inside.index], "MIMIC", panel, panel,
                                "1year_any_diagnosis_only_returned"))
    return rows


def nhanes3():
    scores, metadata = cohort()
    d = scores.loc[~scores.unknown]
    assert not (d.event.eq(0) & d.time_yr.lt(20)).any()
    rows = []
    matches = {"renal": "kidney", "diabetes": "metabolic", "influenza_pneumonia": "immune_heme",
               "cardiovascular": "cardiovascular"}
    for key, match in matches.items():
        y = outcomes(d)[key] & d.time_yr.le(20)
        rows.extend(comparisons(d, y, "NHANES III", key, match, "20year_or_ultrasound"))
    path = EXT / "nhanes3_organ" / "HGUHS.xpt"
    assert sha(path) == read(path.with_name("liver_ultrasound.json"))["source"]["sha256"]
    u = pd.read_sas(path, encoding="latin1").set_index("SEQN", verify_integrity=True)
    liver = scores.join(u[["GUPHSPFR"]], on="ID", validate="one_to_one")
    liver = liver.loc[liver.GUPHSPFR.isin([1, 2]) & liver.age.between(20, 74)]
    assert len(liver) == 12475 and liver.GUPHSPFR.eq(2).sum() == 2892
    rows.extend(comparisons(liver, liver.GUPHSPFR.eq(2), "NHANES III", "ultrasound_steatosis",
                            "liver", "20year_or_ultrasound"))
    return rows, scores, metadata


def nhanes_assays(scores):
    source = EXT / "nhanes3_source"
    original = read(EXT / "audit" / "nhanes3.json")["sources"]
    for filename in ["lab.dat", "lab.sas", "exam.dat", "exam.sas"]:
        assert sha(source / filename) == original[filename]["sha256"]
    lab_fields = ["SEQN", "PHPFAST", "I1P", "SGP", "FRP", "FBP"]
    exam_fields = ["SEQN", "WTPFEX6", "SDPPSU6", "SDPSTRA6", "SDPPHASE"]
    lab, _ = read_fwf("lab.dat", load_layout("lab.sas", lab_fields), {1971, 1977}, 29314)
    exam, _ = read_fwf("exam.dat", load_layout("exam.sas", exam_fields), {6229, 6233}, 31311)
    exam = exam.rename(columns={"WTPFEX6":"mec_weight", "SDPPSU6":"psu",
                                "SDPSTRA6":"strata", "SDPPHASE":"cycle"})
    d = scores.set_index("ID", verify_integrity=True).join(lab).join(exam)
    d["male"] = d.sex
    fasting = d.PHPFAST.between(8, 24, inclusive="left")
    d["I1P"] = d.I1P.where(fasting & d.I1P.gt(0))
    d["HOMA_IR"] = (d.I1P * d.SGP / 405).where(d.SGP.gt(0))
    for field in ("FRP", "FBP"):
        d[field] = d[field].where(d[field].gt(0))
    assay_rows = []
    pooled = {"FRP": "immune_heme", "FBP": "immune_heme", "I1P": "metabolic",
              "HOMA_IR": "metabolic"}
    for marker, panel in pooled.items():
        assay_rows.extend(matrix(d, marker, [panel], "NHANES III", "mec_weight", "log", "primary"))
    tab = pd.DataFrame(assay_rows)
    diag = tab.matched
    tab.loc[diag, "q_dominance"] = multipletests(tab.loc[diag, "p_diagonal"], method="fdr_bh")[1]
    tab["q_association"] = multipletests(tab.p, method="fdr_bh")[1]
    return tab


def summary(tab):
    units = []
    for keys, data in tab.groupby(["cohort", "analysis", "endpoint"], sort=False):
        initial = data.loc[data.comparator.eq("baseline")].iloc[0]
        opponents = data.loc[data.comparator.ne("baseline")]
        evaluated = initial.status == "evaluated"
        unit = {x: initial[x] for x in ("cohort", "analysis", "endpoint", "matched", "status",
                                        "original_n", "original_events", "n", "events")}
        unit["p_dominance"] = opponents.p_gt.max() if evaluated else 1.0
        unit["p_increment"] = initial.p_two if evaluated else 1.0
        if evaluated:
            strongest = opponents.loc[opponents.delta.idxmin()]
            unit.update(delta_increment=initial.delta, increment_lo=initial.lo, increment_hi=initial.hi,
                        strongest_other=strongest.comparator, dominance_delta=strongest.delta,
                        dominance_lo=strongest.lo, dominance_hi=strongest.hi)
        units.append(unit)
    out = pd.DataFrame(units)
    for column in ("increment", "dominance"):
        out["q_" + column] = out.groupby(["cohort", "analysis"])["p_" + column].transform(
            lambda p: multipletests(p, method="fdr_bh")[1])
    return out


def main():
    verify()
    _, mimic = formal("mimic")
    rows = mimic_incident(mimic)
    print("Completed acute MIMIC contrasts", flush=True)
    rows.extend(mimic_readmission(mimic))
    print("Completed MIMIC readmission contrasts", flush=True)
    result, nhanes, meta = nhanes3()
    rows.extend(result)
    ref = nhanes_assays(nhanes)
    comparisons_table = pd.DataFrame(rows)
    aggregate = summary(comparisons_table)
    for name, frame in [("auc_pairs.csv", comparisons_table), ("auc_summary.csv", aggregate),
                        ("nhanes3_assay_matrix.csv", ref)]:
        dest = HERE / name
        tmp = dest.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(dest)
    audit = {
        "nhanes3_freeze": meta,
        "mimic_frozen_predictions_sha256": sha(MIM / "formal_v3_1" / "predictions.csv.gz"),
        "mimic_cohort_n": len(mimic), "comparisons": len(comparisons_table),
        "tests": len(aggregate), "evaluated_tests": int(aggregate.status.eq("evaluated").sum()),
        "nhanes3_ref_rows": len(ref), "independent_auc_check": "sklearn in every fold"
    }
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
