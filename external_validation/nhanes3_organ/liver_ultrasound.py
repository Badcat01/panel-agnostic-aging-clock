"""Associate five frozen NHANES III scores with prevalent ultrasound steatosis."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent
GROUPS = ("liver", "kidney", "metabolic", "immune_heme", "cardiovascular")
BASE_COLUMNS = ["age_10", "age_10_sq", "sex", "full_accel"]
XPT_URL = "https://wwwn.cdc.gov/nchs/data/nhanes3/34a/HGUHS.xpt"
CODEBOOK = "https://wwwn.cdc.gov/nchs/data/nhanes3/34a/HGUHS.htm"


def digest(path):
    with path.open("rb") as stream:
        checksum = sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def paired_oof_auc(base, enhanced, outcome):
    y = outcome.to_numpy(dtype=int)
    pred_base, pred_enhanced = np.zeros(len(y)), np.zeros(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=81326)
    for train, test in folds.split(base, y):
        for predictors, destination in ((base, pred_base), (enhanced, pred_enhanced)):
            model = LogisticRegression(penalty=None, max_iter=500)
            model.fit(predictors.iloc[train], y[train])
            destination[test] = model.predict_proba(predictors.iloc[test])[:, 1]
    auc_base = roc_auc_score(y, pred_base)
    auc_enhanced = roc_auc_score(y, pred_enhanced)
    rng = np.random.default_rng(81327)
    differences = []
    for _ in range(500):
        idx = rng.integers(0, len(y), size=len(y))
        differences.append(roc_auc_score(y[idx], pred_enhanced[idx]) - roc_auc_score(y[idx], pred_base[idx]))
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "auc_base_oof": float(auc_base),
        "auc_enhanced_oof": float(auc_enhanced),
        "delta_auc_oof": float(auc_enhanced - auc_base),
        "delta_auc_ci95_lower": float(lower),
        "delta_auc_ci95_upper": float(upper),
        "brier_base_oof": float(brier_score_loss(y, pred_base)),
        "brier_enhanced_oof": float(brier_score_loss(y, pred_enhanced)),
    }


def fit_comparison(frame, group, include_bmi=False):
    covariates = BASE_COLUMNS + (["bmi"] if include_bmi else [])
    score = group + "_accel"
    frame = frame.loc[:, [*covariates, score, "case"]].dropna().copy()
    sd = frame[score].std()
    frame["score_sd"] = (frame[score] - frame[score].mean()) / sd
    outcome = frame.case
    base = frame[covariates]
    full = frame[[*covariates, "score_sd"]]
    reference = sm.GLM(outcome, sm.add_constant(base), family=sm.families.Binomial()).fit()
    combined = sm.GLM(outcome, sm.add_constant(full), family=sm.families.Binomial()).fit()
    lrt = 2 * (combined.llf - reference.llf)
    lower, upper = np.exp(combined.conf_int().loc["score_sd"])
    return {
        "panel": group,
        "adjusted_for_bmi": include_bmi,
        "n": len(frame),
        "cases": int(outcome.sum()),
        "score_SD_years": float(sd),
        "OR_per_SD": float(np.exp(combined.params["score_sd"])),
        "OR_ci95_lower": float(lower),
        "OR_ci95_upper": float(upper),
        "OR_Wald_p": float(combined.pvalues["score_sd"]),
        "LRT": float(lrt),
        "LRT_p": float(chi2.sf(lrt, 1)),
        **paired_oof_auc(base, full, outcome),
    }


def main():
    prediction_path = EXTERNAL / "results" / "nhanes3" / "predictions.csv.gz"
    input_path = EXTERNAL / "inputs" / "nhanes3.csv.gz"
    image_path = HERE / "HGUHS.xpt"
    predictions = pd.read_csv(prediction_path, dtype={"ID": int})
    bmi = pd.read_csv(input_path, usecols=["ID", "bmi"], dtype={"ID": int})
    selected = predictions.loc[predictions[[f"{g}_accel" for g in GROUPS]].notna().all(axis=1)]
    if len(selected) != 14_807 or selected.ID.duplicated().any() or not selected.full_accel.notna().all():
        raise ValueError("The linked frozen cohort must contain 14,807 uniquely scored people in all panels")

    ultrasound = pd.read_sas(image_path, format="xport", encoding="latin1")
    if len(ultrasound) != 14_797 or ultrasound.SEQN.duplicated().any() or not ultrasound.SEQN.notna().all():
        raise ValueError("HGUHS dimensions or identifiers differ from the CDC codebook")
    expected_codes = {1: 10_703, 2: 3_153, 7: 127, 8: 814}
    if ultrasound.GUPHSPFR.value_counts().to_dict() != expected_codes:
        raise ValueError("HGUHS recoded findings differ from CDC counts")

    linked = selected.merge(ultrasound[["SEQN", "GUPHSPFR"]].rename(columns={"SEQN": "ID"}),
                            on="ID", how="left", validate="one_to_one")
    flow = {
        "all_five_scores_n": len(selected),
        "no_ultrasound_row": int(linked.GUPHSPFR.isna().sum()),
        "image_ungradable": int(linked.GUPHSPFR.eq(7).sum()),
        "image_missing": int(linked.GUPHSPFR.eq(8).sum()),
        "valid_ultrasound_before_age_restriction": int(linked.GUPHSPFR.isin([1, 2]).sum()),
        "valid_ultrasound_outside_exam_age_20_74": int((linked.GUPHSPFR.isin([1, 2]) & ~linked.age.between(20, 74)).sum()),
    }
    data = linked.loc[linked.GUPHSPFR.isin([1, 2]) & linked.age.between(20, 74)].copy()
    if not data.sex.isin([0, 1]).all():
        raise ValueError("Invalid sex coding in the ultrasound analytic subset")
    data = data.merge(bmi, on="ID", how="left", validate="one_to_one")
    data["case"] = data.GUPHSPFR.eq(2).astype(int)
    data["age_10"] = (data.age - 50) / 10
    data["age_10_sq"] = data.age_10**2
    records = [fit_comparison(data, g) for g in GROUPS]
    records.append(fit_comparison(data, "liver", include_bmi=True))
    pd.DataFrame(records).to_csv(HERE / "liver_ultrasound.csv", index=False, encoding="utf-8-sig")
    output = {
        "outcome": "Prevalent moderate-severe ultrasound hepatic steatosis at baseline",
        "definition": {"case": "GUPHSPFR=2", "control": "GUPHSPFR=1", "excluded": "7/8/absent"},
        "source": {"url": XPT_URL, "codebook": CODEBOOK, "sha256": digest(image_path)},
        "frozen_predictions_sha256": digest(prediction_path),
        "analysis_input_sha256": digest(input_path),
        "flow": flow,
        "n": len(data),
        "cases": int(data.case.sum()),
        "age_summary": {"min": float(data.age.min()), "max": float(data.age.max()),
                        "median": float(data.age.median())},
        "comparisons": records,
        "notes": [
            "The imaging outcome is independent of the clinical inputs. Original scans were read retrospectively and blind to other participant data.",
            "This is a cross-sectional association; the replicate exam was conducted at the same survey site and during the same survey period.",
            "Base predictors are age, age squared, sex, and frozen full aging acceleration. Enhanced model adds one frozen panel acceleration.",
            "Liver is the primary panel for this outcome; comparisons for the other groups contextualize specificity. P values are nominal.",
            "Outcome-model coefficients are fit in NHANES III. OOF uses five stratified folds. Paired bootstrap resamples fixed OOF predictions rather than retraining models.",
            "Unweighted analysis of the all-panel analytic subset. Cohort estimates can differ from representative population estimates.",
        ],
    }
    (HERE / "liver_ultrasound.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(f"Matched imaging outcomes: {len(data):,}; moderate/severe steatosis: {int(data.case.sum()):,}")
    print(pd.DataFrame(records)[["panel", "adjusted_for_bmi", "n", "cases", "OR_per_SD", "LRT_p", "delta_auc_oof"]].to_string(index=False))


if __name__ == "__main__":
    main()
