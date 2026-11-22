"""Exploratory, within-cohort contrasts of the frozen organ score and its inputs."""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from evaluate_incident import ENDPOINTS, MIMIC, ORGAN_REFERENCE, RUN, digest, read_json, verify
from mimic_iv_panels import ci95, compare_auc, prepare_features


def load_inputs():
    formal = MIMIC / "formal_v3_1"
    audit = read_json(formal / "audit.json")
    sources = {
        "landmark_index": audit["index_sha256"],
        "lab_panel": audit["lab_audit"]["lab_panel_sha256"],
        "vitals": audit["vitals_sha256"],
        "predictions": audit["predictions_sha256"],
    }
    for name, expected in sources.items():
        if digest(formal / f"{name}.csv.gz") != expected:
            raise ValueError(f"Changed formal input {name}")
    index = pd.read_csv(formal / "landmark_index.csv.gz", usecols=["subject_id", "hadm_id", "age"])
    labs = pd.read_csv(formal / "lab_panel.csv.gz")
    vitals = pd.read_csv(formal / "vitals.csv.gz", usecols=["subject_id","hadm_id","sbp","dbp","pulse","bmi"])
    raw = index.merge(labs, on="hadm_id", validate="one_to_one")
    raw = raw.merge(vitals, on=["subject_id", "hadm_id"], how="left", validate="one_to_one").set_index("subject_id")
    pred = pd.read_csv(formal / "predictions.csv.gz").set_index("ID", verify_integrity=True)
    if not raw.index.is_unique or set(raw.index) != set(pred.index):
        raise ValueError("The scoring inputs do not reproduce the frozen cohort")
    bundle = verify()
    if audit["model_freeze_sha256"] != read_json(ORGAN_REFERENCE)["model_freeze_sha256"]:
        raise ValueError("Inconsistent scoring reference")
    raw = raw.loc[pred.index]
    np.testing.assert_array_equal(raw.age, pred.age)
    values, mask, _ = prepare_features(raw, bundle)
    return bundle, pred, values, mask


def held_out_predictions(base, organ, mask, values, y):
    # Same fixed ridge penalty for every model. Only the shared covariates and the
    # frozen organ score are rescaled locally. Marker units are fixed by NHANES.
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261121)
    for train, test in split.split(base, y):
        design = np.column_stack([base, organ])
        scale = design[train].std(axis=0)
        scale[scale == 0] = 1.0
        design = (design - design[train].mean(axis=0)) / scale
        b, score = design[:, :-1], design[:, -1]
        knots = np.quantile(score[train], [.25, .75])
        hinge = np.maximum(score[:, None] - knots, 0)
        choices = {
            "base": b,
            "score": design,
            "flex_score": np.column_stack([design, hinge]),
            "mask": np.column_stack([b, mask]),
            "inputs": np.column_stack([b, mask, values]),
        }
        probabilities = []
        for features in choices.values():
            fit = LogisticRegression(C=1, solver="lbfgs", max_iter=1500, tol=1e-7)
            fit.fit(features[train], y[train])
            probabilities.append(fit.predict_proba(features[test])[:, 1])
        yield list(choices), compare_auc(np.column_stack(probabilities), y[test])


def summarize(base, organ, mask, values, y):
    folds = list(held_out_predictions(base, organ, mask, values, y))
    names = folds[0][0]
    weights = np.array([entry[2] for _, entry in folds], dtype=float)
    weights /= weights.sum()
    auc = sum(weight * entry[0] for weight, (_, entry) in zip(weights, folds))
    cov = sum(weight**2 * entry[1] for weight, (_, entry) in zip(weights, folds))
    out = {f"auc_{name}": val for name, val in zip(names, auc)}
    for first, second in (("base","score"), ("score","flex_score"), ("base","mask"),
                          ("mask","inputs"), ("base","inputs"), ("score","inputs")):
        i, j = names.index(first), names.index(second)
        delta = auc[j] - auc[i]
        low, high = ci95(delta, cov[i, i] + cov[j, j] - 2*cov[i, j], (-1, 1))
        key = f"delta_{second}_over_{first}"
        out.update({key:delta, f"{key}_ci95_low":low, f"{key}_ci95_high":high})
    return out


def main():
    bundle, scores, x, observed = load_inputs()
    results = []
    priors = pd.read_csv(RUN / "results" / "incident_metrics.csv").set_index("panel")
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        for panel in ENDPOINTS:
            path = RUN / "inputs" / f"{panel}.csv.gz"
            prior = priors.loc[panel]
            if digest(path) != prior.input_sha256:
                raise ValueError(f"Changed endpoint {panel}")
            cases = pd.read_csv(path).set_index("ID", verify_integrity=True)
            pred = scores.loc[cases.index]
            rows = scores.index.get_indexer(cases.index)
            cols = [bundle["features"].index(f) for f in bundle["groups"][panel]]
            mask, values = observed[np.ix_(rows, cols)], x[np.ix_(rows, cols)]
            np.testing.assert_array_equal(mask.sum(axis=1), pred[f"{panel}_n_observed"])
            y = cases.event.to_numpy(bool)
            assert len(y) == prior.n and y.sum() == prior.events
            age = (pred.age.to_numpy() - 60) / 10
            base = np.column_stack([age, age**2, pred.sex, pred.full_accel])
            organ = pred[f"{panel}_accel"].to_numpy()
            selected = {"primary": np.ones(len(y), dtype=bool)}
            flag = next((f for f in ("post_observed","post_landmark_tested","postlab_observed") if f in cases), None)
            if flag:
                monitored = cases[flag].astype("string").str.lower().isin(["1","1.0","true"]).to_numpy(bool)
                assert monitored.sum() == prior.post_observed_n and monitored[y].all()
                if not monitored.all():
                    selected["observed_again"] = monitored
            for subset, keep in selected.items():
                res = summarize(base[keep], organ[keep], mask[keep], values[keep], y[keep])
                results.append(dict(panel=panel, subset=subset, n=int(keep.sum()), events=int(y[keep].sum()),
                                    **res))
                print(panel, subset, round(res["auc_inputs"], 5), flush=True)
    out = RUN / "results" / "diagnostic_representation.csv"
    pd.DataFrame(results).to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
