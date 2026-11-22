"""Validate metabolic score against diabetes mortality without baseline self-reported diabetes."""

import json

import pandas as pd

from mortality import HERE, ROOT, COVARIATES, cohort, cox_result, fit_cox, oof_prediction, outcomes, save_json, sha


def adult_history(path):
    """Decode doctor-diagnosed history using adult.sas 1-based positions 1561–1564."""
    records = []
    with path.open("rb") as stream:
        for row in stream:
            participant = int(row[:5])
            diagnosis, sex, at_pregnancy, outside_pregnancy = row[1560:1564].decode("ascii")
            if diagnosis == "2":
                category = "no_selfreported_history"
            elif diagnosis == "1" and (sex == "1" or (sex == "2" and at_pregnancy == "2")):
                category = "known_nongestational"
            elif diagnosis == "1" and sex == "2" and at_pregnancy == "1":
                category = {
                    "1": "known_nongestational", "2": "gestational_history_only"
                }.get(outside_pregnancy, "unknown")
            else:
                category = "unknown"
            records.append((participant, category))
    frame = pd.DataFrame(records, columns=["ID", "diabetes_history"])
    if frame.ID.duplicated().any():
        raise ValueError("Adult questionnaire participant identifier is not unique")
    return frame


def main():
    data, audit = cohort()
    adult_path = ROOT / "nhanes3_source" / "adult.dat"
    source_hash = sha(adult_path)
    source_audit = json.loads((ROOT / "audit" / "nhanes3.json").read_text(encoding="utf-8"))
    if source_hash != source_audit["sources"][adult_path.name]["sha256"]:
        raise ValueError("Adult questionnaire differs from previously verified source")
    data = data.merge(adult_history(adult_path), on="ID", how="left", validate="one_to_one")
    if not data.diabetes_history.notna().all():
        raise ValueError("Adult questionnaire did not match all cohort participants")
    data["target"] = outcomes(data)["diabetes"]
    categories = {}
    for name, group in data.groupby("diabetes_history"):
        categories[name] = {"n": len(group), "diabetes_deaths": int(group.target.sum()),
                            "unknown_cause_deaths": int(group.unknown.sum())}

    selected = data.loc[data.diabetes_history.eq("no_selfreported_history") & ~data.unknown].copy()
    target = outcomes(selected)["diabetes"]
    before = fit_cox(selected, COVARIATES + ["full"], target)
    after = fit_cox(selected, COVARIATES + ["full", "metabolic"], target)
    scores = oof_prediction(selected, target, "metabolic")
    outputs = {
        "endpoint": "CDC public underlying cause of death UCOD_LEADING 007 (Diabetes mellitus)",
        "n": len(selected),
        "events": int(target.sum()),
        "history_categories": categories,
        "included_history": "HAD1=2: no self-reported physician-diagnosed diabetes. Public survey coding also groups borderline or prediabetic responses here.",
        "excluded_history": "Known non-gestational diabetes; gestational diabetes only was excluded separately, as were unknown, blank, or inconclusive diagnosis histories.",
        "other_exclusions": "Deaths with unknown underlying cause are excluded, as in the main analysis.",
        "interpretation": "Excludes reported diagnosis history only. Underlying diabetes deaths can reflect undiagnosed diabetes present at baseline or diabetes diagnosed subsequently; this is not prediction of incident diabetes.",
        "scale": "Full 14,807-person SD and mean are reused from the main analysis.",
        "source": {"file": str(adult_path.relative_to(ROOT)), "sha256": source_hash,
                   "codebook": "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/ADULT-acc.pdf",
                   "covariate_source_audit": audit},
        "incremental": cox_result(before, after, "metabolic"),
        "oof_prediction": scores,
    }
    save_json(HERE / "diabetes_without_known.json", outputs)
    print(f"Diabetes death after excluding reported baseline history: {outputs['n']:,} participants, {outputs['events']} deaths")
    print(json.dumps(outputs["incremental"], indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
