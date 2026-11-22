"""Run aggregate HRS reference and community disease analyses on the local releases."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyreadstat
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from hrs_near_visit import HERE, SUPPLEMENT, PANELS, WINDOWS, HRS, assemble, formal, indexed_records, sha, verify_reference
from hrs_references import matrix

HEALTH = SUPPLEMENT / "healthspan"
CARDIAC_SUBTYPES = ("hrtatte", "conhrtfe", "angine", "hrtrhme")
FLOW_SPECS = [("naive_cd4", -1), ("naive_cd8", -1), ("temra_cd8", 1), ("cd4_cd8", -1)]
SIMPLE_SPECS = {
    ("hrs", "diabe"): ("metabolic", "glucose"),
    ("elsa", "diabe"): ("metabolic", "hba1c"),
    ("elsa", "hearte"): ("cardiovascular", "sbp"),
    ("elsa", "stroke"): ("cardiovascular", "sbp"),
    ("charls", "diabe"): ("metabolic", "hba1c"),
    ("charls", "hearte"): ("cardiovascular", "sbp"),
    ("charls", "stroke"): ("cardiovascular", "sbp"),
    ("charls", "kidneye"): ("kidney", "egfrcr"),
}


def load(path):
    return pd.read_csv(path, dtype={"ID": str})


def risk_set(base, visits, endpoint):
    first_wave = visits.wave.min()
    clean_at_baseline = visits.loc[visits.wave.eq(first_wave) & visits[endpoint].eq(0), "ID"]
    base = base.loc[base.index.isin(clean_at_baseline)].copy()
    followup = visits.loc[visits.wave.gt(first_wave) & visits.time.gt(0) & visits[endpoint].isin([0, 1])]
    rows = followup.join(base[["death", "death_upper", "horizon"]], on="ID", how="inner")
    rows = rows.loc[rows.time.le(rows.horizon) & (rows.death.eq(0) | rows.time.le(rows.death_upper + .003))]
    first = rows.loc[rows[endpoint].eq(1)].groupby("ID").time.min().reindex(base.index)
    last = rows.groupby("ID").time.max().reindex(base.index)
    competed = base.death.eq(1) & base.death_or_censor.le(base.horizon)
    base["status"] = np.select([first.notna(), competed], [1, 2], 0)
    base["t"] = last.mask(competed, base.death_or_censor).mask(first.notna(), first)
    cols = ["age", "sex", "education", "fi0", "full_z", "ridge_lp"]
    base = base.loc[base.t.gt(0) & base[cols].notna().all(axis=1)]
    base["age10"] = (base.age - 60) / 10
    return base


def reference_analyses(near):
    near["age_c"] = (near.age - 65) / 10
    near["age_sq"] = near.age_c ** 2
    basic = ["age_c", "age_sq", "sex", "full_accel"]
    near["log_ntprobnp"] = np.log(near.PNTBNPE.where(near.PNTBNPE.gt(0)))
    near["log_cmv"] = np.log(near.PCMVGE.where(near.PCMVGE.gt(0)))
    references, audit = [], []
    flow, path = indexed_records("flocyt2016.sas7bdat", ["PCD4N_PCT", "PCD8N_PCT", "PCD8TEMRA_PCT", "PCD4T_PCT", "PCD8T_PCT"])
    assert flow.notna().sum().min() > 9000
    assert not ((flow < 0) | (flow > 1)).any().any()
    flow["cd4_cd8"] = np.log((flow.PCD4T_PCT / flow.PCD8T_PCT.where(flow.PCD8T_PCT.gt(0))).where(flow.PCD4T_PCT.gt(0)))
    flow = flow.rename(columns={"PCD4N_PCT": "naive_cd4", "PCD8N_PCT": "naive_cd8", "PCD8TEMRA_PCT": "temra_cd8"})
    flow = near.join(flow[[field for field, _ in FLOW_SPECS]], validate="one_to_one")
    for days in WINDOWS:
        constrained = near.loc[near.lag_days.between(0, days) & near.cardiovascular_accel.notna()]
        audit.append({"lookback_days": days, "n_complete_vitals": len(constrained),
                      "n_five_complete_with_ntprobnp": int(constrained[[f"{p}_accel" for p in PANELS] + ["log_ntprobnp"]].notna().all(axis=1).sum()),
                      "min_lag_days": constrained.lag_days.min(), "median_lag_days": constrained.lag_days.median(),
                      "max_lag_days": constrained.lag_days.max()})
        for row in matrix(constrained, "log_ntprobnp", f"NTproBNP_prior{days}d", "cardiovascular",
                          "prior_vitals_NTproBNP", basic, PANELS):
            references.append(row)
    window90 = flow.loc[flow.lag_days.between(0, 90) & flow.cardiovascular_accel.notna()]
    for subset, group, panels in [(flow, "same_draw_four", PANELS[:-1]), (window90, "vitals_prior90d_five", PANELS)]:
        for controls, suffix in [(basic, ""), (basic + ["log_cmv"], "_CMV")]:
            for field, direction in FLOW_SPECS:
                for row in matrix(subset, field, field, "immune_heme", group + suffix, controls, panels, direction):
                    references.append(row)
    output = pd.DataFrame(references)
    output["matched"] = output.panel.eq(output.expected)
    for family, rows in output.groupby("group"):
        chosen = rows.loc[rows.matched & rows.dominance_p_wild_IUT.notna()]
        if not chosen.empty:
            output.loc[chosen.index, "dominance_q_BH"] = multipletests(chosen.dominance_p_wild_IUT, method="fdr_bh")[1]
        output.loc[rows.index, "association_q_BH"] = multipletests(rows.p, method="fdr_bh")[1]
    output.to_csv(HERE / "hrs_reference_extensions.csv", index=False)
    pd.DataFrame(audit).to_csv(HERE / "hrs_23_vitals_coverage.csv", index=False)
    return path, output


def common_risk_template(cohort, endpoints):
    base_path = HEALTH / f"{cohort}_baseline.csv.gz"
    visit_path = HEALTH / f"{cohort}_visits.csv.gz"
    base = load(base_path).set_index("ID", verify_integrity=True)
    visits = load(visit_path)
    assert visits[["ID", "wave"]].duplicated().sum() == 0
    roster = pd.read_csv(HEALTH / "disease_diagonal.csv").set_index(["cohort", "endpoint"])
    for endpoint in endpoints:
        risk = risk_set(base, visits, endpoint)
        if (cohort, endpoint) in roster.index:
            old = roster.loc[cohort, endpoint]
            required = old.compared.split(";")
            original_complete = risk.loc[risk[required].notna().all(axis=1)]
            assert len(original_complete) == old.n
            assert original_complete.status.eq(1).sum() == old.events
        yield endpoint, risk


def harmonized_endpoints(cohort, visits):
    datasets = HRS.parent
    name, idcol = {"hrs": ("H_HRS_d.dta", "hhidpn"), "elsa": ("h_elsa_g3.dta", "idauniq")}[cohort]
    path = next(datasets.rglob(name))
    waves = sorted(visits.wave.unique())
    fields = [f"r{wave}{endpoint}" for wave in waves for endpoint in CARDIAC_SUBTYPES]
    source, metadata = pyreadstat.read_dta(path, usecols=[idcol, *fields])
    assert set(fields) <= set(metadata.column_names)
    ids = source.pop(idcol).astype("int64").astype(str)
    if cohort == "hrs":
        ids = ids.str.zfill(9)
        ids = ids.str[:-3] + "-" + ids.str[-3:]
    source.index = ids.rename("ID")
    assert source.index.is_unique
    observations = []
    for wave in waves:
        subtype = source[[f"r{wave}{x}" for x in CARDIAC_SUBTYPES]]
        subtype.columns = list(CARDIAC_SUBTYPES)
        subtype = subtype.where(subtype.isin([0, 1]))
        subtype["wave"] = wave
        observations.append(subtype.reset_index())
    return visits.merge(pd.concat(observations), on=["ID", "wave"], how="left", validate="one_to_one"), path


def disease_analyses(near, rscript):
    frames = []
    for endpoint, risk in common_risk_template("hrs", ("hearte", "stroke")):
        ref = verify_reference()["panels"]["cardiovascular"]
        risk["cardiovascular"] = near.cardiovascular_accel / ref["accel_sd_years"]
        risk["lag_days"] = near.lag_days
        for days in WINDOWS:
            clean = risk.loc[risk.lag_days.between(0, days) & risk[PANELS].notna().all(axis=1)].copy()
            clean["lookback_days"] = days
            clean["endpoint"] = endpoint
            clean["cohort"] = "hrs"
            clean["family"] = "primary"
            frames.append(clean)
    source_hashes = {}
    for cohort, windows, panels in [("hrs", (30, 90, 180), PANELS),
                                     ("elsa", (0,), ["metabolic", "immune_heme", "cardiovascular"])]:
        base = load(HEALTH / f"{cohort}_baseline.csv.gz").set_index("ID", verify_integrity=True)
        visits, source_path = harmonized_endpoints(cohort, load(HEALTH / f"{cohort}_visits.csv.gz"))
        source_hashes[source_path.name] = sha(source_path)
        if cohort == "hrs":
            ref = verify_reference()["panels"]["cardiovascular"]
            base["cardiovascular"] = near.cardiovascular_accel / ref["accel_sd_years"]
            base["lag_days"] = near.lag_days
        else:
            base["lag_days"] = 0
        for endpoint in CARDIAC_SUBTYPES:
            risk = risk_set(base, visits, endpoint)
            for days in windows:
                clean = risk.loc[risk.lag_days.between(0, days) & risk[panels].notna().all(axis=1)].copy()
                clean["lookback_days"] = days
                clean["endpoint"] = endpoint
                clean["cohort"] = cohort
                clean["family"] = "subtypes"
                frames.append(clean)
    hearte = pd.concat(frames, ignore_index=False).reset_index()

    marker_frames = []
    for cohort in ("hrs", "elsa", "charls"):
        raw, _ = formal(cohort)
        if cohort == "charls":
            x = raw.creatinine.where(raw.creatinine.gt(0)) / np.where(raw.sex.eq(0), .7, .9)
            raw["egfrcr"] = (142 * np.minimum(x, 1) ** np.where(raw.sex.eq(0), -.241, -.302)
                             * np.maximum(x, 1) ** -1.200 * .9938 ** raw.age * np.where(raw.sex.eq(0), 1.012, 1))
        endpoints = [endpoint for c, endpoint in SIMPLE_SPECS if c == cohort]
        for endpoint, risk in common_risk_template(cohort, endpoints):
            group, marker = SIMPLE_SPECS[(cohort, endpoint)]
            risk["marker"] = raw[marker]
            if marker == "glucose":
                vbs, _ = indexed_records("hrs2016vbs.sav", ["PFASTYN"])
                risk["marker"] = risk.marker.where(vbs.PFASTYN.eq(1))
            comp = risk[[group, "marker"]].notna().all(axis=1)
            use = risk.loc[comp].copy()
            use["group_z"] = use[group]
            use["cohort"] = cohort
            use["endpoint"] = endpoint
            use["group"] = group
            use["marker_name"] = marker
            marker_frames.append(use)
    simple = pd.concat(marker_frames).reset_index()
    with tempfile.TemporaryDirectory(prefix="aging_clock_community_") as target:
        path = Path(target)
        hearte.to_csv(path / "hrs_events.csv", index=False)
        simple.to_csv(path / "markers.csv", index=False)
        subprocess.run([str(rscript), str(HERE / "fit_community.R"), str(path), str(HERE)], check=True)
    return source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    args = parser.parse_args()
    near = assemble(args.device)
    flow, _ = reference_analyses(near)
    source_hashes = disease_analyses(near, args.rscript)
    audit = dict(near.attrs)
    audit["windows_days"] = WINDOWS
    audit["same_day_available"] = int((near.lag_days.eq(0) & near.cardiovascular_accel.notna()).sum())
    audit["source_sha256"][flow.name] = sha(flow)
    audit["source_sha256"].update(source_hashes)
    audit["source_sha256"].update({p.name: sha(p) for p in HEALTH.glob("*_baseline.csv.gz") if p.name.split("_")[0] in {"hrs","elsa","charls"} and p.name.count("_")==1})
    audit["source_sha256"].update({p.name: sha(p) for p in HEALTH.glob("*_visits.csv.gz") if p.name.split("_")[0] in {"hrs","elsa","charls"} and p.name.count("_")==1})
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
