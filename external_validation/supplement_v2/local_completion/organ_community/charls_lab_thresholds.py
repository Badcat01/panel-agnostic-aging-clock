"""Exploratory paired CHARLS lab-threshold transitions on measured returnees."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from hrs_near_visit import HERE, SUPPLEMENT, sha

HEALTH = SUPPLEMENT / "healthspan"
sys.path.insert(0, str(HEALTH))
from repeat_biomarkers import charls, egfr
from prepare_healthspan import DATA, META, read

PANELS = ["kidney", "metabolic", "immune_heme", "cardiovascular"]


def egfr_values(frame, sex):
    female = sex.reindex(frame.index).eq(0)
    x = frame.creatinine.where(frame.creatinine.gt(0)) / np.where(female, .7, .9)
    y = frame.cystatin.where(frame.cystatin.gt(0)) / .8
    e_cr = (142 * x.clip(upper=1) ** np.where(female, -.241, -.302) * x.clip(lower=1) ** -1.2
            * .9938 ** frame.age * np.where(female, 1.012, 1))
    e_comb = (135 * x.clip(upper=1) ** np.where(female, -.219, -.144) * x.clip(lower=1) ** -.544
              * y.clip(upper=1) ** -.323 * y.clip(lower=1) ** -.778 * .9961 ** frame.age
              * np.where(female, .963, 1))
    return pd.DataFrame({"egfr_cys": egfr(frame.cystatin, frame.age, sex.reindex(frame.index)),
                         "egfr_cr": e_cr, "egfr_combined": e_comb})


def run():
    first, later = charls()
    base_path = HEALTH / "charls_baseline.csv.gz"
    visit_path = HEALTH / "charls_visits.csv.gz"
    scores = pd.read_csv(base_path, dtype={"ID": str}).set_index("ID", verify_integrity=True)
    followup = pd.read_csv(visit_path, dtype={"ID": str})
    baseline = followup.loc[followup.wave.eq(1), ["ID", "kidneye", "diabe"]].set_index("ID", verify_integrity=True)
    source = next(DATA.rglob("H_CHARLS_D_Data.dta"))
    communities = read(source, ["ID", "communityID"]).set_index("ID", verify_integrity=True)
    sex = scores.sex
    kidney_first, kidney_later = egfr_values(first, sex), egfr_values(later, sex)
    later_complete = kidney_later.notna().all(axis=1).reindex(kidney_first.index, fill_value=False)
    both_labs = kidney_first.ge(60).all(axis=1) & later_complete
    changes = [
        ("original", "kidney", "egfr_cys", kidney_first.egfr_cys, kidney_later.egfr_cys, "kidneye", 60, -1),
        ("original", "metabolic", "hba1c", first.hba1c.where(first.hba1c.gt(0)),
         later.hba1c.where(later.hba1c.gt(0)), "diabe", 6.5, 1),
    ]
    changes.extend(("same_roster", "kidney", name, kidney_first[name].where(both_labs),
                    kidney_later[name], "kidneye", 60, -1) for name in kidney_first)
    interval = ((later.origin - first.origin) / pd.Timedelta(days=365.25)).rename("interval")
    rows = []
    for category, matching, marker, start, finish, diagnosis, cut, direction in changes:
        known_free = baseline[diagnosis].eq(0)
        values = pd.concat([start.rename("baseline_measure"), finish.rename("followup_measure"), interval], axis=1)
        d = scores[["age", "sex", "education", "full_z", *PANELS]].join(values)
        d = d.join(communities.replace("", np.nan), validate="one_to_one")
        d = d.loc[d.interval.gt(0) & known_free.reindex(d.index).fillna(False)].dropna()
        above = d.baseline_measure.ge(cut)
        d = d.loc[above if direction == -1 else ~above].copy()
        crossed = d.followup_measure.ge(cut)
        d["event"] = (~crossed if direction == -1 else crossed).astype(int)
        d["age10"] = (d.age - 60) / 10
        d["age10sq"] = d.age10 ** 2
        d["measure"] = np.log(d.baseline_measure)
        n = len(d)
        evt = int(d.event.sum())
        assert d.index.is_unique and d.communityID.nunique() >= 50
        row = {"family": category, "matched": matching, "marker": marker,
               "n": n, "events": evt, "communities": d.communityID.nunique(),
               "baseline_median": d.baseline_measure.median(), "followup_median": d.followup_measure.median(),
               "paired_median_change": (d.followup_measure - d.baseline_measure).median(), "threshold": cut}
        if evt < 20:
            rows.append(row)
            continue
        adjust = ["age10", "age10sq", "sex", "interval", "measure", "full_z"]
        controls = sm.add_constant(d[adjust]).join(pd.get_dummies(d.education, drop_first=True, dtype=float))
        slopes, influences = {}, {}
        for panel in PANELS:
            model = sm.GLM(d.event, controls.assign(organ=d[panel]), family=sm.families.Binomial())
            fit = model.fit()
            assert fit.converged
            slopes[panel] = fit.params.organ
            scores_i = model.score_obs(fit.params)
            info = np.linalg.inv(-model.hessian(fit.params))
            influences[panel] = scores_i @ info[:, -1]
            np.testing.assert_allclose(np.sum(influences[panel]), 0, atol=1e-8)
        coupled = pd.DataFrame(influences, index=d.index).groupby(d.communityID).sum().to_numpy()
        factor = len(coupled) / (len(coupled) - 1)
        position = PANELS.index(matching)
        beta, other_beta = slopes[matching], max(v for p, v in slopes.items() if p != matching)
        se = np.sqrt(factor * (coupled[:, position] ** 2).sum())
        pair_effects = [beta - slopes[p] for p in PANELS if p != matching]
        ci_scale = [np.sqrt(factor * ((coupled[:, position] - coupled[:, i]) ** 2).sum())
                    for i in range(len(PANELS)) if i != position]
        p = norm.sf(np.array(pair_effects) / np.array(ci_scale)).max()
        rows.append({**row, "matched_OR": np.exp(beta), "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se),
                     "largest_other_OR": np.exp(other_beta), "matched_log_OR_margin": beta - other_beta, "p_IUT": p})
    output = pd.DataFrame(rows)
    for _, subset in output.dropna(subset=["p_IUT"]).groupby("family"):
        output.loc[subset.index, "q_IUT"] = multipletests(subset.p_IUT, method="fdr_bh")[1]
    assert output.loc[output.family.eq("same_roster"), "n"].nunique() == 1
    output.to_csv(HERE / "charls_incident_lab_thresholds.csv", index=False)
    for path in [base_path, visit_path]:
        META["healthspan/" + path.name] = {"sha256": sha(path)}
    (HERE / "charls_threshold_sources.json").write_text(json.dumps(META, ensure_ascii=False, indent=2), encoding="utf8")
    print(output.to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
