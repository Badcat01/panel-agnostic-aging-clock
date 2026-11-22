"""Test if earlier blood results add to the latest ELSA blood draw for later disability."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / "healthspan"
sys.path.insert(0, str(PREVIOUS))
from elsa_last_core import core
from repeat_biomarkers import elsa
from prepare_healthspan import DATA, META, read


def auc_contribution(risk, y, weights):
    """Return comparable weighted pairs and concordant weighted pairs."""
    ranks = np.unique(risk, return_inverse=True)[1]
    positive = np.bincount(ranks, weights=weights * y, minlength=1 + ranks.max())
    negative = np.bincount(ranks, weights=weights * (1 - y), minlength=1 + ranks.max())
    numerator = positive @ (np.cumsum(negative) - 0.5 * negative)
    return numerator, positive.sum() * negative.sum()


def compare(base, pace, y):
    spl = StratifiedKFold(5, shuffle=True, random_state=20261126)
    blocks = []
    xb = base.to_numpy(dtype=float)
    xp = np.column_stack([xb, pace])
    for train, test in spl.split(xb, y):
        fitted = [
            sm.GLM(y[train], x[train], family=sm.families.Binomial()).fit()
            for x in (xb, xp)
        ]
        ranks = [f.predict(x[test], which="linear") for f, x in zip(fitted, (xb, xp))]
        blocks.append((y[test], ranks))
    rng = np.random.default_rng(20261126)

    def calculate(draw):
        concordance = np.zeros(2)
        total_pairs = 0
        for yy, pr in blocks:
            size = len(yy)
            w = np.bincount(rng.integers(size, size=size), minlength=size) if draw else np.ones(size)
            c0, n0 = auc_contribution(pr[0], yy, w)
            c1, n1 = auc_contribution(pr[1], yy, w)
            assert n0 == n1
            concordance += (c0, c1)
            total_pairs += n0
        return concordance / total_pairs

    both = calculate(False)
    for yy, pr in blocks:
        for r in pr:
            c, n = auc_contribution(r, yy, np.ones(len(yy)))
            np.testing.assert_allclose(c / n, roc_auc_score(yy, r), atol=1e-12)
    boot = np.empty(2000)
    for i in range(len(boot)):
        result = calculate(True)
        boot[i] = result[1] - result[0]
    ci = np.quantile(boot, [0.025, 0.975])
    return dict(base_auc=both[0], augmented_auc=both[1], delta_auc=both[1] - both[0],
                delta_lo=ci[0], delta_hi=ci[1], bootstrap=len(boot))


def main():
    a = pd.concat([elsa(w).assign(wave=w) for w in (8, 9)]).sort_values("origin", kind="stable")
    assert a.index.duplicated().sum() == 2
    a = a.loc[~a.index.duplicated(keep="first")]
    c8, c9, c10 = (core(w) for w in (8, 9, 10))
    harmonized = read(next(DATA.rglob("h_elsa_g3.dta")), ["idauniq", "ragender", "raeducl"])
    harmonized.index = harmonized.idauniq.astype("int64").astype(str)
    assert harmonized.index.is_unique
    base = a[["wave", "age", "origin"]].join(harmonized[["ragender", "raeducl"]])
    base["sex"] = base.ragender.map({1: 1, 2: 0})
    base["age10"] = (base.age - 60) / 10
    base["age10sq"] = base.age10 ** 2
    base["wave9"] = base.wave.eq(9).astype(int)
    visit = pd.concat([c8.reindex(base.index[base.wave.eq(8)]), c9.reindex(base.index[base.wave.eq(9)])])
    days = pd.Timedelta(days=365.25)
    base["follow_years"] = (c10.date.reindex(base.index) - base.origin) / days
    base = base.loc[base.follow_years.gt(0) & visit.date.reindex(base.index).le(base.origin)]

    rows = []
    pace_sources = {}
    for start in (2, 4, 6):
        f = PREVIOUS / f"elsa_pooled_w89_pace_from_w{start}.csv.gz"
        pace_sources[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        pace = pd.read_csv(f, dtype={"ID": str}).set_index("ID", verify_integrity=True)
        data = base.join(pace)
        np.testing.assert_allclose(data.loc[data.pace.notna(), "age"],
                                   data.loc[data.pace.notna(), "landmark_age"])
        data["recent_ba"] = data.baseline_ba + data.pace * data.time_yr
        for endpoint in ("adl", "iadl"):
            covars = ["age10", "age10sq", "sex", "raeducl", "follow_years", "wave9", "time_yr", "recent_ba"]
            pair = data.assign(baseline=visit[endpoint], end=c10[endpoint])
            pair = pair.loc[pair.baseline.eq(0)].dropna(subset=["end", "pace", *covars])
            x0 = sm.add_constant(pd.get_dummies(pair[covars], columns=["raeducl"],
                                                drop_first=True, dtype=float))
            x1 = x0.assign(pace=pair.pace)
            y = pair.end.to_numpy(dtype=int)
            old = sm.GLM(y, x0, family=sm.families.Binomial()).fit(cov_type="HC0")
            fit = sm.GLM(y, x1, family=sm.families.Binomial()).fit(cov_type="HC0")
            lo, hi = np.exp(fit.conf_int().loc["pace"])
            rows.append(dict(first_wave=start, endpoint=endpoint, n=len(pair), events=y.sum(),
                             observational_followup_median=pair.follow_years.median(),
                             OR=np.exp(fit.params.pace), lo=lo, hi=hi, p=fit.pvalues.pace,
                             lrt_p=chi2.sf(2 * (fit.llf - old.llf), 1),
                             **compare(x0, pair.pace.to_numpy(), y)))
    table = pd.DataFrame(rows)
    table["q"] = multipletests(table.p, method="fdr_bh")[1]
    table.to_csv(HERE / "elsa_late_pace_beyond_latest_score.csv", index=False)
    (HERE / "elsa_late_pace_sources.json").write_text(
        json.dumps(dict(source_sha256=pace_sources, sources=META,
                        conditional_on_w10_return=True, paired_fold_auc_crosscheck=True),
                   ensure_ascii=False, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
