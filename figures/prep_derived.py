"""Derive descriptive datasets (Table 1, KM curves, organ profiles) from the archived
individual-level public NHANES files. Read-only on the source tree."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path

R = sys.argv[1]          # .../src/aging_clock
OUT = sys.argv[2]        # derived output dir
Path(OUT).mkdir(parents=True, exist_ok=True)

bundle = json.load(open(f"{R}/runs/nhanes_gpu_20261121/bundle/clock_bundle.json", encoding="utf-8"))
FEATS = bundle["features"]
ref = json.load(open(f"{R}/external_validation/nhanes3_organ/organ_score_reference.json", encoding="utf-8"))["panels"]
PANELS = ["full", "liver", "kidney", "metabolic", "immune_heme", "cardiovascular"]

# ---------------- Table 1: development and temporal holdout -----------------
proc = pd.read_csv(f"{R}/data/processed/nhanes_9918.csv.gz")
dev_ids = pd.read_csv(f"{R}/runs/nhanes_gpu_20261121/development_ids.csv.gz")
tp = pd.read_csv(f"{R}/runs/nhanes_gpu_20261121/temporal_predictions.csv.gz")
proc["n_meas"] = proc[FEATS].notna().sum(axis=1)

def describe(df, label, n_meas=None):
    age = df["age"]; fem = 1 - df["male"]
    return dict(cohort=label, n=len(df), deaths=int(df["event"].sum()),
                age_mean=age.mean(), age_sd=age.std(), age_median=age.median(),
                age_q1=age.quantile(.25), age_q3=age.quantile(.75),
                female_pct=100 * fem.mean(), age65_pct=100 * (age >= 65).mean(),
                fu_median=df["time_yr"].median(),
                meas_median=(n_meas if n_meas is not None else np.nan))

dev = proc[proc.SEQN.isin(dev_ids.SEQN)]
rows = [describe(dev, "NHANES 1999-2008 development", dev["n_meas"].median()),
        describe(tp, "NHANES 2009-2018 temporal holdout", tp["full_n_observed"].median())]
t1 = pd.DataFrame(rows)
# race/ethnicity distribution for the two NHANES sets (NHANES RIDRETH1 codes)
race_map = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
            4: "Non-Hispanic Black", 5: "Other/multiracial"}
tp_r = tp.merge(proc[["SEQN", "race"]], on="SEQN", how="left")
for lab, df in [("dev", dev), ("holdout", tp_r)]:
    vc = df["race"].map(race_map).value_counts(normalize=True) * 100
    for k, v in vc.items():
        t1.loc[t1.cohort.str.contains("development" if lab == "dev" else "holdout"), f"race_{k}"] = v
t1.to_csv(f"{OUT}/table1_nhanes.csv", index=False)
print(t1.T)

# ---------------- Kaplan-Meier by acceleration quartile -----------------------
def km(time, event, grid):
    order = np.argsort(time); t = np.asarray(time)[order]; e = np.asarray(event)[order]
    uniq = np.unique(t[e == 1]); s = 1.0; surv_at = []
    n = len(t); idx = 0; S = {}
    at_risk = n
    j = 0
    for ut in uniq:
        # remove those with time < ut
        while j < n and t[j] < ut:
            j += 1
        r = n - j
        d = np.sum((t == ut) & (e == 1))
        s *= (1 - d / r)
        S[ut] = s
    keys = np.array(list(S.keys())); vals = np.array(list(S.values()))
    out = []
    for g in grid:
        k = np.searchsorted(keys, g, side="right") - 1
        out.append(1.0 if k < 0 else vals[k])
    return np.array(out)

def km_by_quartile(df, accel_col, tcol, ecol, label, tmax):
    q = pd.qcut(df[accel_col], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grid = np.round(np.arange(0, tmax + 1e-9, 1 / 12), 4)
    recs = []
    for lev in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[q == lev]
        s = km(sub[tcol].values, sub[ecol].values, grid)
        for g, v in zip(grid, s):
            recs.append(dict(cohort=label, quartile=lev, time=g, survival=v))
        # number at risk at integer years
    risk = []
    for lev in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[q == lev]
        for yr in range(0, int(tmax) + 1, 2 if tmax <= 12 else 5):
            risk.append(dict(cohort=label, quartile=lev, time=yr, n_at_risk=int((sub[tcol] >= yr).sum()),
                             n=len(sub), deaths=int(sub[ecol].sum())))
    return pd.DataFrame(recs), pd.DataFrame(risk)

km_t, risk_t = km_by_quartile(tp, "full_accel", "time_yr", "event", "NHANES 2009-2018 holdout", 11)
n3 = pd.read_csv(f"{R}/external_validation/results/nhanes3/predictions.csv.gz")
km_n3, risk_n3 = km_by_quartile(n3, "full_accel", "time_yr", "event", "NHANES III", 25)
pd.concat([km_t, km_n3]).to_csv(f"{OUT}/km_quartiles.csv", index=False)
pd.concat([risk_t, risk_n3]).to_csv(f"{OUT}/km_numbers_at_risk.csv", index=False)

# ---------------- Risk-equivalent age vs chronological age --------------------
tp[["age", "full_ba"]].to_csv(
    f"{OUT}/holdout_age_vs_rea.csv.gz", index=False)

# ---------------- organ score correlation and sex-age profiles ---------------
acc = tp[[f"{p}_accel" for p in PANELS]].copy()
for p in PANELS:
    acc.loc[tp[f"{p}_no_observations"] == True, f"{p}_accel"] = np.nan
corr = acc.corr(method="spearman")
corr.index = PANELS; corr.columns = PANELS
corr.to_csv(f"{OUT}/organ_spearman_holdout.csv")

prof = []
bins = [20, 30, 40, 50, 60, 70, 80, 90]
tp["age_bin"] = pd.cut(tp["age"], bins=bins, right=False, labels=[f"{a}-{b-1}" for a, b in zip(bins[:-1], bins[1:])])
for p in PANELS:
    sd = ref[p]["accel_sd_years"]
    score = 50 + 10 * (tp[f"{p}_accel"] - ref[p]["accel_mean_years"]) / sd
    score[tp[f"{p}_no_observations"] == True] = np.nan
    for sex, sexlab in [(0, "Female"), (1, "Male")]:
        for ab in tp["age_bin"].cat.categories:
            m = (tp["male"] == sex) & (tp["age_bin"] == ab) & score.notna()
            v = score[m]
            if len(v) < 30:
                continue
            se = v.std() / np.sqrt(len(v))
            prof.append(dict(panel=p, sex=sexlab, age_bin=ab, age_mid=int(ab.split("-")[0]) + 5,
                             n=len(v), mean=v.mean(), lo=v.mean() - 1.96 * se, hi=v.mean() + 1.96 * se))
pd.DataFrame(prof).to_csv(f"{OUT}/organ_sex_age_profiles.csv", index=False)
print(pd.DataFrame(prof).groupby(["panel", "sex"])["mean"].describe())
print(corr.round(2))
print(risk_t.head(12)); print(risk_n3.head(8))
