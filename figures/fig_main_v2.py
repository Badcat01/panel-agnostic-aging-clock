"""Main Figures 1-6, version 2: row-aligned grid layout.

Rules: every panel in a row shares the same plot-area top and bottom (Box.top/Box.bottom);
one centred single-line title per panel; legends in a band under the title; column
splits are consistent within each figure."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import figlib as F
import data_src as DS
from fig_main import src, write_source, yrows, FIGDIR, PANDIR

FTMASK_PNG = os.environ.get("AC_FTMASK_PNG", Path(__file__).parent / "assets" / "ftmask_embedded.png")          # 2048 x 665 px raster extracted from the uploaded PDF
COH = {"temporal": "Temporal holdout", "nhanes3": "NHANES III", "hrs": "HRS", "charls": "CHARLS",
       "elsa": "ELSA", "share": "SHARE", "klosa": "KLoSA", "mimic": "MIMIC-IV", "mimic_iv": "MIMIC-IV"}
CTRY = {"temporal": "USA", "nhanes3": "USA", "hrs": "USA", "charls": "China", "elsa": "UK"}
LEG = dict(top=12.5)     # rows that contain a legend band
NOLEG = dict(top=8.0)    # rows without legends


def km(B, cohort, title, xmax, xt, fig, pid):
    B.title(title)
    k = DS.derived("km_quartiles.csv"); k = k[k.cohort == cohort]
    labs = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    B.legend([F.handle(c, marker=None) for c in F.QUART], labs)
    ax = B.plot(13)
    for q, c in zip(["Q1", "Q2", "Q3", "Q4"], F.QUART):
        kk = k[k.quartile == q]
        ax.step(kk.time, 100 * (1 - kk.survival), where="post", color=c, lw=1.1)
    ax.set_xlim(0, xmax); ax.set_xticks(xt); ax.set_ylim(0, None)
    ax.set_xlabel("Years since examination"); ax.set_ylabel("Cumulative mortality (%)")
    F.style_axis(ax, "y")
    src(fig, pid, k)


# ================================================================ FIGURE 1
def f1a(B):
    B.title("Study design")
    img = plt.imread(FTMASK_PNG)
    w = 2048 / 300 * 25.4; h = 665 / 300 * 25.4          # native 300 dpi size (173.4 x 56.3 mm)
    ax = B.ax((B.w - w) / 2, 7.0, w, h)
    ax.imshow(img, interpolation="none", aspect="auto")
    ax.set_axis_off()
    src("Fig1", "a", pd.DataFrame({"element": ["Development cohort", "Stochastic panel masking", "Frozen FT-Mask",
                                               "Temporal holdout", "Community cohorts", "Hospital admissions",
                                               "DNA methylation clock comparison"],
                                   "n": ["19,426", "40% full / 35% random / 25% single system", "5 seeds",
                                         "25,614", "43,104 (4 primary) + 3 supplementary", "130,242",
                                         "2,525 (NHANES); 3,967 (HRS)"]}))


def f1b(B):
    B.title("Measured inputs per setting")
    rows = [("Development", 43, 1), ("Temporal holdout", 43, 1), ("NHANES III", 40, 1), ("HRS", 31, 1),
            ("MIMIC-IV", 19, 1), ("CHARLS", 18, 1), ("ELSA", 12, 1), ("MHAS", 10, 0), ("SHARE", 6, 0),
            ("KLoSA", 1, 0)]
    B.legend([F.patch(F.C_CLOCK), F.patch("#b9b8b3")], ["Primary", "Supplementary"])
    ax = B.plot(22)
    for i, (lab, v, prim) in enumerate(rows):
        ax.barh(i, v, height=0.62, color=F.C_CLOCK if prim else "#b9b8b3", zorder=2)
        ax.text(v + 0.8, i, str(v), va="center", ha="left", fontsize=F.FS_ANN)
    yrows(ax, [r[0] for r in rows])
    ax.set_xlim(0, 50); ax.set_xticks([0, 10, 20, 30, 43])
    ax.set_xlabel("Median measured inputs (of 43)")
    F.style_axis(ax, "x")
    src("Fig1", "b", pd.DataFrame(rows, columns=["setting", "median_measured_inputs", "primary"]))


def f1c(B):
    B.title("Risk-equivalent vs chronological age")
    d = DS.derived("holdout_age_vs_rea.csv.gz")
    ax = B.plot(12, right=13)
    cmap = LinearSegmentedColormap.from_list("b", ["#eef4fc", "#a7c9f0", "#2a78d6", "#0d2f5c"])
    hb = ax.hexbin(d["age"], d["full_ba"], gridsize=32, cmap=cmap, mincnt=1, bins="log", linewidths=0.1,
                   extent=(20, 80, 0, 120))
    ax.plot([20, 80], [20, 80], ls="--", lw=0.7, color=F.INK2, zorder=3)
    ax.set_xlim(19, 81); ax.set_ylim(0, 120); ax.set_yticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("Risk-equivalent age (years)")
    cax = B.ax(B.w - 11, B.top + 4, 1.8, B.h - B.top - B.bottom - 8)
    cb = plt.colorbar(hb, cax=cax)
    cb.set_ticks([1, 10, 100]); cb.set_ticklabels(["1", "10", "100"]); cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.4, length=1.5, width=0.4)
    cb.set_label("Participants", fontsize=5.8, labelpad=1.5)
    src("Fig1", "c", d[["age", "full_ba"]].rename(columns={"age": "chronological_age",
                                                         "full_ba": "risk_equivalent_age"}).round(2))


def f1d(B):
    B.title("Temporal holdout discrimination")
    s = DS.fig1b(); t = s[s.cohort == "temporal"].set_index("model")
    rows = [("Clinical clock", "full_lp", F.C_CLOCK, "o"), ("Ridge Cox", "ridge_lp", F.C_RIDGE, "s"),
            ("Age + sex Cox", "age_sex_lp", F.C_AGESEX, "D")]
    ax = B.plot(21)
    for i, (lab, k, c, mk) in enumerate(rows):
        F.ci_dot(ax, i, t.loc[k, "C"], t.loc[k, "lower"], t.loc[k, "upper"], c, marker=mk, ms=4.0)
        ax.text(t.loc[k, "upper"] + 0.003, i, f"{t.loc[k,'C']:.3f}", va="center", fontsize=F.FS_ANN)
    yrows(ax, [r[0] for r in rows], pad=0.7)
    ax.set_xlim(0.815, 0.905); ax.set_xticks([0.82, 0.84, 0.86, 0.88, 0.90])
    ax.set_xlabel("Harrell's C (95% CI)")
    F.style_axis(ax, "x")
    src("Fig1", "d", t.reset_index())


def f1e(B):
    km(B, "NHANES 2009-2018 holdout", "Holdout mortality by acceleration quartile", 11, range(0, 12, 2), "Fig1", "e")


def f1f(B):
    B.title("Comparison with clinical PhenoAge")
    tm = DS.temporal_metrics()["phenoage_complete_cases"]
    rows = [("Holdout,\nPhenoAge-complete", tm["C_DL"], tm["C_ridge_cox"], tm["C_PhenoAge"]),
            ("NHANES III,\nall inputs", 0.8591, None, 0.8471),
            ("NHANES III,\nsame nine inputs", 0.8479, None, 0.8471)]
    B.legend([F.handle(F.C_CLOCK, ls="none"), F.handle(F.C_RIDGE, "s", ls="none"),
              F.handle("#4d4d4d", "D", ls="none", ms=3.2)], ["Clinical clock", "Ridge Cox", "PhenoAge"])
    ax = B.plot(25)
    for i, (lab, c, r, p) in enumerate(rows):
        ax.plot([p, c], [i, i], color="#dddddd", lw=0.9, zorder=1)
        F.ci_dot(ax, i, p, None, None, "#4d4d4d", marker="D", ms=3.4)
        if r is not None:
            F.ci_dot(ax, i, r, None, None, F.C_RIDGE, marker="s", ms=3.6)
        F.ci_dot(ax, i, c, None, None, F.C_CLOCK, ms=4.2)
        ax.text(max(c, p) + 0.0025, i, f"Δ{c - p:+.3f}", va="center", fontsize=F.FS_ANN, color=F.INK2)
    yrows(ax, [r[0] for r in rows], pad=0.7)
    ax.set_xlim(0.84, 0.895); ax.set_xticks([0.84, 0.85, 0.86, 0.87, 0.88, 0.89])
    ax.set_xlabel("Harrell's C")
    F.style_axis(ax, "x")
    src("Fig1", "f", pd.DataFrame(rows, columns=["setting", "C_clock", "C_ridge", "C_PhenoAge"]).assign(
        n=[tm["n"], 15065, 15065], deaths=[tm["deaths"], 6520, 6520]))


def fig1():
    W = F.FIG_W
    r2, r3 = 68, 132
    panels = [("a", f1a, 0, 0, W, 66),
              ("b", f1b, 0, r2, 58, 62, LEG), ("c", f1c, 58, r2, 62, 62, LEG), ("d", f1d, 120, r2, 63, 62, LEG),
              ("e", f1e, 0, r3, 91.5, 62, LEG), ("f", f1f, 91.5, r3, 91.5, 62, LEG)]
    F.build(panels, 194, f"{FIGDIR}/Fig1", f"{PANDIR}/Fig1", "Fig1")
    write_source("Fig1")


# ================================================================ FIGURE 2
def f2a(B):
    B.title("Paired gain in discrimination")
    s = DS.fig1b()
    order = ["temporal", "nhanes3", "hrs", "charls", "elsa"]
    B.legend([F.handle(F.C_RIDGE), F.handle(F.C_AGESEX, "s", ms=3.4)], ["vs same-input ridge Cox", "vs age + sex Cox"],
             cx=57 + (B.w - 60) / 2)
    ax = B.plot(57)
    n = len(order)
    cols = [(0.5, "Cohort", "left"), (33.5, "n", "right"), (44.0, "Deaths", "right"), (53.5, "Inputs", "right")]
    for x, h, ha in cols:
        B.text(x, B.top - 1.0, h, ha=ha, va="bottom", fontsize=F.FS_TICK, fontweight="bold")
    rec = []
    for i, c in enumerate(order):
        r = s[(s.cohort == c) & (s.model == "ridge_lp")].iloc[0]
        a = s[(s.cohort == c) & (s.model == "age_sex_lp")].iloc[0]
        F.ci_dot(ax, i - 0.14, r.clock_minus_model, r.delta_low, r.delta_high, F.C_RIDGE)
        F.ci_dot(ax, i + 0.14, a.clock_minus_model, a.delta_low, a.delta_high, F.C_AGESEX, marker="s", ms=3.4)
        y = B.rowy(i, n)
        B.text(0.5, y, f"{COH[c]} ({CTRY[c]})", ha="left", va="center", fontsize=F.FS_TICK)
        B.text(33.5, y, F.fmt_n(r.n), ha="right", va="center", fontsize=F.FS_TICK)
        B.text(44.0, y, F.fmt_n(r.events), ha="right", va="center", fontsize=F.FS_TICK)
        B.text(53.5, y, str(int(r.median_features)), ha="right", va="center", fontsize=F.FS_TICK)
        rec += [dict(cohort=c, comparator=k, delta=x.clock_minus_model, lo=x.delta_low, hi=x.delta_high,
                     n=x.n, deaths=x.events, inputs=x.median_features) for k, x in (("ridge", r), ("age_sex", a))]
    ax.axvline(0, color=F.REF, lw=0.6, ls="--", zorder=1)
    ax.axhline(0.5, color="#d0d0d0", lw=0.5)
    ax.set_yticks([]); ax.set_ylim(n - 1 + 0.6, -0.6); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.004, 0.08); ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_xlabel("ΔHarrell's C, clock − comparator (95% CI)")
    F.style_axis(ax, "x")
    src("Fig2", "a", pd.DataFrame(rec))


def f2b(B):
    B.title("Time-dependent AUC")
    h = DS.horizons(); h = h[h.supported == True]
    rows = []
    for c in ["NHANES III", "HRS 2016", "CHARLS", "ELSA"]:
        for yr in [3, 5, 10]:
            hh = h[(h.cohort == c) & (h.horizon_years == yr)]
            if len(hh):
                rows.append((f"{c.replace(' 2016', '')}, {yr} y", hh.set_index("method")))
    B.legend([F.handle(F.C_CLOCK, ls="none"), F.handle(F.C_RIDGE, "s", ls="none"),
              F.handle(F.C_AGESEX, "D", ls="none", ms=3.2)], ["Clock", "Ridge Cox", "Age + sex"])
    ax = B.plot(22)
    rec = []
    for i, (lab, hh) in enumerate(rows):
        ax.plot([hh.loc["age_sex", "auc"], hh.loc["clock", "auc"]], [i, i], color="#dddddd", lw=0.8, zorder=1)
        F.ci_dot(ax, i, hh.loc["age_sex", "auc"], None, None, F.C_AGESEX, marker="D", ms=3.2)
        F.ci_dot(ax, i, hh.loc["ridge", "auc"], None, None, F.C_RIDGE, marker="s", ms=3.4)
        F.ci_dot(ax, i, hh.loc["clock", "auc"], None, None, F.C_CLOCK, ms=4.0)
        rec.append(dict(row=lab, clock=hh.loc["clock", "auc"], ridge=hh.loc["ridge", "auc"],
                        age_sex=hh.loc["age_sex", "auc"]))
    yrows(ax, [r[0] for r in rows])
    ax.set_xlim(0.735, 0.915); ax.set_xticks([0.75, 0.80, 0.85, 0.90])
    ax.set_xlabel("Cumulative/dynamic AUC")
    F.style_axis(ax, "x")
    src("Fig2", "b", pd.DataFrame(rec))


def f2c(B):
    km(B, "NHANES III", "NHANES III mortality by acceleration quartile", 25, range(0, 26, 5), "Fig2", "c")


def f2d(B):
    B.title("Discrimination in women and men")
    g = DS.subgroups()
    order = ["temporal", "nhanes3", "hrs", "charls", "elsa", "mimic"]
    B.legend([F.handle(F.C_FEMALE), F.handle(F.C_MALE, "s", ms=3.3)], ["Women", "Men"])
    ax = B.plot(22)
    rec = []
    for i, c in enumerate(order):
        for sx, off, col, mk in [("female", -0.15, F.C_FEMALE, "o"), ("male", 0.15, F.C_MALE, "s")]:
            r = g[(g.cohort == c) & (g.group == sx)].iloc[0]
            F.ci_dot(ax, i + off, r.harrell_C, r.harrell_C_low, r.harrell_C_high, col, marker=mk,
                     ms=3.6 if mk == "o" else 3.3)
            rec.append(dict(cohort=c, sex=sx, n=r.n, C=r.harrell_C, lo=r.harrell_C_low, hi=r.harrell_C_high))
    yrows(ax, [COH[c] + (" (1 y)" if c == "mimic" else "") for c in order])
    ax.set_xlim(0.72, 0.90); ax.set_xticks([0.74, 0.78, 0.82, 0.86, 0.90])
    ax.set_xlabel("Harrell's C (95% CI)")
    F.style_axis(ax, "x")
    src("Fig2", "d", pd.DataFrame(rec))


def f2e(B):
    B.title("Frozen clock versus multicohort retraining")
    l = DS.loco()
    order = ["temporal", "nhanes3", "hrs", "charls", "elsa", "share", "klosa", "mimic_iv"]
    ax = B.plot(25)
    rec = []
    for i, c in enumerate(order):
        r = l[l.target == c].iloc[0]
        sig = r.delta_vs_v1_hi < 0 or r.delta_vs_v1_lo > 0
        F.ci_dot(ax, i, r.delta_multi_minus_v1, r.delta_vs_v1_lo, r.delta_vs_v1_hi, F.C_RIDGE,
                 mfc=F.C_RIDGE if sig else "white")
        rec.append(dict(target=c, n=r.n, events=r.events, C_frozen=r.v1_C, C_multicohort=r.multi_C,
                        delta=r.delta_multi_minus_v1, lo=r.delta_vs_v1_lo, hi=r.delta_vs_v1_hi))
    ax.axvline(0, color=F.REF, lw=0.6, ls="--")
    yrows(ax, [COH[c] for c in order])
    ax.set_xlim(-0.019, 0.011); ax.set_xticks([-0.015, -0.01, -0.005, 0, 0.005, 0.01])
    ax.set_xticklabels(["−0.015", "−0.010", "−0.005", "0", "0.005", "0.010"])
    ax.set_xlabel("ΔHarrell's C, retrained − frozen (95% CI)")
    F.style_axis(ax, "x")
    src("Fig2", "e", pd.DataFrame(rec))


def f2f(B):
    B.title("Hazard per SD acceleration")
    s = DS.cohorts(); tm = DS.temporal_metrics()["full_accel_adjusted_age_sex"]
    rows = [("Temporal holdout", tm["hr_per_holdout_sd"], tm["ci95"][0], tm["ci95"][1], tm["sd_years"])]
    for c in ["NHANES III", "HRS 2016", "CHARLS", "ELSA"]:
        r = s[s.cohort == c].iloc[0]
        rows.append((c.replace(" 2016", ""), r.accel_HR, r.accel_HR_lower, r.accel_HR_upper, r.accel_SD_years))
    ax = B.plot(22)
    for i, (lab, hr, lo, hi, sd) in enumerate(rows):
        F.ci_dot(ax, i, hr, lo, hi, F.C_CLOCK, ms=4.0)
        ax.text(hi * 1.03, i, f"{hr:.2f}", va="center", fontsize=F.FS_ANN)
    ax.set_xscale("log"); ax.axvline(1, color=F.REF, lw=0.6, ls="--"); ax.minorticks_off()
    ax.set_xlim(0.95, 3.4); ax.set_xticks([1, 1.5, 2, 3]); ax.set_xticklabels(["1", "1.5", "2", "3"])
    yrows(ax, [r[0] for r in rows])
    ax.set_xlabel("HR per SD, age/sex-adjusted (95% CI)")
    F.style_axis(ax, "x")
    src("Fig2", "f", pd.DataFrame(rows, columns=["cohort", "HR", "lo", "hi", "SD_years"]))


def fig2():
    L, R = 108, 75
    panels = [("a", f2a, 0, 0, L, 64, LEG), ("b", f2b, L, 0, R, 64, LEG),
              ("c", f2c, 0, 65, L, 60, LEG), ("d", f2d, L, 65, R, 60, LEG),
              ("e", f2e, 0, 126, L, 58, NOLEG), ("f", f2f, L, 126, R, 58, NOLEG)]
    F.build(panels, 184, f"{FIGDIR}/Fig2", f"{PANDIR}/Fig2", "Fig2")
    write_source("Fig2")


# ================================================================ FIGURE 3
PANEL_ORDER = [("share", "SHARE", 6), ("elsa", "ELSA", 12), ("charls", "CHARLS", 18), ("mimic", "MIMIC-IV", 19),
               ("hrs", "HRS", 31), ("nhanes3", "NHANES III", 40), ("full", "Full", 43)]


def panel_ticks(ax):
    ax.set_xticks(range(len(PANEL_ORDER)))
    ax.set_xticklabels([f"{lab}\n{n}" for _, lab, n in PANEL_ORDER])
    ax.set_xlim(-0.5, len(PANEL_ORDER) - 0.5)
    ax.set_xlabel("Panel (number of inputs)")


def f3a(B):
    B.title("Simulated panels in 25,614 participants")
    p = DS.panels_sim()
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_RIDGE, "s", ms=3.2), F.handle(F.C_AGESEX, "D", ms=3.0)],
             ["Frozen masked clock", "Panel-specific", "No masking"])
    ax = B.plot(14)
    m0, m1, m2 = [], [], []
    for k, lab, n in PANEL_ORDER:
        a = p[(p.panel == k) & (p.model == "M1_nomask")].iloc[0]
        m0.append(a.M0_C); m1.append(a.comparator_C)
        b = p[(p.panel == k) & (p.model == "M2_panel")]
        m2.append(b.iloc[0].comparator_C if len(b) else np.nan)
    xs = np.arange(len(PANEL_ORDER))
    ax.plot(xs, m1, color=F.C_AGESEX, marker="D", ms=3.0, lw=0.9, mec="white", mew=0.4)
    ax.plot(xs, m2, color=F.C_RIDGE, marker="s", ms=3.2, lw=0.9, mec="white", mew=0.4)
    ax.plot(xs, m0, color=F.C_CLOCK, marker="o", ms=3.8, lw=1.1, mec="white", mew=0.4)
    panel_ticks(ax); ax.set_ylim(0.826, 0.882); ax.set_ylabel("Harrell's C")
    F.style_axis(ax, "y")
    src("Fig3", "a", pd.DataFrame({"panel": [x[1] for x in PANEL_ORDER], "inputs": [x[2] for x in PANEL_ORDER],
                                   "C_frozen": m0, "C_nomask": m1, "C_panel_specific": m2}))


def f3b(B):
    B.title("Cost of a single frozen model")
    p = DS.panels_sim()
    B.legend([F.handle(F.C_AGESEX, "D", ms=3.0), F.handle(F.C_RIDGE, "s", ms=3.2)],
             ["vs no masking", "vs panel-specific"])
    ax = B.plot(14)
    rec = []
    for i, (k, lab, n) in enumerate(PANEL_ORDER):
        for mdl, off, col, mk in [("M1_nomask", -0.14, F.C_AGESEX, "D"), ("M2_panel", 0.14, F.C_RIDGE, "s")]:
            r = p[(p.panel == k) & (p.model == mdl)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.plot([i + off, i + off], [r.ci_low, r.ci_high], color=col, lw=0.9)
            ax.plot([i + off], [r.delta_M0_minus_comparator], marker=mk, color=col, ms=3.2, mec="white", mew=0.4)
            rec.append(dict(panel=lab, comparator=mdl, delta=r.delta_M0_minus_comparator, lo=r.ci_low, hi=r.ci_high))
    ax.axhline(0, color=F.REF, lw=0.6, ls="--")
    ax.axhline(-0.003, color=F.C_RIDGE, lw=0.6, ls=":")
    panel_ticks(ax); ax.set_ylim(-0.0115, 0.0115)
    ax.set_yticks([-0.01, -0.005, 0, 0.005, 0.01]); ax.set_yticklabels(["−0.010", "−0.005", "0", "0.005", "0.010"])
    ax.set_ylabel("ΔHarrell's C, frozen − comparator")
    F.style_axis(ax, "y")
    src("Fig3", "b", pd.DataFrame(rec))


def f3c(B):
    B.title("Random input retention")
    r = DS.rich()
    lab = {"M1_nomask": ("vs no masking", F.C_AGESEX, "D"), "M3_XGB_raw": ("vs XGBoost-Cox", F.C_RIDGE, "s"),
           "M3_XGB_aug": ("vs XGBoost-Cox + masking", F.C_DNAM, "^")}
    B.legend([F.handle(v[1], v[2], ms=3.0) for v in lab.values()], [v[0] for v in lab.values()])
    ax = B.plot(14)
    for k, (l, c, mk) in lab.items():
        d = r[r.comparator == k].sort_values("available_assays")
        ax.fill_between(d.available_assays, d["min"], d["max"], color=c, alpha=0.18, lw=0)
        ax.plot(d.available_assays, d["mean"], color=c, marker=mk, ms=3.0, lw=1.0, mec="white", mew=0.4)
    ax.axhline(0, color=F.REF, lw=0.6, ls="--")
    ax.set_xticks([1, 3, 6, 12, 18, 22]); ax.set_xlim(0, 23)
    ax.set_xlabel("Randomly retained inputs per participant"); ax.set_ylabel("ΔHarrell's C, frozen − comparator")
    F.style_axis(ax, "y")
    src("Fig3", "c", r)


def f3d(B):
    B.title("Pooled comparison in four community cohorts")
    re = DS.random_effects().set_index("model")
    rows = [("age_sex", "Age + sex Cox"), ("ridge_frozen", "Ridge Cox"), ("ridge_panel", "Ridge Cox, panel refit"),
            ("M6_MICE", "Ridge Cox + imputation"), ("M5_MLP", "MLP + missing indicators"),
            ("M4_RSF", "Random survival forest"), ("M3_XGB_raw", "XGBoost-Cox"),
            ("M3_XGB_aug", "XGBoost-Cox + masking"), ("M1_nomask", "Transformer, no masking"),
            ("M2_panel", "Transformer, panel-specific")]
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_CLOCK, mfc="white")], ["95% CI excludes 0", "95% CI includes 0"])
    ax = B.plot(35)
    rec = []
    for i, (k, l) in enumerate(rows):
        r = re.loc[k]
        sig = r.low > 0 or r.high < 0
        F.ci_dot(ax, i, r.delta, r.low, r.high, F.C_CLOCK, mfc=F.C_CLOCK if sig else "white", ms=3.6)
        rec.append(dict(comparator=l, delta=r.delta, lo=r.low, hi=r.high, I2=r.I2, k=r.k))
    ax.axvline(0, color=F.REF, lw=0.6, ls="--")
    yrows(ax, [r[1] for r in rows])
    ax.tick_params(axis="y", labelsize=5.7)
    ax.set_xlim(-0.016, 0.07); ax.set_xticks([0, 0.02, 0.04, 0.06])
    ax.set_xlabel("Pooled ΔHarrell's C (95% CI)")
    F.style_axis(ax, "x")
    src("Fig3", "d", pd.DataFrame(rec))


def fig3():
    H2 = 91.5
    panels = [("a", f3a, 0, 0, H2, 66, dict(top=12.5, bottom=13.5)), ("b", f3b, H2, 0, H2, 66, dict(top=12.5, bottom=13.5)),
              ("c", f3c, 0, 67, H2, 66, LEG), ("d", f3d, H2, 67, H2, 66, LEG)]
    F.build(panels, 133, f"{FIGDIR}/Fig3", f"{PANDIR}/Fig3", "Fig3")
    write_source("Fig3")


# ================================================================ FIGURE 4
DNAM_LAB = {"HorvathAge": "Horvath", "HannumAge": "Hannum", "SkinBloodAge": "Skin & blood",
            "PhenoAge": "DNAm PhenoAge", "ZhangAge": "Zhang", "LinAge": "Lin", "GrimAgeMort": "GrimAge",
            "GrimAge2Mort": "GrimAge2", "DunedinPoAm": "DunedinPoAm", "HorvathTelo": "DNAm telomere length",
            "clinical": "Clinical clock", "age_sex": "Age + sex", "joint_GrimAge2": "Clinical + GrimAge2"}
HRS_LAB = {"clock_Horvath": "Horvath", "clock_Hannum": "Hannum", "clock_DNAmPhenoAge": "DNAm PhenoAge",
           "clock_SkinBlood": "Skin & blood", "clock_Lin": "Lin", "clock_Weidner": "Weidner",
           "clock_VidalBralo": "Vidal-Bralo", "clock_Yang": "Yang (epiTOC)", "clock_Zhang_2017": "Zhang 2017",
           "clock_Bocklandt": "Bocklandt", "clock_Garagnani": "Garagnani", "clock_GrimAge": "GrimAge",
           "clock_DunedinPoAm": "DunedinPoAm", "clock_DunedinPACE": "DunedinPACE", "clock_GrimAge2": "GrimAge2"}


def mcol(k):
    return F.C_CLOCK if k == "clinical" else F.C_JOINT if k.startswith("joint") else F.C_AGESEX if k == "age_sex" else F.C_DNAM


def dnam_legend(B):
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_DNAM), F.handle(F.C_JOINT)], ["Clinical", "DNAm", "Joint"])


def f4a(B):
    B.title("NHANES DNAm subset (2,525; 1,359 deaths)")
    d = DS.dnam_nhanes(); d = d[d.analysis == "primary"].sort_values("C", ascending=False)
    dnam_legend(B)
    ax = B.plot(28)
    for i, r in enumerate(d.itertuples()):
        F.ci_dot(ax, i, r.C, r.C_lower, r.C_upper, mcol(r.model), ms=3.4, lw=0.8)
    yrows(ax, [DNAM_LAB[m] for m in d.model])
    ax.tick_params(axis="y", labelsize=5.7)
    ax.set_xlim(0.715, 0.81); ax.set_xticks([0.72, 0.74, 0.76, 0.78, 0.80])
    ax.set_xlabel("Harrell's C, age/sex-adjusted (95% CI)")
    F.style_axis(ax, "x")
    src("Fig4", "a", d)


def f4b(B):
    B.title("HRS DNAm cohort (3,967; 732 deaths)")
    d = DS.hrs_disc(); d = d[d.analysis == "all_clocks_complete"]
    rows = [("Clinical + GrimAge2", float(d[d.clock == "clock_GrimAge2"].C_joint.iloc[0]), F.C_JOINT),
            ("Clinical clock", float(d.C_clinical.iloc[0]), F.C_CLOCK)]
    rows += [(HRS_LAB[r.clock], r.C_dnam, F.C_DNAM) for r in d.itertuples()]
    rows = sorted(rows, key=lambda x: -x[1])
    dnam_legend(B)
    ax = B.plot(28)
    for i, (l, v, c) in enumerate(rows):
        ax.plot([0.735, v], [i, i], color="#e6e6e6", lw=0.8, zorder=1)
        F.ci_dot(ax, i, v, None, None, c, ms=3.4)
    yrows(ax, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlim(0.735, 0.822); ax.set_xticks([0.74, 0.76, 0.78, 0.80, 0.82])
    ax.set_xlabel("Harrell's C, age/sex-adjusted")
    F.style_axis(ax, "x")
    src("Fig4", "b", pd.DataFrame(rows, columns=["model", "C", "colour"]).drop(columns="colour"))


def f4c(B):
    B.title("Clinical clock versus GrimAge2")
    d = DS.dnam_nhanes(); h = DS.hrs_disc()
    rows = []
    for a, l in [("primary", "Primary"), ("cells", "+ cell composition"), ("weighted", "Survey-weighted"),
                 ("weighted_cells", "Weighted + cells")]:
        r = d[(d.analysis == a) & (d.model == "GrimAge2Mort")].iloc[0]
        rows.append(("NHANES", l, r.clinical_minus_C, r.delta_lower, r.delta_upper))
    for a, l in [("primary", "Primary (prespecified)"), ("survey_design", "Survey-weighted")]:
        r = h[(h.analysis == a) & (h.clock == "clock_GrimAge2")].iloc[0]
        rows.append(("HRS", l, r.delta, r.lower, r.upper))
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_JOINT), F.handle(F.C_RIDGE, marker=None, ls=":")],
             ["NHANES", "HRS", "Non-inferiority margin"])
    ax = B.plot(32)
    for i, (c, l, e, lo, hi) in enumerate(rows):
        F.ci_dot(ax, i, e, lo, hi, F.C_CLOCK if c == "NHANES" else F.C_JOINT, ms=3.8)
    ax.axvline(0, color=F.REF, lw=0.6, ls="--")
    ax.axvline(-0.010, color=F.C_RIDGE, lw=0.8, ls=":")
    ax.axhline(3.5, color="#d0d0d0", lw=0.5)
    yrows(ax, [l for c, l, *_ in rows])
    ax.set_xlim(-0.014, 0.033); ax.set_xticks([-0.01, 0, 0.01, 0.02, 0.03])
    ax.set_xticklabels(["−0.01", "0", "0.01", "0.02", "0.03"])
    ax.set_xlabel("ΔHarrell's C, clinical − GrimAge2 (95% CI)")
    F.style_axis(ax, "x")
    src("Fig4", "c", pd.DataFrame(rows, columns=["cohort", "analysis", "delta", "lo", "hi"]).assign(
        noninferiority_margin=-0.010))


def two_score_legend(B):
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_DNAM, "s", ms=3.4)], ["Clinical clock", "GrimAge2"])


def f4d(B):
    B.title("Mortality hazard per SD")
    hn = DS.dnam_hr().set_index(["model", "term"]); hh = DS.hrs_hr()
    hh = hh[(hh.analysis == "primary")].set_index(["model", "term"])
    rows = [("NHANES\nseparate", hn.loc[("clinical", "full_accel")], hn.loc[("GrimAge2Mort", "GrimAge2Mort")]),
            ("NHANES\nmutually adjusted", hn.loc[("joint_GrimAge2", "full_accel")], hn.loc[("joint_GrimAge2", "GrimAge2Mort")]),
            ("HRS\nseparate", hh.loc[("clinical", "a")], hh.loc[("dnam", "b")]),
            ("HRS\nmutually adjusted", hh.loc[("joint", "a")], hh.loc[("joint", "b")])]
    two_score_legend(B)
    ax = B.plot(26)
    rec = []
    for i, (l, a, b) in enumerate(rows):
        F.ci_dot(ax, i - 0.14, a.HR, a.lower, a.upper, F.C_CLOCK)
        F.ci_dot(ax, i + 0.14, b.HR, b.lower, b.upper, F.C_DNAM, marker="s", ms=3.4)
        rec += [dict(row=l.replace("\n", " "), score="clinical", HR=a.HR, lo=a.lower, hi=a.upper),
                dict(row=l.replace("\n", " "), score="GrimAge2", HR=b.HR, lo=b.lower, hi=b.upper)]
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xlim(1.25, 3.5); ax.set_xticks([1.5, 2, 2.5, 3]); ax.set_xticklabels(["1.5", "2", "2.5", "3"])
    ax.axhline(1.5, color="#d0d0d0", lw=0.5)
    yrows(ax, [r[0] for r in rows])
    ax.set_xlabel("HR per SD (95% CI)")
    F.style_axis(ax, "x")
    src("Fig4", "d", pd.DataFrame(rec))


def f4e(B):
    B.title("HRS healthspan outcomes")
    h = DS.hrs_health(); h = h[h.clock == "clock_GrimAge2"]
    rows = [("adl", "cause_specific", "ADL disability\ncause-specific"), ("adl", "fine_gray", "ADL disability\nFine–Gray"),
            ("lw", "cause_specific", "Cognitive impairment\ncause-specific"),
            ("lw", "fine_gray", "Cognitive impairment\nFine–Gray")]
    two_score_legend(B)
    ax = B.plot(30)
    rec = []
    for i, (e, m, l) in enumerate(rows):
        a = h[(h.endpoint == e) & (h.model == m) & (h.term == "a")].iloc[0]
        b = h[(h.endpoint == e) & (h.model == m) & (h.term == "b")].iloc[0]
        F.ci_dot(ax, i - 0.14, a.ratio, a.lower, a.upper, F.C_CLOCK)
        F.ci_dot(ax, i + 0.14, b.ratio, b.lower, b.upper, F.C_DNAM, marker="s", ms=3.4)
        rec += [dict(row=l.replace("\n", " "), score=k, HR=x.ratio, lo=x.lower, hi=x.upper, n=x.n, events=x.events, q=x.q_BH)
                for k, x in (("clinical", a), ("GrimAge2", b))]
    ax.axvline(1, color=F.REF, lw=0.6, ls="--")
    ax.axhline(1.5, color="#d0d0d0", lw=0.5)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xlim(0.95, 1.7); ax.set_xticks([1, 1.2, 1.4, 1.6]); ax.set_xticklabels(["1.0", "1.2", "1.4", "1.6"])
    yrows(ax, [r[2] for r in rows])
    ax.set_xlabel("Mutually adjusted HR per SD (95% CI)")
    F.style_axis(ax, "x")
    src("Fig4", "e", pd.DataFrame(rec))


def f4f(B):
    B.title("Mutual information gain (NHANES)")
    l = DS.dnam_lrt()
    clocks = list(l.dnam.unique())
    B.legend([F.patch(F.C_CLOCK), F.patch(F.C_DNAM)], ["Clinical added to DNAm", "DNAm added to clinical"])
    ax = B.plot(28)
    rec = []
    for i, c in enumerate(clocks):
        a = l[(l.dnam == c) & (l.addition == "clinical")].iloc[0]
        b = l[(l.dnam == c) & (l.addition == c)].iloc[0]
        ax.barh(i - 0.19, a.LR, height=0.36, color=F.C_CLOCK, zorder=2)
        ax.barh(i + 0.19, b.LR, height=0.36, color=F.C_DNAM, zorder=2)
        rec.append(dict(dnam_clock=c, LR_add_clinical=a.LR, p_add_clinical=a.p, LR_add_dnam=b.LR, p_add_dnam=b.p))
    yrows(ax, [DNAM_LAB[c] for c in clocks])
    ax.tick_params(axis="y", labelsize=5.6)
    ax.set_xlim(0, 450); ax.set_xlabel("Likelihood-ratio χ² (1 d.f.)")
    F.style_axis(ax, "x")
    src("Fig4", "f", pd.DataFrame(rec))


def fig4():
    H2 = 91.5
    panels = [("a", f4a, 0, 0, H2, 70, LEG), ("b", f4b, H2, 0, H2, 70, LEG),
              ("c", f4c, 0, 71, H2, 56, LEG), ("d", f4d, H2, 71, H2, 56, LEG),
              ("e", f4e, 0, 128, H2, 58, LEG), ("f", f4f, H2, 128, H2, 58, LEG)]
    F.build(panels, 186, f"{FIGDIR}/Fig4", f"{PANDIR}/Fig4", "Fig4")
    write_source("Fig4")


# ================================================================ FIGURE 5
ICU_LAB = {"hospital24": "Clinical clock", "oasis": "OASIS", "sapsii": "SAPS-II", "sofa": "SOFA"}


def outcome_legend(B, c=F.INK2):
    B.legend([F.handle(c, "o", ls="none", mfc="white"), F.handle(c, "s", ls="none", ms=3.4)], ["In-hospital", "1-year"])


def f5a(B):
    B.title("MIMIC-IV, 24-h landmark")
    m = DS.mimic_main(); m = m[m.panel == "a"]
    B.legend([F.handle(F.C_CLOCK, ls="none"), F.handle(F.C_RIDGE, "s", ls="none", ms=3.4),
              F.handle(F.C_AGESEX, "D", ls="none", ms=3.2)], ["Clock", "Ridge Cox", "Age + sex"])
    ax = B.plot(19)
    for i, o in enumerate(["In-hospital", "365-day"]):
        for mdl, off, c, mk in [("Clinical clock", -0.22, F.C_CLOCK, "o"), ("Same-input ridge Cox", 0, F.C_RIDGE, "s"),
                                ("Age + sex Cox", 0.22, F.C_AGESEX, "D")]:
            r = m[(m.outcome == o) & (m.model == mdl)].iloc[0]
            F.ci_dot(ax, i + off, r.value, r.low, r.high, c, marker=mk, ms=3.4)
    yrows(ax, ["In-hospital\ndeath", "1-year\ndeath"], pad=0.55)
    ax.set_xlim(0.63, 0.84); ax.set_xticks([0.65, 0.70, 0.75, 0.80])
    ax.set_xlabel("AUC (95% CI)")
    F.style_axis(ax, "x")
    src("Fig5", "a", m)


def f5b(B):
    B.title("ICU: standalone AUC")
    r = DS.icu_raw(); r = r[r.population == "all_27005"]
    outcome_legend(B)
    ax = B.plot(19)
    rec = []
    for i, k in enumerate(["hospital24", "sapsii", "oasis", "sofa"]):
        c = F.C_CLOCK if k == "hospital24" else F.C_AGESEX
        for o, off, mk, fill in [("hospital_event", -0.15, "o", "white"), ("event", 0.15, "s", None)]:
            x = r[(r.score == k) & (r.outcome == o)].iloc[0]
            F.ci_dot(ax, i + off, x.AUC, x.lower, x.upper, c, marker=mk, ms=3.4, mfc=fill)
            rec.append(dict(score=ICU_LAB[k], outcome=o, AUC=x.AUC, lo=x.lower, hi=x.upper, n=x.n, events=x.events))
    yrows(ax, ["Clinical\nclock", "SAPS-II", "OASIS", "SOFA"])
    ax.set_xlim(0.65, 0.80); ax.set_xticks([0.66, 0.70, 0.74, 0.78])
    ax.set_xlabel("AUC (95% CI)")
    F.style_axis(ax, "x")
    src("Fig5", "b", pd.DataFrame(rec))


def f5c(B):
    B.title("ICU: value added by the clock")
    r = DS.icu_inc(); r = r[(r.population == "all_27005") & (r.clock == "hospital24")]
    outcome_legend(B, F.C_CLOCK)
    ax = B.plot(19)
    rec = []
    for i, k in enumerate(["sapsii", "oasis", "sofa"]):
        for o, off, mk, fill in [("hospital_event", -0.15, "o", "white"), ("event", 0.15, "s", None)]:
            x = r[(r.base_severity == k) & (r.outcome == o)].iloc[0]
            lo, hi = [float(v) for v in x.incremental_auc_oof_ci95.strip("[]").split(",")]
            F.ci_dot(ax, i + off, x.incremental_auc_oof, lo, hi, F.C_CLOCK, marker=mk, ms=3.4, mfc=fill)
            rec.append(dict(base=ICU_LAB[k], outcome=o, base_auc=x.baseline_auc_oof, aug_auc=x.augmented_auc_oof,
                            delta=x.incremental_auc_oof, lo=lo, hi=hi))
    ax.axvline(0, color=F.REF, lw=0.6, ls="--")
    yrows(ax, ["SAPS-II\n+ clock", "OASIS\n+ clock", "SOFA\n+ clock"])
    ax.set_xlim(-0.005, 0.125); ax.set_xticks([0, 0.04, 0.08, 0.12])
    ax.set_xlabel("ΔAUC, out-of-fold (95% CI)")
    F.style_axis(ax, "x")
    src("Fig5", "c", pd.DataFrame(rec))


def f5d(B):
    B.title("SAPS-II with prior history")
    m = DS.mimic_main(); m = m[m.panel == "c"]
    B.legend([F.handle(F.C_AGESEX, ls="none"), F.handle(F.C_CLOCK, ls="none")], ["SAPS-II", "+ clinical clock"])
    ax = B.plot(19)
    for i, o in enumerate(["In-hospital", "365-day"]):
        r = m[m.outcome == o].iloc[0]
        ax.annotate("", xy=(r.augmented_auc, i), xytext=(r.base_auc, i),
                    arrowprops=dict(arrowstyle="-|>", color="#9fb9d8", lw=1.0, shrinkA=3, shrinkB=3, mutation_scale=6))
        F.ci_dot(ax, i, r.base_auc, None, None, F.C_AGESEX, ms=4.2)
        F.ci_dot(ax, i, r.augmented_auc, None, None, F.C_CLOCK, ms=4.2)
        ax.text((r.base_auc + r.augmented_auc) / 2, i - 0.22, f"+{r.value:.3f}", ha="center", va="bottom",
                fontsize=F.FS_ANN, color=F.INK)
    yrows(ax, ["In-hospital\n(834 deaths)", "1-year\n(2,463 deaths)"], pad=0.55)
    ax.set_xlim(0.70, 0.80); ax.set_xticks([0.70, 0.73, 0.76, 0.79])
    ax.set_xlabel("Out-of-fold AUC")
    F.style_axis(ax, "x")
    src("Fig5", "d", m)


def f5e(B):
    B.title("System scores in hospital")
    p = DS.mimic_panels()
    outcome_legend(B)
    ax = B.plot(22)
    order = ["full", "immune_heme", "kidney", "liver", "metabolic", "cardiovascular"]
    rec = []
    for i, k in enumerate(order):
        for e, off, mk, fill in [("index_hospital_mortality", -0.15, "o", "white"),
                                 ("post_landmark_365d_mortality", 0.15, "s", None)]:
            x = p[(p.endpoint == e) & (p.panel == k)].iloc[0]
            F.ci_dot(ax, i + off, x.frozen_panel_auc, x.frozen_panel_auc_ci95_lower, x.frozen_panel_auc_ci95_upper,
                     F.ORGAN_C[k], marker=mk, ms=3.3, mfc=fill)
            rec.append(dict(panel=k, endpoint=e, n=x.n_compared, deaths=x.deaths, auc=x.frozen_panel_auc,
                            lo=x.frozen_panel_auc_ci95_lower, hi=x.frozen_panel_auc_ci95_upper))
    yrows(ax, [F.ORGAN_L[k] for k in order])
    ax.tick_params(axis="y", labelsize=5.7)
    ax.set_xlim(0.6, 0.83); ax.set_xticks([0.6, 0.65, 0.7, 0.75, 0.8])
    ax.set_xlabel("AUC, score alone (95% CI)")
    F.style_axis(ax, "x")
    src("Fig5", "e", pd.DataFrame(rec))


def f5f(B):
    B.title("Subgroups, 1-year death")
    sg = DS.subgroups(); sg = sg[sg.cohort == "mimic"]
    rows = [("all", "All"), ("female", "Women"), ("male", "Men"), ("age20_49", "20–49 y"),
            ("age50_64", "50–64 y"), ("age65_79", "65–79 y"), ("age80plus", "≥80 y")]
    ax = B.plot(17)
    for i, (g, l) in enumerate(rows):
        r = sg[sg.group == g].iloc[0]
        c = F.C_FEMALE if g == "female" else (F.C_MALE if g == "male" else F.C_CLOCK)
        F.ci_dot(ax, i, r.harrell_C, r.harrell_C_low, r.harrell_C_high, c, ms=3.6)
    ax.axhline(2.5, color="#d0d0d0", lw=0.5)
    yrows(ax, [r[1] for r in rows])
    ax.set_xlim(0.64, 0.85); ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_xlabel("Harrell's C (95% CI)")
    F.style_axis(ax, "x")
    src("Fig5", "f", sg)


def fig5():
    T = 61
    panels = [("a", f5a, 0, 0, T, 62, LEG), ("b", f5b, T, 0, T, 62, LEG), ("c", f5c, 2 * T, 0, T, 62, LEG),
              ("d", f5d, 0, 63, T, 62, LEG), ("e", f5e, T, 63, T, 62, LEG), ("f", f5f, 2 * T, 63, T, 62, LEG)]
    F.build(panels, 125, f"{FIGDIR}/Fig5", f"{PANDIR}/Fig5", "Fig5")
    write_source("Fig5")


# ================================================================ FIGURE 6
ORG5 = ["liver", "kidney", "metabolic", "immune_heme", "cardiovascular"]
CAUSE_L = {"all_cause": "All-cause", "cardiovascular": "Cardiovascular", "diabetes": "Diabetes",
           "renal": "Renal", "influenza_pneumonia": "Influenza/pneumonia", "malignant_neoplasm": "Cancer",
           "accidents": "Accidents"}
MATCH = {"cardiovascular": "cardiovascular", "diabetes": "metabolic", "renal": "kidney",
         "influenza_pneumonia": "immune_heme"}
SEQ = LinearSegmentedColormap.from_list("s", ["#f4f8fd", "#a7c9f0", "#2a78d6", "#123f7a"])
DIV = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f3f3f3", "#d43c3b"])


def heat_axes(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def f6a(B):
    B.title("Correlation between scores")
    c = DS.derived("organ_spearman_holdout.csv", index_col=0)
    order = ["full"] + ORG5
    c = c.loc[order, order]
    ax = B.plot(20, right=2)
    n = len(order)
    for i in range(n):
        for j in range(i + 1):
            v = c.iloc[i, j]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=SEQ((v - 0.3) / 0.7), ec="white", lw=1.0))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.3, color="white" if v > 0.62 else F.INK)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([F.ORGAN_L[k] for k in order], fontsize=5.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticklabels([F.ORGAN_L[k] for k in order], fontsize=5.5)
    heat_axes(ax)
    src("Fig6", "a", c.reset_index())


def f6b(B):
    B.title("NHANES III cause-specific mortality by system score")
    m = DS.cause_matrix()
    causes = ["diabetes", "renal", "cardiovascular", "influenza_pneumonia", "malignant_neoplasm", "accidents", "all_cause"]
    ax = B.plot(27, right=17)
    norm = TwoSlopeNorm(vmin=np.log(0.55), vcenter=0, vmax=np.log(2.0))
    rec = []
    for i, cz in enumerate(causes):
        for j, o in enumerate(ORG5):
            r = m[(m.endpoint == cz) & (m.panel == o)].iloc[0]
            v = np.log(r.hr)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=DIV(norm(v)), ec="white", lw=1.0))
            ax.text(j, i, f"{r.hr:.2f}" + ("*" if r.exploratory_matrix_fdr < 0.05 else ""), ha="center", va="center",
                    fontsize=5.4, color="white" if abs(v) > 0.45 else F.INK)
            if MATCH.get(cz) == o:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fc="none", ec=F.INK, lw=0.9, zorder=5))
            rec.append(dict(cause=cz, panel=o, n=r.n, deaths=r.deaths, HR=r.hr, lo=r.lo, hi=r.hi,
                            fdr=r.exploratory_matrix_fdr, matched=MATCH.get(cz) == o))
    ax.set_xlim(-0.5, len(ORG5) - 0.5); ax.set_ylim(len(causes) - 0.5, -0.5)
    ax.set_xticks(range(len(ORG5)))
    ax.set_xticklabels([F.ORGAN_L[o] for o in ORG5], fontsize=5.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(causes))); ax.set_yticklabels([CAUSE_L[c] for c in causes], fontsize=5.6)
    heat_axes(ax)
    cax = B.ax(B.w - 12.5, B.top + 3, 1.8, B.h - B.top - B.bottom - 6)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax)
    ticks = [0.6, 0.8, 1.0, 1.4, 2.0]
    cb.set_ticks(np.log(ticks)); cb.set_ticklabels([str(t) for t in ticks])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.4, length=1.5, width=0.4)
    cb.set_label("HR per SD", fontsize=5.8, labelpad=1.5)
    src("Fig6", "b", pd.DataFrame(rec))


def f6c(B):
    B.title("Serum cystatin C (NHANES III, ≥60 y)")
    e = DS.organ_effects(); e = e[e.panel == "a"]
    order = ["kidney", "liver", "metabolic", "immune_heme", "cardiovascular"]
    ax = B.plot(22)
    for i, o in enumerate(order):
        r = e[e.organ == o].iloc[0]
        F.ci_dot(ax, i, r.value, r.low, r.high, F.ORGAN_C[o], ms=4.0)
    ax.axvline(0, color=F.REF, lw=0.6, ls="--")
    yrows(ax, [F.ORGAN_L[o] for o in order])
    ax.set_xlim(-6, 15); ax.set_xticks([-5, 0, 5, 10, 15]); ax.set_xticklabels(["−5", "0", "5", "10", "15"])
    ax.set_xlabel("Difference per SD higher score (%, 95% CI)")
    F.style_axis(ax, "x")
    src("Fig6", "c", e)


def f6d(B):
    B.title("HRS: measured and DNAm-derived references")
    v = DS.hrs_vbs_ref(); v = v[(v.group == "clock_GrimAge2") & (v.target == "cystatin")]
    p = DS.hrs_proxy(); p = p[p.marker == "proxy_loga1c"]
    rows = []
    for o in ["kidney", "metabolic", "liver", "immune_heme"]:
        r = v[v.panel == o].iloc[0]; rows.append(("Cystatin C", o, r.partial_r, r.ci95_low, r.ci95_high, o == "kidney"))
    for o in ["metabolic", "kidney", "liver", "immune_heme"]:
        r = p[p.panel == o].iloc[0]; rows.append(("DNAm HbA1c", o, r.partial_r, r.ci95_low, r.ci95_high, o == "metabolic"))
    B.legend([F.handle(F.INK2), F.handle(F.INK2, mfc="white")], ["Matched system", "Other systems"])
    ax = B.plot(31)
    for i, (t, o, e, lo, hi, mt) in enumerate(rows):
        F.ci_dot(ax, i, e, lo, hi, F.ORGAN_C[o], ms=4.0 if mt else 3.2, mfc=F.ORGAN_C[o] if mt else "white")
    ax.axvline(0, color=F.REF, lw=0.6, ls="--")
    ax.axhline(3.5, color="#d0d0d0", lw=0.5)
    yrows(ax, [f"{t} · {F.ORGAN_L[o]}" for t, o, *_ in rows])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlim(-0.22, 0.4); ax.set_xticks([-0.2, 0, 0.2, 0.4]); ax.set_xticklabels(["−0.2", "0", "0.2", "0.4"])
    ax.set_xlabel("Partial correlation (95% CI)")
    F.style_axis(ax, "x")
    src("Fig6", "d", pd.DataFrame(rows, columns=["reference", "panel", "partial_r", "lo", "hi", "matched"]))


def f6e(B):
    B.title("Matched score added to whole-body model")
    e = DS.organ_effects(); e = e[e.panel == "b"]
    lab = {"Diabetes death": "Diabetes death", "Stroke/heart death": "Cardiovascular death",
           "Renal death": "Renal death", "Influenza/pneumonia": "Influenza/pneumonia death",
           "Liver steatosis": "Hepatic steatosis", "Renal readmission": "Renal readmission*"}
    ax = B.plot(31)
    for i, r in enumerate(e.itertuples()):
        F.ci_dot(ax, i, r.value, r.low, r.high, F.ORGAN_C[r.organ], ms=4.0)
    ax.axvline(0, color=F.REF, lw=0.6, ls="--")
    ax.axhline(4.5, color="#d0d0d0", lw=0.5)
    yrows(ax, [f"{lab[x]} ({int(ev):,})" for x, ev in zip(e.endpoint, e.events)])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlim(-0.016, 0.047); ax.set_xticks([-0.01, 0, 0.01, 0.02, 0.03, 0.04])
    ax.set_xticklabels(["−0.01", "0", "0.01", "0.02", "0.03", "0.04"])
    ax.set_xlabel("Out-of-fold ΔAUC (95% CI)")
    F.style_axis(ax, "x")
    src("Fig6", "e", e)


def f6f(B):
    B.title("Incident disease in community cohorts")
    d = DS.disease_diag()
    lab = {"diabe": "diabetes", "hearte": "heart disease", "stroke": "stroke", "kidneye": "kidney disease"}
    d = d.assign(label=[f"{c.upper()} · {lab[e]}" for c, e in zip(d.cohort, d.endpoint)]).sort_values("q_diagonal")
    B.legend([F.handle(F.INK2, "o", ls="none"), F.handle(F.INK2, "o", ls="none", mfc="white")],
             ["Matched score", "Strongest other"])
    ax = B.plot(28, right=13)
    for i, r in enumerate(d.itertuples()):
        ax.plot([r.strongest_other_hr, r.matched_hr], [i, i], color="#dadada", lw=0.8, zorder=1)
        F.ci_dot(ax, i, r.strongest_other_hr, None, None, F.INK2, mfc="white", ms=3.4)
        F.ci_dot(ax, i, r.matched_hr, None, None, F.ORGAN_C[r.matched], ms=4.0)
        qtxt = "<0.001" if r.q_diagonal < 0.001 else f"{r.q_diagonal:.3f}"
        B.text(B.w - 1.0, B.rowy(i, len(d)), f"q {qtxt}", ha="right", va="center", fontsize=5.3, color=F.INK2)
    ax.axvline(1, color=F.REF, lw=0.6, ls="--")
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xlim(0.65, 3.4); ax.set_xticks([0.75, 1, 1.5, 2, 3]); ax.set_xticklabels(["0.75", "1", "1.5", "2", "3"])
    yrows(ax, list(d.label))
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlabel("HR per SD")
    F.style_axis(ax, "x")
    src("Fig6", "f", d)


def fig6():
    H2 = 91.5
    heat = dict(top=8.0, bottom=13.0)
    panels = [("a", f6a, 0, 0, 72, 64, heat), ("b", f6b, 72, 0, 111, 64, heat),
              ("c", f6c, 0, 65, H2, 58, LEG), ("d", f6d, H2, 65, H2, 58, LEG),
              ("e", f6e, 0, 124, H2, 58, LEG), ("f", f6f, H2, 124, H2, 58, LEG)]
    F.build(panels, 182, f"{FIGDIR}/Fig6", f"{PANDIR}/Fig6", "Fig6")
    write_source("Fig6")


if __name__ == "__main__":
    for w in sys.argv[1:]:
        globals()[f"fig{w}"]()
        print("built", w)
