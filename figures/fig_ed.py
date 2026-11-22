import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch
import figlib as F
import data_src as DS
from fig_main import SOURCE, yrows, OUT
from fig_main_b import COH, DNAM_LAB, HRS_LAB, ORG5

EDDIR = f"{OUT}/06_Extended_Data"
EDPAN = f"{OUT}/06_Extended_Data/Panels"
SRCDIR = f"{OUT}/08_Source_Data"


def src(fig, panel, df):
    SOURCE.setdefault(fig, {})[panel] = df.reset_index(drop=True)


def write_source(fig):
    with pd.ExcelWriter(f"{SRCDIR}/Source_Data_{fig}.xlsx") as xw:
        for p, df in SOURCE.get(fig, {}).items():
            df.to_excel(xw, sheet_name=f"{p}"[:31], index=False)


def build(fid, panels, H):
    F.build(panels, H, f"{EDDIR}/{fid}", f"{EDPAN}/{fid}", fid, fig_formats=("pdf", "png", "tiff"))
    write_source(fid)


# ------------------------------------------------------------ flow helper
def flow(B, title, steps, side=None):
    """steps: list of (text, kind) kind: 'main' or 'excl'; vertical flow chart."""
    B.title(title)
    ax = B.ax(1, 7, B.w - 2, B.h - 8)
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(steps)
    gap = 1.0 / n
    for i, (txt, kind) in enumerate(steps):
        yc = 1 - gap * (i + 0.5)
        h = gap * 0.68
        x0, w = (0.04, 0.92) if kind != "split" else (0.04, 0.92)
        fc = F.LIGHT if kind == "main" else ("#eef6f2" if kind == "out" else "#fbf3ee")
        ec = "#c5ced8" if kind == "main" else ("#9dcbb6" if kind == "out" else "#e5b8a0")
        F.rounded_box(ax, x0, yc - h / 2, w, h, fc, ec, lw=0.5)
        ax.text(0.5, yc, txt, ha="center", va="center", fontsize=5.6, color=F.INK, transform=ax.transAxes,
                linespacing=1.3, zorder=4)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((0.5, yc - h / 2), (0.5, yc - gap + gap * 0.34), transform=ax.transAxes,
                                         arrowstyle="-|>", mutation_scale=5, lw=0.7, color="#5f6b78"))


# =============================================================== ED Fig. 1
def e1a(B):
    flow(B, "NHANES development and holdout", [
        ("NHANES 1999–2018 adults ≥20 years\nwith linked mortality\nn = 50,808; 8,268 deaths", "main"),
        ("Excluded: <22 of 43 inputs measured\nn = 3,243", "excl"),
        ("Analytic sample\nn = 47,565; 7,538 deaths", "main"),
        ("Development, 1999–2008\nn = 19,426; 4,202 deaths", "out"),
        ("Temporal holdout, 2009–2018\nn = 25,614; 1,977 deaths", "out"),
        ("DNAm subset, 1999–2002\n(all 2,532 excluded from training)\nn = 2,525; 1,359 deaths", "out")])
    src("EDFig1", "a", pd.DataFrame({"step": ["initial", "analytic", "development", "holdout", "dnam"],
                                     "n": [50808, 47565, 19426, 25614, 2525], "deaths": [8268, 7538, 4202, 1977, 1359]}))


def e1b(B):
    flow(B, "MIMIC-IV v3.1 hospital cohort", [
        ("Patients with a hospital admission\nn = 223,452", "main"),
        ("Adults, first recorded admission,\nalive and in hospital at 24 h\nn = 165,661", "main"),
        ("Excluded: no eligible laboratory\nresult before the 24-h landmark\nn = 35,419", "excl"),
        ("Analytic cohort\nn = 130,242 (median 19 inputs)", "main"),
        ("In-hospital deaths 4,116\n365-day deaths 17,469", "out"),
        ("ICU subcohort n = 27,005\nPrior-history subcohort n = 9,213", "out")])
    src("EDFig1", "b", pd.DataFrame({"step": ["admitted", "eligible", "analytic", "icu", "prior_history"],
                                     "n": [223452, 165661, 130242, 27005, 9213]}))


def e1c(B):
    flow(B, "HRS 2016 DNAm head-to-head", [
        ("HRS 2016 venous blood study,\nclinical clock scorable\nn = 9,873", "main"),
        ("Matched to quality-controlled\nDNAm release\nn = 4,018", "main"),
        ("Excluded: age <50 years at draw\nor no valid follow-up\nn = 51", "excl"),
        ("Analytic sample\nn = 3,967; 3,926 households\n732 deaths", "main"),
        ("Incident ADL disability\nn = 3,261; 629 events", "out"),
        ("Incident cognitive impairment\nn = 3,095; 732 events", "out")])
    src("EDFig1", "c", pd.DataFrame({"step": ["scorable", "matched", "analytic", "adl", "cognition"],
                                     "n": [9873, 4018, 3967, 3261, 3095]}))


def ed1():
    build("EDFig1", [("a", e1a, 0, 0, 61, 112), ("b", e1b, 61, 0, 61, 112), ("c", e1c, 122, 0, 61, 112)], 112)


# =============================================================== ED Fig. 2
def e2a(B):
    B.title("FT-Mask architecture and training")
    ax = B.ax(1, 7, B.w - 2, B.h - 8)
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    blocks = [(0.00, "Inputs", "Age + up to 43\nroutine measures\n(log/z-scaled,\nclipped ±5 SD)"),
              (0.205, "Tokenizer", "Feature-specific\naffine embedding\n+ [CLS] token;\nunmeasured masked"),
              (0.41, "Encoder", "3 Transformer\nlayers, d = 64,\n4 heads,\ndropout 0.1"),
              (0.615, "Head", "[CLS] → log\nrelative hazard\n(Cox partial\nlikelihood)"),
              (0.82, "Ensemble", "Mean of 5 seeds;\nfrozen with\nreference scaling")]
    for x, h, t in blocks:
        F.rounded_box(ax, x, 0.36, 0.18, 0.56, F.LIGHT, "#c5ced8", lw=0.5)
        ax.add_patch(Rectangle((x, 0.885), 0.18, 0.035, transform=ax.transAxes, fc=F.C_CLOCK, ec="none", zorder=3))
        ax.text(x + 0.09, 0.83, h, ha="center", va="top", fontsize=6.0, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.09, 0.73, t, ha="center", va="top", fontsize=5.5, transform=ax.transAxes, linespacing=1.3)
        if x < 0.8:
            ax.add_patch(FancyArrowPatch((x + 0.182, 0.64), (x + 0.203, 0.64), transform=ax.transAxes,
                                         arrowstyle="-|>", mutation_scale=5, lw=0.7, color="#5f6b78"))
    # masking bar
    segs = [(0.40, "Full measured panel 40%", F.C_CLOCK), (0.35, "Random subset 35%", "#62a0e6"),
            (0.25, "Single organ system 25%", "#a7c9f0")]
    x = 0.0
    for w, t, c in segs:
        ax.add_patch(Rectangle((x, 0.05), w * 0.999, 0.14, transform=ax.transAxes, fc=c, ec="white", lw=1.0))
        ax.text(x + w / 2, 0.12, t, ha="center", va="center", fontsize=5.6,
                color="white" if c != "#a7c9f0" else F.INK, transform=ax.transAxes)
        x += w
    src("EDFig2", "a", pd.DataFrame({"masking_regime": ["full", "random_subset", "single_system"],
                                     "probability": [0.40, 0.35, 0.25]}))


def e2b(B):
    B.title("Nested cross-validation by survey cycle")
    c = DS.cv()["nested_outer"]
    d = pd.DataFrame(c)
    ax = B.ax(13, 9, B.w - 17, B.h - 19)
    ax.bar(range(len(d)), d.c, width=0.6, color=F.C_CLOCK, zorder=2)
    for i, r in d.iterrows():
        ax.text(i, r.c - 0.0008, f"{r.c:.3f}", ha="center", va="top", fontsize=F.FS_ANN, color="white")
    ax.axhline(DS.cv()["nested_macro_c"], color=F.INK2, ls="--", lw=0.6)
    ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{y}–{str(y+1)[2:]}" for y in d.year])
    ax.set_ylim(0.85, 0.892); ax.set_ylabel("Outer-fold Harrell's C"); ax.set_xlabel("Held-out NHANES cycle")
    F.style_axis(ax, "y")
    src("EDFig2", "b", d.assign(macro_C=DS.cv()["nested_macro_c"]))


def e2c(B):
    B.title("Input attributions by age group (holdout)")
    g = DS.ig(); g = g[g.cohort == "temporal"]
    piv = g.pivot_table(index="feature", columns="age_group", values="mean_abs_IG")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index[:20]]
    cols = [c for c in ["20–49", "50–64", "65–79", "80+"] if c in piv.columns]
    piv = piv[cols]
    ax = B.ax(22, 8, B.w - 36, B.h - 17)
    cmap = LinearSegmentedColormap.from_list("s", ["#f4f8fd", "#a7c9f0", "#2a78d6", "#123f7a"])
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    nice = {"rdw": "RDW", "crp": "CRP", "hba1c": "HbA1c", "uacr": "UACR", "bun": "BUN", "alp": "ALP", "ast": "AST",
            "alt": "ALT", "ggt": "GGT", "ldh": "LDH", "wbc": "WBC", "rbc": "RBC", "hgb": "Haemoglobin", "mcv": "MCV",
            "mch": "MCH", "mpv": "MPV", "hdl": "HDL cholesterol", "bmi": "BMI", "sbp": "Systolic BP",
            "dbp": "Diastolic BP", "lymph_pct": "Lymphocytes %", "mono_pct": "Monocytes %", "neut_pct": "Neutrophils %",
            "eos_pct": "Eosinophils %", "uric_acid": "Uric acid", "total_protein": "Total protein"}
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([nice.get(f, f.replace("_", " ").capitalize()) for f in piv.index],
                                                       fontsize=5.4)
    ax.set_xlabel("Age group (years)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = B.ax(B.w - 12.5, 14, 1.8, B.h - 32)
    cb = plt.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4); cb.set_label("Mean |IG|", fontsize=5.6, labelpad=2.5)
    src("EDFig2", "c", piv.reset_index())


def e2d(B):
    B.title("System contributions by age group (holdout)")
    s = DS.shapley(); s = s[s.cohort == "temporal"]
    ages = ["20–49", "50–64", "65–79", "80+"]
    ax = B.ax(13, 12.5, B.w - 17, B.h - 22)
    B.legend([F.handle(F.ORGAN_C[o]) for o in ORG5], [F.ORGAN_L[o] for o in ORG5], ncol=5, ty=6.4)
    for o in ORG5:
        d = s[s.panel == o].set_index("age_group").loc[ages]
        ax.plot(range(4), d.mean_abs, color=F.ORGAN_C[o], marker="o", ms=3.2, lw=1.0, mec="white", mew=0.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(ages); ax.set_xlabel("Age group (years)")
    ax.set_ylabel("Mean |Shapley value| (log hazard)")
    ax.set_ylim(0, None)
    F.style_axis(ax, "y")
    src("EDFig2", "d", s)


def ed2():
    build("EDFig2", [("a", e2a, 0, 0, 183, 50), ("b", e2b, 0, 51, 70, 66), ("c", e2c, 70, 51, 113, 66),
                     ("d", e2d, 0, 118, 183, 58)], 176)


# =============================================================== ED Fig. 3
def e3a(B):
    B.title("Mean predicted versus observed risk")
    c = DS.fig1c()
    sg = DS.subgroups(); m = sg[(sg.cohort == "mimic") & (sg.group == "all")].iloc[0]
    rows = [(COH[r.cohort] + " (3 y)", r.observed_KM, r.predicted) for r in c.itertuples()]
    rows.append(("MIMIC-IV (1 y)", m.observed_KM, m.mean_predicted))
    B.legend([F.handle(F.INK2, "o", ls="none"), F.handle(F.C_CLOCK, "s", ls="none", mfc="white")],
             ["Observed (Kaplan–Meier)", "Mean predicted (frozen)"])
    ax = B.ax(28, 12.5, B.w - 31, B.h - 21.5)
    for i, (l, o, p) in enumerate(rows):
        ax.plot([100 * o, 100 * p], [i, i], color="#cfd8e3", lw=1.2, zorder=1)
        F.ci_dot(ax, i, 100 * o, None, None, F.INK2, ms=3.8)
        F.ci_dot(ax, i, 100 * p, None, None, F.C_CLOCK, marker="s", ms=3.6, mfc="white")
    yrows(ax, [r[0] for r in rows])
    ax.set_xlim(0, 16); ax.set_xlabel("All-cause mortality (%)")
    F.style_axis(ax, "x")
    src("EDFig3", "a", pd.DataFrame(rows, columns=["cohort", "observed", "predicted"]))


def e3b(B):
    B.title("Expected/observed ratio by sex")
    sg = DS.subgroups()
    order = ["temporal", "nhanes3", "hrs", "charls", "elsa", "mimic"]
    B.legend([F.handle(F.C_FEMALE), F.handle(F.C_MALE, "s")], ["Women", "Men"])
    ax = B.ax(24, 12.5, B.w - 27, B.h - 21.5)
    rec = []
    for i, c in enumerate(order):
        for sx, off, col, mk in [("female", -0.15, F.C_FEMALE, "o"), ("male", 0.15, F.C_MALE, "s")]:
            r = sg[(sg.cohort == c) & (sg.group == sx)].iloc[0]
            F.ci_dot(ax, i + off, r.E_over_O, r.E_over_O_low, r.E_over_O_high, col, marker=mk, ms=3.4)
            rec.append(dict(cohort=c, sex=sx, E_O=r.E_over_O, lo=r.E_over_O_low, hi=r.E_over_O_high))
    ax.axvline(1, color=F.REF, ls="--", lw=0.6)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xlim(0.28, 2.6); ax.set_xticks([0.3, 0.5, 1, 2]); ax.set_xticklabels(["0.3", "0.5", "1", "2"])
    yrows(ax, [COH[c] for c in order])
    ax.set_xlabel("E/O ratio (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig3", "b", pd.DataFrame(rec))


def e3c(B):
    B.title("3-year E/O: frozen vs multicohort model")
    t = DS.risk_transport(); t = t[t.years == 3]
    order = [c for c in ["temporal", "nhanes3", "hrs", "charls", "elsa", "share", "klosa"] if c in set(t.target)]
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_JOINT, "s")], ["Frozen clock", "Multicohort retrained"])
    ax = B.ax(20, 12.5, B.w - 23, B.h - 21.5)
    rec = []
    for i, c in enumerate(order):
        a = t[(t.target == c) & (t.model == "v1")].iloc[0]; b = t[(t.target == c) & (t.model == "multi_mlp")].iloc[0]
        F.ci_dot(ax, i - 0.14, a.E_over_O, None, None, F.C_CLOCK)
        F.ci_dot(ax, i + 0.14, b.E_over_O, None, None, F.C_JOINT, marker="s", ms=3.4)
        rec.append(dict(target=c, E_O_frozen=a.E_over_O, E_O_multicohort=b.E_over_O, observed=a.observed_KM))
    ax.axvline(1, color=F.REF, ls="--", lw=0.6)
    yrows(ax, [COH[c] for c in order])
    ax.set_xlim(0.8, 2.3); ax.set_xticks([1.0, 1.5, 2.0]); ax.set_xlabel("3-year E/O ratio")
    F.style_axis(ax, "x")
    src("EDFig3", "c", pd.DataFrame(rec))


def ed3():
    build("EDFig3", [("a", e3a, 0, 0, 61, 66), ("b", e3b, 61, 0, 61, 66), ("c", e3c, 122, 0, 61, 66)], 66)


# =============================================================== ED Fig. 4
def e4a(B):
    B.title("Sparse-panel cohorts")
    s = DS.cohort_summary()
    rows = [("MHAS", "MHAS (10 inputs)"), ("SHARE", "SHARE (6, dried blood spots)"), ("KLoSA", "KLoSA (1, BMI only)")]
    B.legend([F.handle(F.C_CLOCK, ls="none"), F.handle(F.C_RIDGE, "s", ls="none"),
              F.handle(F.C_AGESEX, "D", ls="none", ms=3.2)], ["Clock", "Ridge Cox", "Age + sex"], ncol=3)
    ax = B.ax(33, 12.5, B.w - 36, B.h - 21.5)
    rec = []
    for i, (k, l) in enumerate(rows):
        r = s[s.cohort == k].iloc[0]
        F.ci_dot(ax, i - 0.18, r.c_clock, r.c_low, r.c_high, F.C_CLOCK)
        F.ci_dot(ax, i, r.c_ridge, None, None, F.C_RIDGE, marker="s", ms=3.4)
        F.ci_dot(ax, i + 0.18, r.c_agesex, None, None, F.C_AGESEX, marker="D", ms=3.2)
        rec.append(dict(cohort=k, n=r.survival_n, deaths=r.deaths, C_clock=r.c_clock, lo=r.c_low, hi=r.c_high,
                        C_ridge=r.c_ridge, C_age_sex=r.c_agesex))
    yrows(ax, [r[1] for r in rows])
    ax.set_xlim(0.72, 0.80); ax.set_xlabel("Harrell's C")
    F.style_axis(ax, "x")
    src("EDFig4", "a", pd.DataFrame(rec))


def e4b(B):
    B.title("Interval-censored death dates")
    d = DS.interval_sens(); d = d[d.cohort.isin(["charls", "elsa", "hrs"])]
    B.legend([F.handle(F.C_CLOCK, ls="none"), F.handle(F.C_RIDGE, "s", ls="none"),
              F.handle(F.C_AGESEX, "D", ls="none", ms=3.2)], ["Clock", "Ridge Cox", "Age + sex"], ncol=3)
    ax = B.ax(20, 12.5, B.w - 23, B.h - 21.5)
    for i, r in enumerate(d.itertuples()):
        F.ci_dot(ax, i - 0.18, r.C_clock, None, None, F.C_CLOCK)
        F.ci_dot(ax, i, r.C_ridge_cox, None, None, F.C_RIDGE, marker="s", ms=3.4)
        F.ci_dot(ax, i + 0.18, r.C_age_sex, None, None, F.C_AGESEX, marker="D", ms=3.2)
    yrows(ax, [COH[c] for c in d.cohort])
    ax.set_xlim(0.73, 0.83); ax.set_xlabel("Harrell's C, unambiguous pairs only")
    F.style_axis(ax, "x")
    src("EDFig4", "b", d)


def e4c(B):
    B.title("NHANES III assay-platform masking")
    j = DS.assay_mask()
    rows = [("Original (40 inputs)", j["scenarios"]["crp_omitted"]["C_original"], None, None)]
    lab = {"crp_omitted": "CRP masked", "manual_diff_omitted": "Manual differential masked",
           "manual_diff_and_crp_omitted": "Both masked"}
    for k, l in lab.items():
        v = j["scenarios"][k]
        rows.append((l, v["C_masked"], v["C_masked_CI95"][0], v["C_masked_CI95"][1]))
    ax = B.ax(31, 8.5, B.w - 34, B.h - 17.5)
    for i, (l, c, lo, hi) in enumerate(rows):
        F.ci_dot(ax, i, c, lo, hi, F.C_CLOCK, ms=3.8)
    yrows(ax, [r[0] for r in rows])
    ax.set_xlim(0.852, 0.865); ax.set_xticks([0.854, 0.858, 0.862]); ax.set_xlabel("Harrell's C (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig4", "c", pd.DataFrame(rows, columns=["scenario", "C", "lo", "hi"]))


def e4d(B):
    B.title("CHARLS survey-design sensitivity")
    w = DS.charls_weights()
    w = w[(w.model != "full_lp")]
    lab = {"all": "No strata", "province": "28 provinces", "city_nested": "126 cities"}
    B.legend([F.handle(F.C_RIDGE, "s"), F.handle(F.C_AGESEX, "D", ms=3.2)], ["vs ridge Cox", "vs age + sex"])
    ax = B.ax(31, 12.5, B.w - 34, B.h - 21.5)
    rows = []
    for st in ["all", "province", "city_nested"]:
        for met in ["Harrell", "Uno_3y"]:
            rows.append((st, met))
    for i, (st, met) in enumerate(rows):
        for mdl, off, c, mk in [("ridge_lp", -0.15, F.C_RIDGE, "s"), ("age_sex_lp", 0.15, F.C_AGESEX, "D")]:
            r = w[(w.strata_approximation == st) & (w.metric == met) & (w.model == mdl)]
            if len(r):
                r = r.iloc[0]
                F.ci_dot(ax, i + off, r.clock_minus_model, r.delta_lo, r.delta_hi, c, marker=mk, ms=3.2)
    ax.axvline(0, color=F.REF, ls="--", lw=0.6)
    yrows(ax, [f"{lab.get(st, st)} · {'Harrell C' if m == 'Harrell' else '3-y Uno C'}" for st, m in rows])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlim(-0.004, 0.035); ax.set_xlabel("Weighted ΔC, clock − comparator (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig4", "d", w)


def ed4():
    build("EDFig4", [("a", e4a, 0, 0, 91.5, 54), ("b", e4b, 91.5, 0, 91.5, 54),
                     ("c", e4c, 0, 55, 91.5, 54), ("d", e4d, 91.5, 55, 91.5, 62)], 117)


# =============================================================== ED Fig. 5
AGES = [("age20_49", "20–49"), ("age50_64", "50–64"), ("age65_79", "65–79"), ("age80plus", "≥80")]
COH6 = ["temporal", "nhanes3", "hrs", "charls", "elsa", "mimic"]
COHC = {"temporal": "#2a78d6", "nhanes3": "#123f7a", "hrs": "#eb6834", "charls": "#c98500", "elsa": "#199e70",
        "mimic": "#6250d6"}


def age_panel(B, col, lo, hi, title, ylab, ref=None, log=False):
    B.title(title)
    sg = DS.subgroups()
    B.legend([F.handle(COHC[c]) for c in COH6], [COH[c] for c in COH6], ncol=6, ty=6.4)
    ax = B.ax(14, 12.5, B.w - 18, B.h - 22)
    rec = []
    for j, c in enumerate(COH6):
        xs, ys, l, h = [], [], [], []
        for i, (g, gl) in enumerate(AGES):
            r = sg[(sg.cohort == c) & (sg.group == g)]
            if not len(r) or r.iloc[0].status != "estimated":
                continue
            r = r.iloc[0]; off = (j - 2.5) * 0.1
            xs.append(i + off); ys.append(r[col]); l.append(r[lo]); h.append(r[hi])
            rec.append(dict(cohort=c, age=gl, n=r.n, value=r[col], lo=r[lo], hi=r[hi]))
        for x, y, a, b in zip(xs, ys, l, h):
            ax.plot([x, x], [a, b], color=COHC[c], lw=0.8)
        ax.plot(xs, ys, color=COHC[c], marker="o", ms=3.0, lw=0.8, mec="white", mew=0.4)
    if ref is not None:
        ax.axhline(ref, color=F.REF, ls="--", lw=0.6)
    if log:
        ax.set_yscale("log"); ax.minorticks_off(); ax.set_yticks([0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.2", "0.5", "1", "2"])
    ax.set_xticks(range(4)); ax.set_xticklabels([a[1] for a in AGES]); ax.set_xlabel("Age group (years)")
    ax.set_ylabel(ylab)
    F.style_axis(ax, "y")
    return pd.DataFrame(rec)


def e5a(B):
    src("EDFig5", "a", age_panel(B, "harrell_C", "harrell_C_low", "harrell_C_high", "Discrimination within age strata",
                                 "Harrell's C (95% CI)", ref=0.5))


def e5b(B):
    src("EDFig5", "b", age_panel(B, "E_over_O", "E_over_O_low", "E_over_O_high", "Calibration within age strata",
                                 "E/O ratio (95% CI)", ref=1, log=True))


def ed5():
    build("EDFig5", [("a", e5a, 0, 0, 183, 64), ("b", e5b, 0, 65, 183, 64)], 129)


# =============================================================== ED Fig. 6
def profile(B, panel):
    B.title(F.ORGAN_L[panel] if panel != "full" else "Whole-body clock")
    p = DS.derived("organ_sex_age_profiles.csv"); p = p[p.panel == panel]
    f = p[p.sex == "Female"].set_index("age_mid"); m = p[p.sex == "Male"].set_index("age_mid")
    ages = sorted(set(f.index) & set(m.index))
    d = pd.DataFrame({"age_mid": ages})
    d["diff"] = [m.loc[a, "mean"] - f.loc[a, "mean"] for a in ages]
    se = [np.sqrt(((m.loc[a, "hi"] - m.loc[a, "mean"]) / 1.96) ** 2 + ((f.loc[a, "hi"] - f.loc[a, "mean"]) / 1.96) ** 2)
          for a in ages]
    d["lo"] = d["diff"] - 1.96 * np.array(se); d["hi"] = d["diff"] + 1.96 * np.array(se)
    d["n_women"] = [int(f.loc[a, "n"]) for a in ages]; d["n_men"] = [int(m.loc[a, "n"]) for a in ages]
    ax = B.ax(12, 8.5, B.w - 15, B.h - 18)
    c = F.ORGAN_C[panel]
    ax.fill_between(d.age_mid, d.lo, d.hi, color=c, alpha=0.18, lw=0)
    ax.plot(d.age_mid, d["diff"], color=c, marker="o", ms=3.0, lw=1.1, mec="white", mew=0.4)
    ax.axhline(0, color=F.REF, ls="--", lw=0.6)
    ax.set_ylim(-6, 7); ax.set_xticks([25, 45, 65, 85]); ax.set_xlim(21, 89)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Men − women (score points)")
    F.style_axis(ax, "y")
    return d


def ed6():
    def mk(panel):
        def f(B):
            src("EDFig6", panel, profile(B, panel))
        return f
    order = ["full"] + ORG5
    panels = []
    for k, pnl in enumerate(order):
        r, c = divmod(k, 3)
        panels.append(("abcdef"[k], mk(pnl), c * 61, r * 56, 61, 55))
    build("EDFig6", panels, 111)


# =============================================================== ED Fig. 7
def e7a(B):
    B.title("NHANES: clinical clock minus each DNAm clock")
    d = DS.dnam_nhanes()
    clocks = ["GrimAge2Mort", "GrimAgeMort", "DunedinPoAm", "PhenoAge", "HorvathTelo", "HannumAge", "HorvathAge",
              "LinAge", "ZhangAge", "SkinBloodAge"]
    an = [("primary", "Primary"), ("cells", "+ cells"), ("weighted", "Weighted"), ("weighted_cells", "Weighted\n+ cells")]
    M = np.array([[d[(d.analysis == a) & (d.model == c)].clinical_minus_C.iloc[0] for a, _ in an] for c in clocks])
    L = np.array([[d[(d.analysis == a) & (d.model == c)].delta_lower.iloc[0] for a, _ in an] for c in clocks])
    ax = B.ax(28, 8, B.w - 42, B.h - 19)
    cmap = LinearSegmentedColormap.from_list("s", ["#f4f8fd", "#a7c9f0", "#2a78d6", "#123f7a"])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=0.05)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:+.3f}" + ("" if L[i, j] > 0 else "†"), ha="center", va="center", fontsize=5.3,
                    color="white" if M[i, j] > 0.03 else F.INK)
    ax.set_xticks(range(len(an))); ax.set_xticklabels([a[1] for a in an], fontsize=5.6)
    ax.set_yticks(range(len(clocks))); ax.set_yticklabels([DNAM_LAB[c] for c in clocks], fontsize=5.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = B.ax(B.w - 11, 12, 1.8, B.h - 30)
    cb = plt.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4); cb.set_label("ΔHarrell's C", fontsize=5.6, labelpad=1.5)
    src("EDFig7", "a", pd.DataFrame(M, index=[DNAM_LAB[c] for c in clocks], columns=[a[0] for a in an]).reset_index())


def e7b(B):
    B.title("HRS: prespecified sensitivity analyses")
    h = DS.hrs_disc(); h = h[h.clock == "clock_GrimAge2"]
    lab = {"primary": "Primary", "one_year_landmark": "1-year landmark", "age_cap_80": "Age capped at 80",
           "age_cap_85": "Age capped at 85", "time_lower_yr": "Death date, interval start",
           "time_upper_yr": "Death date, interval end", "cell_composition": "+ cell composition",
           "survey_design": "Survey-weighted"}
    h = h[h.analysis.isin(lab)].drop_duplicates("analysis")
    B.legend([F.handle(F.C_CLOCK, ls="none"), F.handle(F.C_DNAM, "s", ls="none"), F.handle(F.C_JOINT, "D", ls="none", ms=3.2)],
             ["Clinical", "GrimAge2", "Joint"], ncol=3)
    ax = B.ax(33, 12.5, B.w - 36, B.h - 21.5)
    for i, r in enumerate(h.itertuples()):
        ax.plot([r.C_dnam, r.C_joint], [i, i], color="#dedede", lw=0.8, zorder=1)
        F.ci_dot(ax, i, r.C_dnam, None, None, F.C_DNAM, marker="s", ms=3.4)
        F.ci_dot(ax, i, r.C_clinical, None, None, F.C_CLOCK, ms=3.8)
        F.ci_dot(ax, i, r.C_joint, None, None, F.C_JOINT, marker="D", ms=3.2)
    yrows(ax, [lab[a] for a in h.analysis])
    ax.set_xlim(0.79, 0.84); ax.set_xticks([0.79, 0.80, 0.81, 0.82, 0.83, 0.84]); ax.set_xlabel("Harrell's C")
    F.style_axis(ax, "x")
    src("EDFig7", "b", h)


def e7c(B):
    B.title("HRS: independent information (likelihood ratio)")
    l = DS.hrs_lrt(); l = l[l.analysis.isin(["primary", "other_clock"])].drop_duplicates("clock")
    l = l.assign(a=-np.log10(l.p_add_clinical), b=-np.log10(l.p_add_dnam)).sort_values("b", ascending=False)
    B.legend([F.patch(F.C_CLOCK), F.patch(F.C_DNAM)], ["Clinical added to DNAm", "DNAm added to clinical"])
    ax = B.ax(26, 12.5, B.w - 29, B.h - 21.5)
    for i, r in enumerate(l.itertuples()):
        ax.barh(i - 0.19, r.a, height=0.36, color=F.C_CLOCK, zorder=2)
        ax.barh(i + 0.19, r.b, height=0.36, color=F.C_DNAM, zorder=2)
    ax.axvline(-np.log10(0.01), color=F.REF, ls="--", lw=0.6)
    yrows(ax, [HRS_LAB[c] for c in l.clock], pad=0.6)
    ax.tick_params(axis="y", labelsize=5.4)
    ax.set_xlabel("−log10 P, likelihood-ratio test"); ax.set_xlim(0, 68)
    F.style_axis(ax, "x")
    src("EDFig7", "c", l)


def ed7():
    build("EDFig7", [("a", e7a, 0, 0, 91.5, 74), ("b", e7b, 91.5, 0, 91.5, 74), ("c", e7c, 0, 75, 183, 84)], 159)


# =============================================================== ED Fig. 8
def e8a(B):
    B.title("System score added to whole body")
    p = DS.mimic_panels(); p = p[p.panel != "full"]
    B.legend([F.handle(F.INK2, "o", ls="none", mfc="white"), F.handle(F.INK2, "s", ls="none")],
             ["In-hospital", "365-day"])
    ax = B.ax(24, 12.5, B.w - 27, B.h - 21.5)
    rec = []
    for i, o in enumerate(ORG5):
        for e, off, mk, fill in [("index_hospital_mortality", -0.15, "o", "white"),
                                 ("post_landmark_365d_mortality", 0.15, "s", None)]:
            x = p[(p.endpoint == e) & (p.panel == o)].iloc[0]
            F.ci_dot(ax, i + off, x.incremental_auc_oof, x.incremental_auc_oof_ci95_lower,
                     x.incremental_auc_oof_ci95_upper, F.ORGAN_C[o], marker=mk, ms=3.4, mfc=fill)
            rec.append(dict(panel=o, endpoint=e, base=x.baseline_auc_oof, aug=x.augmented_auc_oof,
                            delta=x.incremental_auc_oof, lo=x.incremental_auc_oof_ci95_lower,
                            hi=x.incremental_auc_oof_ci95_upper, q=x.incremental_auc_oof_q_BH_organs))
    ax.axvline(0, color=F.REF, ls="--", lw=0.6)
    yrows(ax, [F.ORGAN_L[o] for o in ORG5])
    ax.set_xlim(-0.002, 0.005); ax.set_xticks([-0.002, 0, 0.002, 0.004])
    ax.set_xticklabels(["−0.002", "0", "0.002", "0.004"])
    ax.set_xlabel("Out-of-fold ΔAUC (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig8", "a", pd.DataFrame(rec))


def e8b(B):
    B.title("MIMIC-IV subgroups (365-day death)")
    sg = DS.subgroups(); sg = sg[sg.cohort == "mimic"]
    rows = [("all", "All"), ("female", "Women"), ("male", "Men"), ("age20_49", "20–49 years"),
            ("age50_64", "50–64 years"), ("age65_79", "65–79 years"), ("age80plus", "≥80 years")]
    ax = B.ax(24, 8.5, B.w - 27, B.h - 17.5)
    for i, (g, l) in enumerate(rows):
        r = sg[sg.group == g].iloc[0]
        c = F.C_FEMALE if g == "female" else (F.C_MALE if g == "male" else F.C_CLOCK)
        F.ci_dot(ax, i, r.harrell_C, r.harrell_C_low, r.harrell_C_high, c, ms=3.8)
    ax.axhline(2.5, color="#d0d0d0", lw=0.5)
    yrows(ax, [r[1] for r in rows])
    ax.set_xlim(0.64, 0.85); ax.set_xlabel("Harrell's C (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig8", "b", sg)


def e8c(B):
    B.title("ICU: common-window sensitivity")
    r = DS.icu_inc(); r = r[(r.population == "minus6_with_lab")]
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_JOINT, "s")], ["Hospital 0–24 h clock", "ICU-window clock"])
    rows = [(b, o) for o in ["event", "hospital_event"] for b in ["sapsii", "oasis", "sofa"]]
    ax = B.ax(30, 12.5, B.w - 33, B.h - 21.5)
    rec = []
    for i, (b, o) in enumerate(rows):
        for ck, off, c, mk in [("hospital24", -0.15, F.C_CLOCK, "o"), ("icu_minus6", 0.15, F.C_JOINT, "s")]:
            x = r[(r.base_severity == b) & (r.outcome == o) & (r.clock == ck)].iloc[0]
            lo, hi = [float(v) for v in x.incremental_auc_oof_ci95.strip("[]").split(",")]
            F.ci_dot(ax, i + off, x.incremental_auc_oof, lo, hi, c, marker=mk, ms=3.2)
            rec.append(dict(base=b, outcome=o, clock=ck, delta=x.incremental_auc_oof, lo=lo, hi=hi))
    ax.axhline(2.5, color="#d0d0d0", lw=0.5)
    SOURCE.setdefault("EDFig8", {}).pop("c", None)
    L = {"sapsii": "SAPS-II", "oasis": "OASIS", "sofa": "SOFA"}
    yrows(ax, [f"{L[b]} · {'1-year' if o == 'event' else 'in-hospital'}" for b, o in rows])
    ax.set_xlim(0, 0.125); ax.set_xlabel("ΔAUC with clock (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig8", "b", pd.DataFrame(rec))


def ed8():
    build("EDFig8", [("a", e8a, 0, 0, 91.5, 64), ("b", e8c, 91.5, 0, 91.5, 64)], 64)


# =============================================================== ED Fig. 9
def e9a(B):
    B.title("Prespecified organ evidence framework")
    m = DS.organ_matrix()
    ax = B.ax(26, 16, B.w - 30, B.h - 19)
    cols = [("independent_marker", "Independent\nreference"), ("future_disease", "Future\ndisease"),
            ("functional_change", "Functional\nchange")]
    order = ["kidney", "metabolic", "cardiovascular", "immune_heme", "liver"]
    for i, o in enumerate(order):
        r = m[m.organ == o].iloc[0]
        for j, (k, _) in enumerate(cols):
            ok = r[k] == 1
            ax.plot(j, i, "o", ms=6.5, color=F.ORGAN_C[o] if ok else "white", mec=F.ORGAN_C[o], mew=0.9)
        ax.text(len(cols) - 0.35, i, r.criteria_met.replace(" of ", "/"), va="center", ha="left", fontsize=F.FS_TICK)
    ax.set_xlim(-0.5, len(cols) + 0.3); ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=5.6)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order))); ax.set_yticklabels([F.ORGAN_L[o] for o in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    src("EDFig9", "a", m)


def heat_partial(B, df, rows, title, figpanel):
    B.title(title)
    cmap = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f3f3f3", "#d43c3b"])
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.35)
    org4 = ["liver", "kidney", "metabolic", "immune_heme"]
    ax = B.ax(26, 8, B.w - 40, B.h - 20)
    rec = []
    for i, (key, lab, matched) in enumerate(rows):
        for j, o in enumerate(org4):
            v = df(key, o)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=cmap(norm(v)), ec="white", lw=1.0))
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.3,
                    color="white" if abs(v) > 0.22 else F.INK)
            if o == matched:
                ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fc="none", ec=F.INK, lw=0.9, zorder=5))
            rec.append(dict(reference=lab, panel=o, partial_r=v, matched=o == matched))
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([F.ORGAN_L[o].replace("Immune/blood", "Immune/\nblood") for o in org4],
                                                fontsize=5.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=5.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = B.ax(B.w - 11, 12, 1.8, B.h - 30)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4); cb.set_label("Partial r", fontsize=5.6, labelpad=1.5)
    src("EDFig9", figpanel, pd.DataFrame(rec))


def e9b(B):
    v = DS.hrs_vbs_ref(); v = v[v.group == "clock_GrimAge2"]
    rows = [("cystatin", "Cystatin C", "kidney"), ("il6", "IL-6", "immune_heme"), ("stnfr1", "sTNFR1", "immune_heme"),
            ("cmv", "CMV IgG", "immune_heme"), ("ntprobnp", "NT-proBNP", None)]
    heat_partial(B, lambda k, o: float(v[(v.target == k) & (v.panel == o)].partial_r.iloc[0]), rows,
                 "HRS measured references", "b")


def e9c(B):
    p = DS.hrs_proxy()
    rows = [("proxy_loga1c", "DNAm log HbA1c", "metabolic"), ("proxy_leptin", "DNAm leptin", "metabolic"),
            ("proxy_cystatin", "DNAm cystatin C", "kidney"), ("proxy_b2m", "DNAm B2M", "kidney"),
            ("proxy_logcrp", "DNAm log CRP", "immune_heme"), ("proxy_gdf15", "DNAm GDF15", "immune_heme"),
            ("proxy_pai1", "DNAm PAI-1", None)]
    heat_partial(B, lambda k, o: float(p[(p.marker == k) & (p.panel == o)].partial_r.iloc[0]), rows,
                 "HRS DNAm-derived surrogates", "c")


def e9d(B):
    B.title("NHANES III matched-outcome AUC")
    v = DS.n3_validation()
    lab = {"metabolic": "Diabetes death", "kidney": "Renal death", "cardiovascular": "Cardiovascular death",
           "immune_heme": "Influenza/pneumonia death", "liver": "Hepatic steatosis"}
    B.legend([F.handle(F.INK2, "o", ls="none", mfc="white"), F.handle(F.INK2, "o", ls="none")],
             ["Whole-body base model", "+ matched system score"])
    ax = B.ax(33, 12.5, B.w - 36, B.h - 21.5)
    for i, r in enumerate(v.itertuples()):
        ax.plot([r.auc_full, r.auc_both], [i, i], color="#dadada", lw=0.9, zorder=1)
        F.ci_dot(ax, i, r.auc_full, None, None, F.INK2, mfc="white", ms=3.6)
        F.ci_dot(ax, i, r.auc_both, None, None, F.ORGAN_C[r.panel], ms=4.0)
    yrows(ax, [lab[p] for p in v.panel])
    ax.set_xlim(0.6, 0.9); ax.set_xlabel("Out-of-fold AUC")
    F.style_axis(ax, "x")
    src("EDFig9", "d", v)


def ed9():
    build("EDFig9", [("a", e9a, 0, 0, 80, 56), ("b", e9b, 80, 0, 103, 56),
                     ("c", e9c, 0, 57, 103, 66), ("d", e9d, 103, 57, 80, 66)], 123)


# =============================================================== ED Fig. 10
def e10a(B):
    B.title("Incident ADL disability (community)")
    s = DS.healthspan_s1(); s = s[s.panel == "a"]
    B.legend([F.handle(F.C_CLOCK), F.handle(F.C_AGESEX, "s")], ["Age, sex, education", "+ frailty index"])
    ax = B.ax(22, 12.5, B.w - 25, B.h - 21.5)
    for i, c in enumerate(["hrs", "elsa", "charls"]):
        a = s[(s.cohort == c) & (s.model == "demog")].iloc[0]; b = s[(s.cohort == c) & (s.model == "fiadj")].iloc[0]
        F.ci_dot(ax, i - 0.14, a.value, a.low, a.high, F.C_CLOCK)
        F.ci_dot(ax, i + 0.14, b.value, b.low, b.high, F.C_AGESEX, marker="s", ms=3.4)
    ax.axvline(1, color=F.REF, ls="--", lw=0.6)
    yrows(ax, [f"{COH[c]} ({int(s[s.cohort == c].events.iloc[0]):,})" for c in ["hrs", "elsa", "charls"]])
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xlim(0.9, 1.9)
    ax.set_xticks([1, 1.25, 1.5, 1.75]); ax.set_xticklabels(["1", "1.25", "1.5", "1.75"])
    ax.set_xlabel("HR per SD acceleration (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig10", "a", s)


def e10b(B):
    B.title("HRS physical function, 2014 → 2018")
    import data_src as D2
    p = D2.csv(f"{D2.SV}/local_completion/health_long/hrs_physical_2014_2018.csv")
    rows = [("grp", False, "Grip strength (kg)"), ("grp", True, "Grip strength (kg), + FI"),
            ("timwlk", False, "Walk time (log)"), ("timwlk", True, "Walk time (log), + FI")]
    ax = B.ax(32, 8.5, B.w - 35, B.h - 17.5)
    for i, (m, fi, l) in enumerate(rows):
        r = p[(p.measure == m) & (p.adjusted_for_FI == fi)].iloc[0]
        sc = 1 if m == "grp" else 10
        F.ci_dot(ax, i, r.beta * sc, r.lo * sc, r.hi * sc, F.C_CLOCK if m == "grp" else F.C_JOINT, ms=3.8)
    ax.axvline(0, color=F.REF, ls="--", lw=0.6)
    yrows(ax, ["Grip (kg)", "Grip (kg) · + FI", "Walk time (×10 log)", "Walk time (×10 log) · + FI"])
    ax.set_xlim(-1.2, 0.8); ax.set_xlabel("Difference per SD acceleration (95% CI)")
    F.style_axis(ax, "x")
    src("EDFig10", "b", p)


def e10c(B):
    B.title("Exposures replicated across three NHANES stages")
    x = DS.xwas(); x = x[x.three_stage == True]
    lab = {"LBXCOT": "Serum cotinine", "LBXBCD": "Blood cadmium", "LBXBPB": "Blood lead", "LBXTHG": "Blood mercury",
           "LBXVTO": "Blood toluene", "LBXVXY": "Blood m/p-xylene", "URXMZP": "Urinary MBzP", "URXMBP": "Urinary MnBP",
           "LBXPFOS": "Serum PFOS", "LBXPFOA": "Serum PFOA", "LBXPFNA": "Serum PFNA", "LBXPFHS": "Serum PFHxS",
           "LBXPFDE": "Serum PFDA", "two_day_HEI2015": "HEI-2015 diet score",
           "leisure_MVPA_log_weekly_minutes": "Leisure MVPA (log min/wk)", "education_level": "Education level",
           "poverty_income_ratio": "Income-to-poverty ratio"}
    exps = [e for e in lab if e in set(x.exposure)]
    pans = ["full_accel"] + [f"{o}_accel" for o in ORG5]
    M = np.full((len(exps), len(pans)), np.nan)
    for i, e in enumerate(exps):
        for j, pn in enumerate(pans):
            r = x[(x.exposure == e) & (x.panel == pn)]
            if len(r):
                M[i, j] = r.temporal_replication_beta.iloc[0]
    ax = B.ax(30, 8, B.w - 44, B.h - 22)
    cmap = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f3f3f3", "#d43c3b"])
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    for i in range(len(exps)):
        for j in range(len(pans)):
            v = M[i, j]
            fc = cmap(norm(v)) if np.isfinite(v) else "white"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=fc, ec="#e6e6e6" if not np.isfinite(v) else "white", lw=0.8))
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.0,
                        color="white" if abs(v) > 1.2 else F.INK)
    ax.set_xlim(-0.5, len(pans) - 0.5); ax.set_ylim(len(exps) - 0.5, -0.5)
    ax.set_xticks(range(len(pans)))
    ax.set_xticklabels([F.ORGAN_L[p.replace("_accel", "")] for p in pans], fontsize=5.6, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(exps))); ax.set_yticklabels([lab[e] for e in exps], fontsize=5.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = B.ax(B.w - 11, 14, 1.8, B.h - 36)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4)
    cb.set_label("β (years per SD), replication", fontsize=5.4, labelpad=1.5)
    src("EDFig10", "c", x)


def ed10():
    build("EDFig10", [("a", e10a, 0, 0, 91.5, 50), ("b", e10b, 91.5, 0, 91.5, 50),
                      ("c", e10c, 0, 51, 183, 96)], 147)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        globals()[f"ed{w}"]()
        print("built ED", w)
