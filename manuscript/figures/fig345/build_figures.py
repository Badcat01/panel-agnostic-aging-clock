"""Publication panels from audited aggregate results. Run with the project Python."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, StrMethodFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
EXT = HERE.parents[2] / "external_validation"
SUP = EXT / "supplement_v2"
NH3 = EXT / "nhanes3_organ"
MIMIC = EXT / "mimic_iv" / "formal_v3_1"
LOCAL = SUP / "local_completion"

BLUE, RED, GRAY = "#164E63", "#A6492F", "#586475"
INK, RULE, PALE = "#22272E", "#9299A1", "#DDEBF0"
WIDTH = 180 / 25.4
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.linewidth": 0.65,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "xtick.major.size": 3,
        "xtick.major.width": 0.65,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "axes.unicode_minus": True,
    }
)


def place(fig, x, y, width, height):
    return fig.add_axes((x, y, width, height))


def heading(fig, x, y, letter, text):
    fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="top")
    fig.text(x + 0.029, y, text, fontsize=7, fontweight="bold", ha="left", va="top")


def dataset(fig, x, y, text):
    fig.text(x, y, text, fontsize=7, ha="left", va="top", linespacing=1.3)


def estimate(ax, y, point, low, high, color, marker="o", size=4.4):
    assert low <= point <= high
    ax.errorbar(
        point, y, xerr=[[point - low], [high - point]], fmt=marker, color=color,
        ecolor=color, elinewidth=0.8, capsize=2.1, capthick=0.8, markersize=size,
        markeredgewidth=0.55, zorder=3,
    )


def axis(ax, limits, ticks, yvalues, labels, xlabel, *, centre=None, formatter="{x:.2f}"):
    ax.set_xlim(*limits)
    ax.set_yticks(yvalues, labels)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(StrMethodFormatter(formatter))
    ax.tick_params(axis="y", pad=4)
    if centre is not None:
        ax.axvline(centre, color=RULE, linestyle=(0, (3, 2)), linewidth=0.7, zorder=0)


def export(fig, stem):
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(OUT / f"{stem}{suffix}", dpi=300)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    renderer = fig.canvas.get_renderer()
    # Render validation catches future axis-label and page boundary failures.
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if min(box.x0, box.y0) < -1 or box.x1 > w + 1 or box.y1 > h + 1:
            raise ValueError(f"Text is outside {stem}: {artist.get_text()!r}")
    plt.close(fig)


def save_source(table, name):
    table.to_csv(HERE / (name + "_source.csv"), index=False, float_format="%.12g")


def fig3():
    d = pd.read_csv(SUP / "biomarkers" / "dnam_discrimination.csv")
    labels = {
        "joint_GrimAge2": "Clinical + GrimAge2", "clinical": "Clinical clock",
        "GrimAge2Mort": "GrimAge2", "GrimAgeMort": "GrimAge",
        "DunedinPoAm": "DunedinPoAm", "PhenoAge": "DNAm PhenoAge",
        "HorvathTelo": "DNAm telomere length", "HannumAge": "Hannum",
        "HorvathAge": "Horvath", "LinAge": "Lin", "ZhangAge": "Zhang",
        "SkinBloodAge": "Skin and blood", "age_sex": "Age + sex",
    }
    d["display_model"] = d.model.map(labels)
    main = d.loc[d.analysis.eq("primary")].sort_values("C", ascending=False)
    g = d.loc[d.model.eq("GrimAge2Mort")].set_index("analysis")
    joint = d.loc[d.model.eq("joint_GrimAge2")].set_index("analysis")
    assert len(main) == 13 and main[["n", "deaths", "bootstrap_reps"]].nunique().eq(1).all()
    n, deaths = main[["n", "deaths"]].iloc[0]
    fig = plt.figure(figsize=(WIDTH, 119 / 25.4))
    heading(fig, 0.031, 0.965, "a", "Mortality ranking in the same participants")
    dataset(fig, 0.060, 0.918, f"NHANES DNAm subset  |  n = {n:,}, deaths = {deaths:,}")
    heading(fig, 0.556, 0.965, "b", "Clinical clock versus GrimAge2")
    heading(fig, 0.556, 0.477, "c", "Combining clinical and DNAm scores")

    ax = place(fig, 0.222, 0.149, 0.295, 0.691)
    yvalues = np.arange(len(main))[::-1]
    color = {"clinical": BLUE, "joint_GrimAge2": BLUE, "GrimAge2Mort": RED}
    for y, row in zip(yvalues, main.itertuples()):
        marker = {"clinical": "o", "joint_GrimAge2": "D", "GrimAge2Mort": "s"}.get(row.model, "o")
        estimate(ax, y, row.C, row.C_lower, row.C_upper, color.get(row.model, GRAY), marker)
    ax.set_ylim(-0.65, len(main) - 0.35)
    axis(ax, (0.72, 0.806), [0.72, 0.74, 0.76, 0.78, 0.80], yvalues,
         main.display_model.tolist(), "Out-of-fold Harrell’s C")

    ax = place(fig, 0.736, 0.595, 0.226, 0.274)
    sensitivity_labels = ["Main", "Cell composition", "Survey weights", "Both"]
    variant_order = ["primary", "cells", "weighted", "weighted_cells"]
    for y, row in zip(range(3, -1, -1), g.loc[variant_order].itertuples()):
        estimate(ax, y, row.clinical_minus_C, row.delta_lower, row.delta_upper, BLUE)
    ax.set_ylim(-0.55, 3.6)
    axis(ax, (-0.001, 0.034), [0, 0.01, 0.02, 0.03], range(3, -1, -1),
         sensitivity_labels, "Clinical minus GrimAge2 (ΔC)", centre=0)
    ax = place(fig, 0.736, 0.149, 0.226, 0.247)
    main_g, main_j = g.loc["primary"], joint.loc["primary"]
    main_boost = [
        ("Add clinical", main_j.C - main_g.C, np.nan, np.nan, BLUE),
        ("Add GrimAge2", -main_j.clinical_minus_C,
         -main_j.delta_upper, -main_j.delta_lower, RED),
    ]
    # Use the joint model's conditional-prediction bootstrap contrast, available only
    # relative to the clinical score. The reverse comparison uses a point estimate.
    for y, (_, value, low, high, col) in zip([1, 0], main_boost):
        if np.isfinite(low):
            estimate(ax, y, value, low, high, col, "s")
        else:
            ax.plot(value, y, "o", color=col, markersize=4.4)
        ax.annotate(f"{value:+.4f}", (value, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    ax.set_ylim(-0.5, 1.55)
    axis(ax, (-0.001, 0.034), [0, 0.01, 0.02, 0.03], [1, 0],
         [x[0] for x in main_boost], "Increase in Harrell’s C", centre=0)
    save_source(d, "Fig3")
    save_source(pd.DataFrame([r[:4] for r in main_boost],
                             columns=["contrast", "estimate", "low", "high"]), "Fig3_joint")
    export(fig, "Fig3")


def fig4():
    m = pd.read_csv(MIMIC / "panel_metrics.csv").query("panel == 'full'").set_index("endpoint")
    verification = json.loads((MIMIC / "verification.json").read_text(encoding="utf-8"))
    h = pd.read_csv(LOCAL / "clinical_tech" / "sapsii_prior_same_people_increment.csv")
    h = h.loc[h.comorbidity.eq("sapsii")].set_index("outcome")
    fig = plt.figure(figsize=(WIDTH, 99 / 25.4))
    heading(fig, 0.031, 0.956, "a", "Frozen score at the 24-h hospital landmark")
    dataset(fig, 0.060, 0.885, f"MIMIC-IV v3.1  |  n = {verification['n']:,}")
    heading(fig, 0.556, 0.956, "b", "Frozen score versus ridge Cox")
    heading(fig, 0.556, 0.507, "c", "Added value beyond SAPS-II")
    dataset(fig, 0.585, 0.444, f"Subsequent ICU admissions  |  n = {int(h.n.iloc[0]):,}")
    ax = place(fig, 0.211, 0.17, 0.323, 0.612)
    rows = [
        ("index_hospital_mortality", "hospital_event", "In-hospital"),
        ("post_landmark_365d_mortality", "event", "365-day"),
    ]
    model_keys = [
        ("frozen_panel_auc", "Clinical clock", BLUE, "o"),
        ("frozen_ridge_lp_auc", "Same-input ridge Cox", RED, "s"),
        ("frozen_age_sex_lp_auc", "Age + sex Cox", GRAY, "^"),
    ]
    records = []
    label_positions = []
    for i, (key, result, label) in enumerate(rows):
        row = m.loc[key]
        y = 4 - 4 * i
        label_positions.append(y)
        for offset, (base, name, col, shape) in zip((0.73, 0, -0.73), model_keys):
            value, lo, hi = row[[base, base + "_ci95_lower", base + "_ci95_upper"]]
            estimate(ax, y + offset, value, lo, hi, col, shape)
            ax.text(0.859, y + offset, f"{value:.4f}", ha="right", va="center", fontsize=7)
            records.append(dict(panel="a", outcome=label, deaths=row.deaths, n=row.n_compared,
                                model=name, value=value, low=lo, high=hi))
    ax.set_ylim(-1.1, 5)
    axis(ax, (0.63, 0.865), [0.65, 0.70, 0.75, 0.80], label_positions,
         [f"{label}\n{verification['outcomes'][result]['deaths']:,} deaths" for _, result, label in rows],
         "Frozen-score AUC")
    fig.legend(
        handles=[Line2D([0], [0], marker=shape, color=col, linestyle="", markersize=4.4, label=name)
                 for _, name, col, shape in model_keys],
        loc="upper left", bbox_to_anchor=(0.057, 0.838), frameon=False, ncol=3, fontsize=7,
        handletextpad=0.28, handlelength=1.0, columnspacing=0.65, borderaxespad=0,
    )

    ax = place(fig, 0.717, 0.6, 0.245, 0.27)
    for y, (_, result, label) in zip([1, 0], rows):
        record = verification["outcomes"][result]
        contrast = record["comparators"]["ridge_lp"]
        value, (lo, hi) = contrast["clock_minus_comparator"], contrast["CI95"]
        estimate(ax, y, value, lo, hi, BLUE)
        ax.annotate(f"{value:+.4f}", (value, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom")
        records.append(dict(panel="b", outcome=label, n=verification["n"], deaths=record["deaths"],
                            model="Clinical clock minus same-input ridge Cox", value=value, low=lo, high=hi))
    ax.set_ylim(-0.4, 1.55)
    axis(ax, (-0.0005, 0.019), [0, 0.005, 0.010, 0.015],
         [1, 0], [label for _, _, label in rows], "Paired ΔAUC", centre=0, formatter="{x:.3f}")

    ax = place(fig, 0.717, 0.17, 0.245, 0.23)
    for y, (_, result, label) in zip([1, 0], rows):
        row = h.loc[result]
        value = row.incremental_auc_oof
        lo, hi = ast.literal_eval(row.incremental_auc_oof_ci95)
        estimate(ax, y, value, lo, hi, BLUE, "D")
        ax.annotate(f"{value:+.4f}", (value, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom")
        records.append(dict(panel="c", outcome=label, n=row.n, deaths=row.events,
                            model="SAPS-II (prior history) + clinical clock minus SAPS-II (prior history)",
                            value=value, low=lo, high=hi, base_auc=row.baseline_auc_oof,
                            augmented_auc=row.augmented_auc_oof))
    ax.set_ylim(-0.45, 1.6)
    axis(ax, (-0.001, 0.042), [0, 0.01, 0.02, 0.03, 0.04], [1, 0],
         [f"{lab}\n{int(h.loc[ev, 'events']):,} deaths" for _, ev, lab in rows],
         "Out-of-fold ΔAUC", centre=0)
    save_source(pd.DataFrame(records), "Fig4")
    export(fig, "Fig4")


ORGAN_LABELS = {
    "kidney": "Kidney", "metabolic": "Metabolic", "cardiovascular": "Cardiovascular",
    "immune_heme": "Immune / blood", "liver": "Liver",
}


def fig5():
    cyst = json.loads((NH3 / "kidney_cystatin.json").read_text(encoding="utf-8"))
    by_endpoint = pd.read_csv(NH3 / "validation_summary.csv").set_index("panel")
    readmit = pd.read_csv(SUP / "technical" / "readmission_matched.csv").query(
        "organ == 'kidney' and code_position == 'principal' and eligibility == 'one_or_more'"
    ).iloc[0]
    cat = [
        ("kidney", 1, 1, 0, "2 of 3"),
        ("metabolic", 0, 1, 0, "1 of 3"),
        ("cardiovascular", 0, 1, 0, "1 of 3"),
        ("immune_heme", 1, 0, 0, "1 of 3"),
        ("liver", 0, 0, 0, "0 of 3"),
    ]
    fig = plt.figure(figsize=(WIDTH, 135 / 25.4))
    heading(fig, 0.031, 0.959, "a", "Cystatin C in NHANES III")
    dataset(fig, 0.060, 0.909, f"Same participants aged ≥60  |  n = {cyst['counts']['analytic']:,}")
    heading(fig, 0.524, 0.959, "b", "Organ score added to the whole-body model")
    heading(fig, 0.031, 0.431, "c", "Evidence of stronger matched than nonmatched associations")
    ax = place(fig, 0.183, 0.533, 0.286, 0.304)
    records = []
    yvalues = np.arange(len(cat))[::-1]
    for y, (organ, *_) in zip(yvalues, cat):
        stats = cyst["association"][organ]
        value, (lo, hi) = stats["percent_change_per_sd"], stats["robust_95ci_percent_change"]
        estimate(ax, y, value, lo, hi, BLUE if organ == "kidney" else GRAY,
                 "o" if organ == "kidney" else "s")
        records.append(dict(panel="a", study="NHANES III", endpoint="Cystatin C", organ=organ,
                            n=cyst["counts"]["analytic"], value=value, low=lo, high=hi))
    ax.set_ylim(-0.55, 4.55)
    axis(ax, (-6.0, 14.0), [-5, 0, 5, 10], yvalues, list(ORGAN_LABELS.values()),
         "Difference per 1 SD higher group score (%)", centre=0, formatter="{x:.0f}")

    ax = place(fig, 0.745, 0.533, 0.219, 0.304)
    short_endpoints = [
        ("NHANES III", "metabolic", "Diabetes death"),
        ("NHANES III", "cardiovascular", "Stroke/heart death"),
        ("NHANES III", "kidney", "Renal death"),
        ("NHANES III", "immune_heme", "Influenza/pneumonia"),
        ("NHANES III", "liver", "Liver steatosis"),
        ("MIMIC-IV", "kidney", "Renal readmission"),
    ]
    names = []
    for y, (study, organ, outcome) in zip(range(5, -1, -1), short_endpoints):
        if study == "NHANES III":
            row = by_endpoint.loc[organ]
            value, lo, hi = row[["delta_auc", "delta_auc_low", "delta_auc_high"]]
            events, n = row.events_by_horizon, row.n
            if pd.isna(events):
                events = row.events
            marker = "o"
        else:
            value, lo, hi = readmit[["incremental_auc_oof", "ci_low", "ci_high"]]
            events, n = readmit.events, readmit.n
            marker = "D"
        color = BLUE if organ == "metabolic" or study == "MIMIC-IV" else GRAY
        estimate(ax, y, value, lo, hi, color, marker, size=3.6)
        names.append(f"{outcome}  ({int(events):,})")
        records.append(dict(panel="b", study=study, organ=organ, endpoint=outcome, n=n,
                            events=events, value=value, low=lo, high=hi))
    ax.axhline(0.5, color=RULE, linewidth=0.55)
    ax.set_ylim(-0.65, 5.6)
    axis(ax, (-0.022, 0.049), [-0.02, 0, 0.02, 0.04], range(5, -1, -1), names,
         "Out-of-fold ΔAUC", centre=0)
    dataset(fig, 0.553, 0.908, "NHANES III rows above the divider\nMIMIC-IV below  |  Events in parentheses")

    matrix = place(fig, 0.060, 0.064, 0.899, 0.308)
    matrix.set(xlim=(0, 1), ylim=(0, 6.25))
    matrix.axis("off")
    heads = ["Organ group", "Independent marker", "Future disease", "Functional change", "Lines met"]
    xx = [0.0, 0.32, 0.51, 0.71, 0.90]
    for i, (x, head) in enumerate(zip(xx, heads)):
        matrix.text(x, 5.66, head, ha="left" if i == 0 else "center", va="center",
                    fontsize=7, fontweight="bold")
    matrix.hlines(5.2, 0, 1, colors=RULE, linewidth=0.6)
    for i, (organ, *classes, count) in enumerate(cat):
        y = 4.65 - i * 0.87
        if organ == "kidney":
            matrix.add_patch(Rectangle((0, y - .34), 1, .68, facecolor=PALE, edgecolor="none"))
        matrix.text(0.0, y, ORGAN_LABELS[organ], va="center",
                    fontweight="bold" if organ == "kidney" else "normal")
        for x, met in zip(xx[1:4], classes):
            matrix.plot(x, y, "o", markeredgecolor=BLUE if met else GRAY,
                        markerfacecolor=BLUE if met else "white", markeredgewidth=.65, markersize=4)
        matrix.text(xx[-1], y, count, ha="center", va="center",
                    fontweight="bold" if organ == "kidney" else "normal")
    matrix.plot(0.006, 0.11, "o", color=BLUE, markersize=4)
    matrix.text(0.027, 0.11, "Criterion met", va="center")
    matrix.plot(0.22, 0.11, "o", markeredgecolor=GRAY, markerfacecolor="white",
                markeredgewidth=.65, markersize=4)
    matrix.text(0.24, 0.11, "Criterion not met", va="center")
    matrix.text(0.49, 0.11, "Organ specificity required two evidence lines.", va="center")
    save_source(pd.DataFrame(records), "Fig5_effects")
    save_source(pd.DataFrame(cat, columns=["organ", "independent_marker", "future_disease",
                                           "functional_change", "criteria_met"]), "Fig5_matrix")
    export(fig, "Fig5")


def fig_s1():
    directory = SUP / "healthspan"
    c = pd.read_csv(directory / "disc.csv").query("endpoint == 'adl'")
    c = c.pivot(index="cohort", columns="model", values="C")
    assoc = pd.read_csv(directory / "out.csv").query("endpoint == 'adl'").set_index("cohort")
    fig = plt.figure(figsize=(WIDTH, 76 / 25.4))
    heading(fig, 0.031, 0.909, "a", "Incident ADL difficulty")
    heading(fig, 0.555, 0.909, "b", "Added ranking information beyond the FI")
    ax = place(fig, 0.173, 0.235, 0.327, 0.518)
    bx = place(fig, 0.686, 0.235, 0.270, 0.518)
    records = []
    labels = []
    for y, co in zip([2, 1, 0], ["hrs", "elsa", "charls"]):
        row = assoc.loc[co]
        labels.append(f"{co.upper()}\n{int(row.events):,}/{int(row.n):,}")
        for offset, prefix, fmt, col in [(.13, "demog", "s", GRAY), (-.13, "fiadj", "o", BLUE)]:
            value, low, high = row[[prefix + "_est", prefix + "_lower", prefix + "_upper"]]
            estimate(ax, y+offset, value, low, high, col, fmt)
            records.append(dict(panel="a", cohort=co, n=row.n, events=row.events, model=prefix,
                                value=value, low=low, high=high))
        delta = c.loc[co, "FI_clock"] - c.loc[co, "FI"]
        bx.plot(delta, y, "o", color=BLUE, markersize=4.4)
        bx.annotate(f"{delta:+.5f}", (delta, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom")
        records.append(dict(panel="b", cohort=co, n=row.n, events=row.events, model="FI + clinical minus FI",
                            value=delta, base_C=c.loc[co, "FI"], augmented_C=c.loc[co, "FI_clock"]))
    for axes in (ax, bx):
        axes.set_ylim(-0.46, 2.49)
    axis(ax, (0.89, 1.83), [0.9, 1.1, 1.3, 1.5, 1.7],
         [2, 1, 0], labels, "Hazard ratio per 1 SD higher clinical score", centre=1, formatter="{x:.1f}")
    axis(bx, (-0.0011, 0.00142), [-0.001, 0, 0.001], [2, 1, 0],
         ["HRS", "ELSA", "CHARLS"], "Out-of-fold ΔC", centre=0, formatter="{x:.3f}")
    fig.legend(
        handles=[Line2D([0], [0], marker="s", color=GRAY, linestyle="", markersize=4.4, label="Demographics"),
                 Line2D([0], [0], marker="o", color=BLUE, linestyle="", markersize=4.4, label="Demographics + FI")],
        loc="upper left", bbox_to_anchor=(0.060, 0.831), frameon=False, ncol=2, fontsize=7,
        handletextpad=0.28, handlelength=1.0, columnspacing=1.1, borderaxespad=0,
    )
    dataset(fig, 0.585, 0.823, "FI = frailty index")
    save_source(pd.DataFrame(records), "FigS1")
    export(fig, "FigS1")


if __name__ == "__main__":
    fig3()
    fig4()
    fig5()
    fig_s1()
