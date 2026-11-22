"""Build the manuscript figures for transfer, calibration and ablations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
EXT = PROJECT / "external_validation"
SUP = EXT / "supplement_v2"
RUN = PROJECT / "runs" / "nhanes_gpu_20261121"
BLUE = "#164E63"
ORANGE = "#A6492F"
GRAY = "#586475"
INK = "#1B2530"
RULE = "#B9C5CB"
TINT = "#EDF3F5"
SOFT = "#F4F6F7"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRAY,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.65,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "xtick.major.size": 2.8,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "aging-clock-manuscript",
    "savefig.facecolor": "white",
})

COMMUNITY = {
    "temporal": "Time holdout",
    "nhanes3": "NHANES III",
    "hrs": "HRS",
    "charls": "CHARLS",
    "elsa": "ELSA",
    "share": "SHARE",
    "klosa": "KLoSA",
}
SIMULATED = {
    "full": "Full panel",
    "nhanes3": "NHANES III",
    "hrs": "HRS",
    "mimic": "MIMIC",
    "charls": "CHARLS",
    "elsa": "ELSA",
    "share": "SHARE",
}


def canvas(height_mm):
    return plt.figure(figsize=(180 / 25.4, height_mm / 25.4), facecolor="white")


def header(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=8, fontweight="bold", va="center")
    fig.text(x+0.023, y, title, fontweight="bold", va="center")


def whisker(ax, x, y, lo, hi, color, marker):
    ax.plot([lo, hi], [y, y], color=color, lw=1.05, solid_capstyle="round")
    ax.plot(x, y, marker=marker, color=color, markersize=4.2, linestyle="none")


def point_key(color, marker, label, filled=True):
    return Line2D([0], [0], marker=marker, color=color, linewidth=1.0,
                  markerfacecolor=color if filled else "none", markersize=4.2, label=label)


def forest_axis(ax, xlim, ylim, step):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks([])
    ax.set_xticks(np.arange(np.ceil(xlim[0] / step), np.floor(xlim[1] / step)+1)*step)
    ax.spines.left.set_visible(False)
    ax.axvline(0, color=GRAY, lw=0.75, ls=(0, (3, 3)), zorder=1)


def card(ax, x, y, width, height, title, description, color=BLUE, emphasis=False):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=TINT if emphasis else SOFT,
                           edgecolor=RULE, linewidth=.65))
    ax.plot([x, x + width], [y + height, y + height], color=color, lw=1.4,
            solid_capstyle="butt")
    ax.text(x+.014, y+height-.09, title, va="top", fontweight="bold")
    ax.text(x+.014, y+height-.23, description, va="top", linespacing=1.32)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", color=GRAY,
                                 mutation_scale=7, linewidth=.8))


def format_source(frame, name):
    frame.to_csv(HERE / name, index=False, float_format="%.12g")


def export(fig, number):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_extent = fig.bbox
    for artist in fig.findobj(match=matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.xmin >= 0 and box.ymin >= 0, (number, artist.get_text(), box)
            assert box.xmax <= canvas_extent.xmax and box.ymax <= canvas_extent.ymax, (
                number, artist.get_text(), box)
    for extension in ("pdf", "svg", "png"):
        file = HERE.parent / f"Fig{number}.{extension}"
        params = {"dpi": 300} if extension == "png" else {}
        fig.savefig(file, **params)
    plt.close(fig)


def read_validation():
    counts = pd.read_csv(EXT / "results" / "cohort_summary.csv").set_index("cohort")
    comparisons = pd.read_csv(SUP / "models" / "comparisons.csv")
    coverage = {
        "temporal": comparisons.query("setting == 'primary' and panel == 'full' "
                                     "and model == 'M1_nomask'").median_observed.item(),
        "nhanes3": json.loads((EXT / "results" / "nhanes3" / "metrics.json")
                             .read_text(encoding="utf-8"))["median_features"],
        **{c: counts.loc[COMMUNITY[c], "median_features"]
           for c in COMMUNITY if c not in {"temporal", "nhanes3"}},
    }
    survival = SUP / "technical" / "survival"
    score = pd.concat([
        pd.read_csv(survival / f"{key}_C.csv").query("method == 'Harrell'")
        for key in COMMUNITY
    ], ignore_index=True)
    score["median_features"] = score["cohort"].map(coverage)
    np.testing.assert_array_equal(score.B, 2000)
    cal = pd.read_csv(SUP / "technical" / "horizons.csv").query(
        "horizon == 3 and model == 'clock' and supported and cohort in @COMMUNITY")
    for key in COMMUNITY:
        assert cal.loc[cal.cohort == key, "n"].item() == score.loc[score.cohort == key, "n"].iloc[0]
    format_source(score.drop(columns="horizon"), "Fig1b_source.csv")
    format_source(cal[["cohort", "n", "horizon", "events_to_h", "observed_KM",
                       "predicted", "E_over_O", "calibration_slope"]], "Fig1c_source.csv")
    return score.set_index(["cohort", "model"]), cal.set_index("cohort")


def fig1():
    data, calibration = read_validation()
    holdout = json.loads((RUN / "temporal_metrics.json").read_text(encoding="utf-8"))
    fig = canvas(142)
    header(fig, .036, .960, "a", "Development, freeze and validation")
    ax = fig.add_axes([.036, .690, .928, .220], frameon=False)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    dev_counts = f"{holdout['n_train_baselines']:,} participants · {holdout['deaths_train_baselines']:,} deaths"
    ho_counts = f"{holdout['n']:,} participants · {holdout['deaths']:,} deaths"
    card(ax, 0, .153, .29, .73, "NHANES 1999–2008",
         f"{dev_counts}\nAge and 43 routine measures\nMasked mortality training")
    card(ax, .349, .153, .240, .73, "Freeze model and reference",
         "Five-seed ensemble\nFixed weights, preprocessing\nand NHANES risk scale", emphasis=True)
    card(ax, .653, .537, .347, .46, "NHANES 2009–2018 holdout",
         ho_counts, color=GRAY)
    card(ax, .653, .017, .347, .46, "External cohorts and DNAm holdout",
         "Community mortality, inpatient care, DNAm", color=GRAY)
    arrow(ax, (.297, .515), (.344, .515))
    arrow(ax, (.594, .530), (.649, .759))
    arrow(ax, (.594, .491), (.649, .243))

    header(fig, .036, .637, "b", "Mortality discrimination")
    header(fig, .686, .637, "c", "Three-year calibration")
    fig.legend(handles=[point_key(BLUE, "o", "vs. same-input ridge Cox"),
                        point_key(ORANGE, "s", "vs. age + sex Cox")],
               loc="upper left", bbox_to_anchor=(.032, .599), ncols=2,
               borderaxespad=0, handlelength=1.2, columnspacing=1.05)
    fig.legend(handles=[point_key(BLUE, "o", "Observed"), point_key(ORANGE, "s", "Predicted", False)],
               loc="upper left", bbox_to_anchor=(.704, .599), ncols=2,
               borderaxespad=0, handlelength=1.2, columnspacing=1.05)

    text = fig.add_axes([.036, .121, .206, .394], frameon=False)
    text.set(xlim=(0, 1), ylim=(-.58, 6.70))
    text.axis("off")
    for xpos, label, alignment in [(0, "Cohort", "left"), (.74, "n", "right"), (.98, "Inputs", "right")]:
        text.text(xpos, 6.95, label, ha=alignment, weight="bold", va="center")
    left = fig.add_axes([.263, .121, .365, .394])
    forest_axis(left, (-.013, .082), (-.58, 6.70), .02)
    left.set_xlabel("Frozen score − comparator (ΔHarrell C)", labelpad=5)
    right = fig.add_axes([.727, .121, .230, .394], sharey=left)
    right.spines.left.set_visible(False)
    right.set_xticks([0, .05, .10, .15])
    right.set_xlim(0, .165)
    right.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.set_xlabel("All-cause mortality risk (%)", labelpad=5)
    for plot in (left, right, text):
        plot.axhspan(-.58, 1.48, facecolor=SOFT, zorder=-1)
        plot.axhline(5.5, color=RULE, linewidth=.62, zorder=-1)
    for y, (key, name) in zip(range(6, -1, -1), COMMUNITY.items()):
        frozen = data.loc[(key, "full_lp")]
        text.text(0, y, name, va="center", fontweight="bold" if key == "temporal" else "normal")
        text.text(.74, y, f"{int(frozen.n):,}", ha="right", va="center")
        text.text(.98, y, f"{int(frozen.median_features)}", ha="right", va="center")
        for model, color, marker, offset in [("ridge_lp", BLUE, "o", .125),
                                             ("age_sex_lp", ORANGE, "s", -.125)]:
            val = data.loc[(key, model)]
            whisker(left, val.clock_minus_model, y+offset, val.delta_low, val.delta_high, color, marker)
        cell = calibration.loc[key]
        right.plot([cell.observed_KM, cell.predicted], [y, y], color=RULE, lw=1.05, zorder=1)
        right.plot(cell.observed_KM, y, "o", color=BLUE, markersize=4.2, zorder=3)
        right.plot(cell.predicted, y, "s", color=ORANGE, markerfacecolor="none", markersize=4.2, zorder=3)
    fig.text(.036, .037, "Shaded rows show low-coverage panels. Measures are median observed inputs per participant.", color=GRAY)
    export(fig, 1)


def fig2():
    table = pd.read_csv(SUP / "models" / "comparisons.csv")
    primary = table.query("setting == 'primary' and model in ['M1_nomask', 'M2_panel']").copy()
    np.testing.assert_array_equal(primary.n, 25614)
    np.testing.assert_array_equal(primary.deaths, 1977)
    np.testing.assert_array_equal(primary.bootstrap, 2000)
    primary = primary.set_index(["panel", "model"], verify_integrity=True)
    effects = pd.read_csv(SUP / "models" / "community_random_effects.csv").set_index("model")
    methods = {"M3_XGB_raw": "XGBoost",
               "M3_XGB_aug": "Masked XGBoost",
               "M2_panel": "Panel-specific Transformer",
               "M4_RSF": "Random survival forest"}
    rng = pd.read_csv(SUP / "models" / "random_panel_richness.csv")
    assert rng.seed.nunique() == 3 and rng["n"].eq(25614).all()
    pivot = rng.pivot(index=["seed", "available_assays"], columns="model", values="c_index")
    rng_models = {
        "M1_nomask": ("No training mask", BLUE, "o"),
        "M3_XGB_raw": ("XGBoost", GRAY, "^"),
        "M3_XGB_aug": ("Masked XGBoost", ORANGE, "s"),
    }
    ranges = pd.concat([
        (pivot.M0 - pivot[key]).groupby("available_assays").agg(["mean", "min", "max"]).assign(comparator=key)
        for key in rng_models
    ]).reset_index()
    format_source(primary.reset_index().drop(columns="hash"), "Fig2a_source.csv")
    format_source(effects.loc[list(methods)].reset_index(), "Fig2b_source.csv")
    format_source(ranges, "Fig2c_source.csv")
    fig = canvas(143)
    header(fig, .036, .960, "a", "Fixed panels on the same NHANES holdout participants")
    fig.legend(handles=[point_key(BLUE, "o", "vs. no training mask"),
                        point_key(ORANGE, "s", "vs. panel-specific Transformer")],
               loc="upper left", bbox_to_anchor=(.050, .921), borderaxespad=0, ncols=2,
               columnspacing=1.2, handlelength=1.15)
    ax = fig.add_axes([.231, .530, .720, .311])
    forest_axis(ax, (-.012, .0115), (-.6, 6.65), .005)
    ax.axvline(-.003, color=RULE, lw=.9, ls=(0, (1.1, 2.0)), zorder=1)
    ax.text(-.003, 6.78, "−0.003 margin", ha="center", va="bottom")
    ax.set_xlabel("Frozen score − comparator (ΔHarrell C)", labelpad=4)
    txt = fig.add_axes([.036, .530, .173, .311], frameon=False)
    txt.set(xlim=(0, 1), ylim=(-.6, 6.65))
    txt.axis("off")
    for y, (panel, label) in zip(range(6, -1, -1), SIMULATED.items()):
        size = primary.loc[(panel, "M1_nomask"), "panel_size"]
        txt.text(0, y, label, va="center")
        txt.text(1, y, f"{size:d} items", ha="right", va="center")
        if 6 <= size <= 20:
            for to_shade in (ax, txt):
                to_shade.axhspan(y-.50, y+.50, facecolor=SOFT, zorder=-1)
        for model, color, marker, offset in [("M1_nomask", BLUE, "o", .13),
                                             ("M2_panel", ORANGE, "s", -.13)]:
            if (panel, model) in primary.index:
                row = primary.loc[(panel, model)]
                whisker(ax, row.delta_M0_minus_comparator, y+offset, row.ci_low, row.ci_high, color, marker)

    header(fig, .036, .405, "b", "Observed panels across four cohorts")
    header(fig, .570, .405, "c", "Random assay subsets on the same people")
    b = fig.add_axes([.236, .103, .202, .229])
    forest_axis(b, (-.014, .023), (-.59, 3.55), .01)
    b.set_xlabel("Pooled ΔHarrell C", labelpad=4)
    b.axhspan(1.48, 2.50, zorder=-1, facecolor=SOFT)
    names = fig.add_axes([.036, .103, .190, .229], frameon=False)
    names.set(xlim=(0, 1), ylim=(-.59, 3.55))
    names.axis("off")
    iv = fig.add_axes([.456, .103, .050, .229], frameon=False)
    iv.set(xlim=(0, 1), ylim=(-.59, 3.55))
    iv.axis("off")
    iv.text(.98, 3.78, "I²", ha="right", va="center", weight="bold")
    for y, (method, name) in zip(range(3, -1, -1), methods.items()):
        row = effects.loc[method]
        assert row.k == 4
        names.text(0, y, name, va="center")
        iv.text(.98, y, f"{row.I2:.0f}%", ha="right", va="center")
        whisker(b, row.delta, y, row.low, row.high,
                ORANGE if method == "M3_XGB_aug" else BLUE, "D")

    c = fig.add_axes([.610, .103, .342, .229])
    c.set_xlabel("Observed measures retained per participant", labelpad=4)
    c.set_ylabel("ΔHarrell C")
    c.spines.left.set_visible(True)
    c.set_xticks([1, 3, 6, 12, 18, 22])
    c.yaxis.set_major_locator(MultipleLocator(.005))
    c.axhline(0, color=GRAY, lw=.75, linestyle=(0, (3, 3)), zorder=1)
    for model, (name, color, marker) in rng_models.items():
        line = ranges.loc[ranges.comparator == model]
        x = line.available_assays.to_numpy()
        c.fill_between(x, line["min"].to_numpy(), line["max"].to_numpy(), color=color,
                       alpha=.12, linewidth=0)
        c.plot(x, line["mean"], color=color, label=name, marker=marker,
               markersize=3.4, linewidth=1.05)
    c.legend(handles=[point_key(color, marker, label)
                      for label, color, marker in rng_models.values()],
             loc="upper center", bbox_to_anchor=(.53, 1.20), ncols=3,
             handlelength=.9, handletextpad=.35, columnspacing=.6, borderaxespad=0)
    export(fig, 2)


if __name__ == "__main__":
    fig1()
    fig2()
