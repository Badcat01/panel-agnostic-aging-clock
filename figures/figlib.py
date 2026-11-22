"""Shared style, millimetre layout engine and exporter for all manuscript figures.

Every panel is a function draw(B) that builds itself inside a Box B (local mm
coordinates measured from the panel's top-left corner).  The same function is
used for the composite figure and for the stand-alone panel file, so fonts,
sizes and positions are identical in both.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

MM = 1 / 25.4
FIG_W = 183.0  # Nature double-column width (mm)

# ----------------------------------------------------------------- typography
for f in ["/System/Library/Fonts/Supplemental/Arial.ttf",
          "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
          "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
          "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)

FS_TITLE = 7.0
FS_LABEL = 6.5
FS_TICK = 6.0
FS_LEG = 6.0
FS_ANN = 5.6
FS_LETTER = 9.0

INK = "#1a1a1a"
INK2 = "#555555"
GRID = "#e9e9e9"
REF = "#9a9a9a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Arial Unicode MS", "Helvetica", "DejaVu Sans"],
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEG,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "axes.labelpad": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.0,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.default": "regular",
})

# --------------------------------------------------------------------- colour
# validated with the dataviz palette validator (adjacent CVD dE >= 8)
C_CLOCK = "#2a78d6"     # clinical clock (FT-Mask)
C_RIDGE = "#eb6834"     # same-input ridge Cox / main comparator
C_AGESEX = "#8a8984"    # age + sex reference (neutral)
C_DNAM = "#6250d6"      # DNA methylation clocks
C_JOINT = "#199e70"     # clinical + DNAm
C_FEMALE = "#c4426f"
C_MALE = "#3a6fb0"
ORGAN_C = {"full": "#2a78d6", "liver": "#c98500", "kidney": "#6250d6",
           "metabolic": "#eb6834", "immune_heme": "#199e70", "cardiovascular": "#d43c3b"}
ORGAN_L = {"full": "Whole body", "liver": "Liver", "kidney": "Kidney", "metabolic": "Metabolic",
           "immune_heme": "Immune/blood", "cardiovascular": "Cardiovascular"}
QUART = ["#a7c9f0", "#62a0e6", "#2a78d6", "#123f7a"]  # sequential (one hue)
LIGHT = "#f3f5f8"


class Box:
    """A panel region inside a figure; all coordinates in mm from its top-left."""

    def __init__(self, fig, x, y, w, h, W, H, top=12.5, bottom=10.5):
        self.fig, self.x, self.y, self.w, self.h, self.W, self.H = fig, x, y, w, h, W, H
        self.top, self.bottom = top, bottom

    def plot(self, left, right=3.0, top=None, bottom=None, **kw):
        """Plot area aligned to the row grid: shared top/bottom for every panel in a row."""
        t = self.top if top is None else top
        b = self.bottom if bottom is None else bottom
        return self.ax(left, t, self.w - left - right, self.h - t - b, **kw)

    def rowy(self, i, n, pad=0.6, top=None, bottom=None):
        """Local mm y-coordinate of categorical row i (matches yrows limits) in the plot area."""
        t = self.top if top is None else top
        b = self.bottom if bottom is None else bottom
        return t + (i + pad) / (n - 1 + 2 * pad) * (self.h - t - b)

    def rect(self, l, t, w, h):
        return [(self.x + l) / self.W, 1 - (self.y + t + h) / self.H, w / self.W, h / self.H]

    def ax(self, l, t, w, h, **kw):
        a = self.fig.add_axes(self.rect(l, t, w, h), **kw)
        return a

    def fx(self, lx):  # local mm -> figure fraction
        return (self.x + lx) / self.W

    def fy(self, ty):
        return 1 - (self.y + ty) / self.H

    def text(self, lx, ty, s, **kw):
        kw.setdefault("fontsize", FS_ANN)
        kw.setdefault("color", INK)
        return self.fig.text(self.fx(lx), self.fy(ty), s, **kw)

    def title(self, s, ty=1.2):
        """Single-line centred panel title (no subtitle)."""
        return self.fig.text(self.fx(self.w / 2), self.fy(ty), s, ha="center", va="top",
                             fontsize=FS_TITLE, fontweight="bold", color=INK)

    def legend(self, handles, labels, ty=6.4, ncol=None, cx=None, **kw):
        """Legend in its own band directly under the title, centred; never over data."""
        ncol = ncol or len(handles)
        cx = self.w / 2 if cx is None else cx
        return self.fig.legend(handles, labels, loc="upper center",
                               bbox_to_anchor=(self.fx(cx), self.fy(ty)), ncol=ncol,
                               fontsize=FS_LEG, frameon=False, **kw)


def style_axis(ax, grid="x"):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def letter(fig, B, ch):
    fig.text(B.fx(0.0), B.fy(0.6), ch, ha="left", va="top", fontsize=FS_LETTER,
             fontweight="bold", color=INK)


def save_all(fig, base, formats=("pdf", "png", "tiff", "svg"), dpi=600):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for fmt in formats:
        kw = {}
        if fmt == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(f"{base}.{fmt}", dpi=dpi if fmt in ("png", "tiff") else None, **kw)


def build(panels, H, out_fig, out_panel_dir, fig_id, W=FIG_W, panel_formats=("pdf", "png", "svg"),
          fig_formats=("pdf", "png", "tiff")):
    """panels: list of (letter, draw_fn, x, y, w, h[, {"top":..,"bottom":..}]) in mm."""
    fig = plt.figure(figsize=(W * MM, H * MM))
    for spec in panels:
        ch, fn, x, y, w, h = spec[:6]; g = spec[6] if len(spec) > 6 else {}
        B = Box(fig, x, y, w, h, W, H, **g)
        fn(B)
        letter(fig, B, ch)
    save_all(fig, out_fig, fig_formats)
    plt.close(fig)
    # stand-alone panels (no panel letter; identical geometry)
    for spec in panels:
        ch, fn, x, y, w, h = spec[:6]; g = spec[6] if len(spec) > 6 else {}
        f2 = plt.figure(figsize=(w * MM, h * MM))
        B = Box(f2, 0, 0, w, h, w, h, **g)
        fn(B)
        save_all(f2, os.path.join(out_panel_dir, f"{fig_id}{ch}"), panel_formats)
        plt.close(f2)


# ------------------------------------------------------------ plotting helpers
def ci_dot(ax, y, est, lo, hi, color, marker="o", ms=3.6, lw=0.9, label=None, z=3, mfc=None):
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=z)
    hollow = mfc is not None and mfc != color
    ax.plot([est], [y], marker=marker, ms=ms, color=color, mfc=mfc if mfc else color,
            mec=color if hollow else "white", mew=0.8 if hollow else 0.5, ls="none", zorder=z + 1, label=label)


def handle(color, marker="o", ls="-", ms=3.6, mfc=None, lw=0.9):
    from matplotlib.lines import Line2D
    hollow = mfc is not None and mfc != color
    return Line2D([0], [0], color=color, marker=marker, ls=ls, ms=ms, lw=lw,
                  mfc=mfc if mfc else color, mec=color if hollow else "white", mew=0.8 if hollow else 0.5)


def patch(color, ec="none"):
    return Rectangle((0, 0), 1, 1, fc=color, ec=ec, lw=0.5)


def rounded_box(ax, x, y, w, h, fc, ec, lw=0.6, r=0.012):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, transform=ax.transAxes, zorder=2)
    ax.add_patch(p)
    return p


def fmt_n(n):
    return f"{int(n):,}"
