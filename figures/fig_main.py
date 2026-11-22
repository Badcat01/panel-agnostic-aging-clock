"""Shared figure output and axis helpers."""
import os

import pandas as pd

OUT = os.environ["AC_OUT"]
FIGDIR = f"{OUT}/03_Main_Figures"
PANDIR = f"{OUT}/04_Main_Figure_Panels"
SRCDIR = f"{OUT}/08_Source_Data"
os.makedirs(SRCDIR, exist_ok=True)
SOURCE = {}  # figure -> {panel: DataFrame}


def src(fig, panel, df):
    SOURCE.setdefault(fig, {})[panel] = df.reset_index(drop=True)


def write_source(fig):
    with pd.ExcelWriter(f"{SRCDIR}/Source_Data_{fig}.xlsx") as xw:
        for p, df in SOURCE.get(fig, {}).items():
            df.to_excel(xw, sheet_name=f"{fig}{p}"[:31], index=False)


def yrows(ax, labels, pad=0.6):
    n = len(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_ylim(n - 1 + pad, -pad)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
