"""Audit, link and score HRS Core 2016 pre-draw vital signs with the frozen model."""

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyreadstat
import torch

HERE = Path(__file__).resolve().parent
SUPPLEMENT = HERE.parents[1]
sys.path[:0] = [str(SUPPLEMENT / "technical"), str(SUPPLEMENT.parent / "nhanes3_organ")]
from shared import BUNDLE, EXT, formal, sha
from organ_score import score_frame, verify_reference

from study_paths import community_dir

HRS = community_dir("HRS")
VITALS = {"sbp": ["PI859", "PI864", "PI869"], "dbp": ["PI860", "PI865", "PI870"],
          "pulse": ["PI861", "PI866", "PI871"]}
WINDOWS = (0, 7, 30, 90, 180)
PANELS = ["liver", "kidney", "metabolic", "immune_heme", "cardiovascular"]


def indexed_records(filename, variables):
    path = next(HRS.rglob(filename))
    loader = {".sav": pyreadstat.read_sav, ".dta": pyreadstat.read_dta,
              ".sas7bdat": pyreadstat.read_sas7bdat}[path.suffix]
    rows, _ = loader(path, usecols=["HHID", "PN", *variables])
    keys = rows.pop("HHID").str.strip().str.zfill(6) + "-" + rows.pop("PN").str.strip().str.zfill(3)
    return rows.set_index(keys.rename("ID"), verify_integrity=True), path


@lru_cache(maxsize=2)
def assemble(device=None):
    """Return in-memory original scores, cardiovascular estimate, lag and assays."""
    _, scores = formal("hrs")
    assert scores.cardiovascular_accel.isna().all()
    source, vital_path = indexed_records("H16I_R.dta", ["PI854", *sum(VITALS.values(), [])])
    vbs, vbs_path = indexed_records("hrs2016vbs.sav", ["PVBS_N_DAYS", "PNTBNPE", "PCMVGE"])
    blocks = {name: source[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
              for name, cols in VITALS.items()}
    valid = (
        (blocks["sbp"] >= 60) & (blocks["sbp"] <= 300)
        & (blocks["dbp"] >= 30) & (blocks["dbp"] <= 180)
        & (blocks["pulse"] >= 25) & (blocks["pulse"] <= 220)
        & (blocks["sbp"] > blocks["dbp"])
        & source.PI854.astype(str).eq("1").to_numpy()[:, None]
    )
    vitals = pd.DataFrame({name: pd.DataFrame(np.where(valid, matrix, np.nan)).mean(axis=1).to_numpy()
                           for name, matrix in blocks.items()}, index=source.index)
    vitals["complete_triplets"] = valid.sum(axis=1)
    combined = scores.join(vbs, validate="one_to_one").join(vitals, validate="one_to_one")
    combined["lag_days"] = combined.PVBS_N_DAYS
    keep = combined.complete_triplets.gt(0) & combined.lag_days.between(0, max(WINDOWS))
    features = combined.loc[keep, ["age", *VITALS]]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    reference = verify_reference()
    predicted = score_frame(features, BUNDLE, device)
    assert predicted.cardiovascular_n_observed.eq(3).all()
    combined["cardiovascular_accel"] = predicted.cardiovascular_accel.reindex(combined.index)
    assert combined.cardiovascular_accel.notna().equals(keep)
    combined.attrs["device"] = device
    combined.attrs["source_sha256"] = {f.name: sha(f) for f in (vital_path, vbs_path, EXT / "results/hrs/predictions.csv.gz")}
    combined.attrs["model_freeze_sha256"] = reference["model_freeze_sha256"]
    return combined
