"""Score the DNA methylation exclusion set with the verified NHANES GPU ensemble."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parents[1]
PROJECT = EXTERNAL.parent
sys.path.insert(0, str(EXTERNAL / "scripts"))
from reference import BUNDLE, TRAIN_RUN, score_frame, sha, verify
from nhanes_calibration import prepare_table

PANELS = ["liver", "kidney", "metabolic", "immune_heme", "cardiovascular"]
CLOCKS = ["HorvathAge", "HannumAge", "SkinBloodAge", "PhenoAge", "ZhangAge", "LinAge",
          "GrimAgeMort", "GrimAge2Mort", "DunedinPoAm", "HorvathTelo"]
CELLS = ["CD8TPP", "CD4TPP", "Nkcell", "Bcell", "MonoPP"]  # omit sixth compositional component
PROXIES = {"CystatinCMort": ["kidney"], "B2MMort": ["kidney","immune_heme"],
           "LeptinMort": ["metabolic"], "logA1CMort": ["metabolic"], "CRPMort": ["immune_heme"],
           "GDF15Mort": ["immune_heme"], "PAI1Mort": ["cardiovascular"], "ADMMort": ["cardiovascular"]}
SEED = 20261121


def load_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path, content):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def build():
    bundle = verify()
    protocol = load_json(TRAIN_RUN / "protocol.json")
    for name, expected in protocol["data"].items():
        assert sha(PROJECT / "data/processed" / name) == expected
    table, ids, info = prepare_table(PROJECT)
    eligible = table[table[bundle["features"]].notna().sum(axis=1) * 2 >= len(bundle["features"])]
    dev = pd.read_csv(TRAIN_RUN / "development_ids.csv.gz")
    digest = hashlib.sha256(dev.SEQN.to_numpy(dtype="<i8").tobytes()).hexdigest()
    assert digest == protocol["train_ids_sha256"]
    d = eligible[eligible.SEQN.isin(ids)].reset_index(drop=True)
    assert len(d) == 2525 and int(d.event.sum()) == 1359 and not set(dev.SEQN) & set(d.SEQN)
    marker = PROJECT / "data/raw/nhanes/dnmepi.sas7bdat"
    dnam = pd.read_sas(marker, encoding="latin1")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen score reproduction")
    preds = score_frame(d, BUNDLE, "cuda")
    assert np.array_equal(preds.SEQN, d.SEQN)
    extra = ["SEQN","male","time_yr","event","cycle","wt_mec","strata","psu"]
    merged = preds.merge(d[extra], on="SEQN", validate="one_to_one").merge(dnam, on="SEQN", validate="one_to_one")
    assert len(merged)==len(d) and merged[CLOCKS+["full_lp","WTDN4YR"]].notna().all().all() and merged.WTDN4YR.gt(0).all()
    path = HERE / "dnam_frozen_scores.csv.gz"
    merged.to_csv(path, index=False, compression={"method":"gzip","mtime":0})
    write_json(HERE / "roster_audit.json", dict(n=len(d), deaths=int(d.event.sum()), raw_dnam_measured=len(ids),
               device=torch.cuda.get_device_name(0), input_features=len(bundle["features"]),
               development_overlap=0, excluded_for_eligibility=len(ids)-len(d), model_freeze_sha256=sha(TRAIN_RUN/"model_freeze.json"),
               development_id_sha256=digest, dnam_data_sha256=sha(marker), output_sha256=sha(path),
               temporal_scores_sha256=sha(TRAIN_RUN/"temporal_predictions.csv.gz"), source_audit=info["dnam"]))
    print(len(d), int(d.event.sum()), "GPU scoring complete", flush=True)


if __name__ == "__main__":
    build()

