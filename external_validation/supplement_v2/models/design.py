"""Source-anchored panel definitions for prospective supplemental comparisons."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
EXT = OUT.parents[1]
PROJECT = EXT.parent
RUN = PROJECT / "runs/nhanes_gpu_20261121"
sys.path.insert(0, str(PROJECT / "scripts"))
import train_clock_gpu as original
from clock_network import FEATS


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def panels():
    masks = {"full": FEATS}
    for name, n in (("nhanes3", 40), ("hrs", 31), ("mimic", 19), ("charls", 18), ("elsa", 12), ("share", 6), ("klosa",1)):
        if name == "mimic":
            count = read(EXT / "mimic_iv/formal_v3_1/audit.json")["features_coverage"]
        else:
            cov = pd.read_csv(EXT / "results" / name / "coverage.csv").set_index("feature")
            count = cov.baseline_n.to_dict()
        by_frequency = sorted(FEATS, key=lambda f: (-count.get(f, 0), FEATS.index(f)))
        selected = by_frequency[:n]
        assert min(count[f] for f in selected) > 0
        masks[name] = [f for f in FEATS if f in selected]
    return masks


def develop():
    d, t, _ = original.get_cohorts()
    ids = hashlib.sha256(d.SEQN.to_numpy(dtype="<i8").tobytes()).hexdigest()
    assert ids == read(RUN / "protocol.json")["train_ids_sha256"] and len(t) == 25614
    frozen = read(RUN / "model_freeze.json")["files_sha256"]
    for name, expected in frozen.items():
        assert original.sha(RUN / "bundle" / name) == expected
    return d, t


def define():
    original.save_json(OUT / "panel_definitions.json", {
        "order_rule": "Most frequently available at baseline; ties ordered by original frozen feature order. MIMIC 19 and NHANES III 40 are fixed proxy subsets of 40 and 43 available types, not literal participant-specific profiles.",
        "panels": panels(),
        "reference": original.sha(RUN / "model_freeze.json"),
        "temporal": "All primary comparisons use a common cohort with >=1 actually measured marker on every simulated panel, retain natural NHANES missingness.",
        "training": "Same 5 development cycles excluding methylation participants, original outer leave-one / inner leave-two cycles, all hyperparameters selected on development only. FT grid lr [.0005,.001] x steps [32,64,128], 5 independent final seeds.",
    })


if __name__ == "__main__":
    define()
