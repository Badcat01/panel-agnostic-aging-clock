"""Apply the released FT-Mask ensemble and NHANES development 50/10 scale."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from score_clock import score_frame
from verify_frozen import BUNDLE, SCALE, verify


def standardize(scored, ref):
    result = scored[[column for column in ["ID", "SEQN", "age"] if column in scored]].copy()
    for panel, params in ref["panels"].items():
        accel = scored[f"{panel}_accel"]
        number = scored[f"{panel}_n_observed"]
        if not np.array_equal(accel.notna(), number.gt(0)):
            raise ValueError(f"Missingness does not match the measured inputs for {panel}")
        result[f"{panel}_score"] = ref["score_center"] + ref["points_per_sd"] * (
            (accel - params["accel_mean_years"]) / params["accel_sd_years"]
        )
        result[f"{panel}_risk_equivalent_age"] = scored[f"{panel}_ba"]
        result[f"{panel}_accel_years"] = accel
        result[f"{panel}_n_observed"] = number
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ")
    verify()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    frame = pd.read_csv(args.input, dtype={"ID": str, "SEQN": str})
    reference = json.loads(SCALE.read_text(encoding="utf-8"))
    result = standardize(score_frame(frame, BUNDLE, args.device), reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
