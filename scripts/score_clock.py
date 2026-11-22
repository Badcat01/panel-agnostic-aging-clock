"""Score a frozen mortality clock on a CSV in the NHANES training units."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clock_network import FEATS, GROUPS, LOGF, FTMask


def numeric_column(series):
    """Allow missing measurements, but reject invalid or infinite values."""
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{series.name} contains a nonnumeric value") from exc
    if np.isinf(values).any():
        raise ValueError(f"{series.name} contains an infinite value")
    return values


def prepare_features(frame, bundle):
    """Apply the frozen transformations shared by scoring and the linear benchmark."""
    features = bundle["features"]
    if list(features) != list(FEATS) or set(bundle["log_features"]) != set(LOGF):
        raise ValueError("Bundle feature definitions differ from clock_network")
    if not frame.columns.is_unique:
        raise ValueError("Input column names must be unique")
    if "age" not in frame:
        raise ValueError("Input must contain age in years")
    age = numeric_column(frame["age"])
    if not np.isfinite(age).all():
        raise ValueError("Every row must contain a finite age in years")

    mean = np.asarray(bundle["mean"], dtype=np.float64)
    std = np.asarray(bundle["std"], dtype=np.float64)
    age_mean, age_std = float(bundle["age_mean"]), float(bundle["age_std"])
    if mean.shape != (len(features),) or std.shape != mean.shape:
        raise ValueError("Bundle normalization dimensions differ from its feature list")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid bundle biomarker normalization")
    if not np.isfinite([age_mean, age_std]).all() or age_std <= 0:
        raise ValueError("Invalid bundle age normalization")

    x = np.full((len(frame), len(features)), np.nan, dtype=np.float64)
    for j, name in enumerate(features):
        if name in frame:
            x[:, j] = numeric_column(frame[name])
            if name in bundle["log_features"]:
                x[:, j] = np.log(np.maximum(x[:, j], 1e-3))
    observed = np.isfinite(x)
    x = np.nan_to_num(((x - mean) / std).clip(-5, 5), nan=0).astype(np.float32)
    z_age = ((age - age_mean) / age_std).astype(np.float32)
    return x, observed, z_age


@torch.inference_mode()
def score_frame(frame, bundle_dir, device="cpu"):
    """Return ensemble predictions in source row order, including observation counts."""
    bundle_dir = Path(bundle_dir)
    bundle = json.loads((bundle_dir / "clock_bundle.json").read_text(encoding="utf-8"))
    x, observed, z_age = prepare_features(frame, bundle)
    age = frame["age"].to_numpy(dtype=np.float64)
    features = bundle["features"]

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    models = []
    for filename in bundle["models"]:
        model = FTMask(len(features), **bundle["model_kwargs"])
        model.load_state_dict(torch.load(bundle_dir / filename, map_location="cpu", weights_only=True))
        models.append(model.to(device).eval())
    if not models:
        raise ValueError("Bundle contains no model checkpoints")

    output = frame[[c for c in ("SEQN", "ID") if c in frame]].copy()
    output["age"] = age
    for panel, biomarker_names in {"full": features, **GROUPS}.items():
        panel_mask = np.isin(features, biomarker_names)
        num_observed = observed[:, panel_mask].sum(axis=1)
        eligible_rows = np.flatnonzero(num_observed)
        lp = np.full(len(frame), np.nan, dtype=np.float64)
        for start in range(0, len(eligible_rows), 512):
            rows = eligible_rows[start:start + 512]
            values = torch.from_numpy(x[rows]).to(device)
            mask = torch.from_numpy((observed[rows] & panel_mask).astype(np.float32)).to(device)
            ages = torch.from_numpy(z_age[rows]).to(device)
            preds = torch.stack([model(values, mask, ages) for model in models])
            lp[rows] = preds.mean(dim=0).cpu().numpy()

        cal = bundle["calibration"][panel]
        ba = cal["age_mean"] + (lp - cal["lp_mean"]) * cal["scale_years"]
        output[f"{panel}_n_observed"] = num_observed
        output[f"{panel}_no_observations"] = num_observed == 0
        output[f"{panel}_lp"] = lp
        output[f"{panel}_ba"] = ba
        output[f"{panel}_accel"] = ba - (cal["residual_intercept"] + cal["residual_slope"] * age)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Directory containing clock_bundle.json")
    parser.add_argument("--input", required=True, type=Path, help="CSV with age and available biomarker columns")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    frame = pd.read_csv(args.input, dtype={"SEQN": str, "ID": str})
    scores = score_frame(frame, args.bundle, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
