"""Re-score NHANES III with assay-specific masks, retaining the same participants."""

import json

import pandas as pd
import torch

from reference import BUNDLE, REFERENCE, ROOT, score_frame, save_json, sha, verify
from validate_external import discrimination, fixed_time


def main():
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    verify()
    source_path = ROOT / "inputs" / "nhanes3.csv.gz"
    output = ROOT / "results" / "nhanes3"
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if metrics["source_sha256"] != sha(source_path) or metrics["reference_sha256"] != sha(REFERENCE):
        raise ValueError("Run validate_external.py on the intended NHANES III input before sensitivity analyses")

    data = pd.read_csv(source_path, dtype={"ID": "string"})
    pred = pd.read_csv(output / "predictions.csv.gz", dtype={"ID": "string"})
    if not data.ID.equals(pred.ID) or data.ID.duplicated().any():
        raise ValueError("NHANES III source/prediction identity mismatch")
    pd.testing.assert_frame_equal(data[["sex", "time_yr", "event"]], pred[["sex", "time_yr", "event"]])

    manual_diff = ["mono_pct", "neut_pct", "eos_pct"]
    scenarios = {
        "manual_diff_omitted": manual_diff,
        "crp_omitted": ["crp"],
        "manual_diff_and_crp_omitted": [*manual_diff, "crp"],
    }
    summary = {"note": "Post hoc sensitivity. Frozen model, reference baseline hazard, and participants fixed.",
               "n": len(pred), "deaths": int(pred.event.sum()), "scenarios": {}}
    for scenario, omitted in scenarios.items():
        new = score_frame(data.drop(columns=omitted), BUNDLE, "cuda")
        if not new.ID.equals(pred.ID) or not new.full_lp.notna().all():
            raise ValueError("Panel masking lost participants or reordered predictions")
        d = pred.assign(masked_lp=new.full_lp)
        comparison = discrimination(d, ["full_lp", "masked_lp"])
        evidence = {
            "omitted": omitted,
            "C_original": comparison["full_lp"]["C"],
            "C_masked": comparison["masked_lp"]["C"],
            "C_masked_CI95": comparison["masked_lp"]["CI95"],
            "C_original_minus_masked_CI95": comparison["masked_lp"]["delta_CI95"],
        }
        values, _ = fixed_time(d.assign(full_lp=d.masked_lp), reference)
        evidence["horizons"] = {
            h: {"predicted_masked": row["clock"]["predicted_mean"], "observed_KM": row["observed_km"]}
            for h, row in values.items() if row["supported"]
        }
        summary["scenarios"][scenario] = evidence
    save_json(output / "assay_mask_sensitivity.json", summary)
    for name, values in summary["scenarios"].items():
        print(name, values["C_masked"], flush=True)


if __name__ == "__main__":
    main()
