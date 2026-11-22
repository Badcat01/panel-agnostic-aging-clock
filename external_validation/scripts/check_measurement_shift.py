"""Describe measurement shifts against the fixed NHANES transforms; exploratory ablations."""
import json
import numpy as np
import pandas as pd
import torch

from reference import ROOT, BUNDLE, REFERENCE, verify, score_frame, save_json
from validate_external import discrimination, fixed_time


def main():
    bundle = verify()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    ref = json.loads(REFERENCE.read_text(encoding="utf8"))
    rows, checks = [], {}
    for path in sorted((ROOT/"inputs").glob("*.csv.gz")):
        cohort = path.name.split(".")[0]
        x = pd.read_csv(path, dtype={"ID":"string"})
        for k, name in enumerate(bundle["features"]):
            if name not in x or not x[name].notna().any():
                continue
            a = x[name].dropna()
            v = np.log(a.clip(lower=.001)) if name in bundle["log_features"] else a
            z = (v-bundle["mean"][k])/bundle["std"][k]
            rows.append(dict(cohort=cohort, feature=name, n_observed=len(a), median=float(a.median()),
                             median_training_z=float(z.median()), fraction_clipped=float(z.abs().gt(5).mean())))
        if cohort not in {"hrs","mhas"}:
            continue
        feature = {"hrs":"mpv","mhas":"crp"}[cohort]
        original = pd.read_csv(ROOT/"results"/cohort/"predictions.csv.gz",dtype={"ID":"string"})
        masked = score_frame(x.drop(columns=feature),BUNDLE,"cuda")
        assert original.ID.equals(masked.ID)
        d = original.copy()
        d["masked_lp"] = masked.full_lp
        d = d.loc[d.time_yr.gt(0)&d.event.notna()&d.sex.notna()&d.masked_lp.notna()].copy()
        c = discrimination(d,["full_lp","masked_lp"])
        original_time,_ = fixed_time(d,ref)
        d["full_lp"] = d.masked_lp
        masked_time,_ = fixed_time(d,ref)
        checks[cohort] = dict(omitted_feature=feature, n=len(d), deaths=int(d.event.sum()),
                              C_original=c["full_lp"]["C"], C_masked=c["masked_lp"]["C"],
                              original_minus_masked_CI95=c["masked_lp"]["delta_CI95"], horizons={})
        for h,y in masked_time.items():
            if y["supported"]:
                checks[cohort]["horizons"][h] = dict(original_predicted=original_time[h]["clock"]["predicted_mean"],
                        masked_predicted=y["clock"]["predicted_mean"], observed_KM=y["observed_km"])
    pd.DataFrame(rows).to_csv(ROOT/"results"/"measurement_shift.csv",index=False)
    save_json(ROOT/"results"/"exploratory_assay_masks.json",checks)


if __name__=="__main__":
    main()
