"""Exercise shared household, missing organ, and dual-target B2M logic with synthetic people."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from proxy_organ import MATCHES, run


def main(fixture):
    d = pd.read_csv(fixture, dtype={"ID": str})
    rng = np.random.default_rng(142)
    for marker in MATCHES:
        d[marker] = rng.normal(size=len(d))
    d["proxy_cystatin"] += 2*d.organ_kidney
    d["proxy_b2m"] += 1.4*d.organ_kidney + 1.4*d.organ_immune_heme
    d.loc[d.index[:11], "proxy_loga1c"] = np.nan
    d.loc[d.index[:9], "organ_liver"] = np.nan
    d.loc[d.index[-3:], "clock_GrimAge2"] = np.nan
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        source = output / "paired.csv.gz"
        d.to_csv(source, index=False)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        (output / "run_status.json").write_text(json.dumps({"input_hash": digest}))
        run(source, output)
        result = pd.read_csv(output / "dnam_proxy_organ.csv")
        excluded = pd.read_csv(output / "dnam_proxy_unavailable.csv")
        assert len(result) == 32 and result.groupby("marker").n.nunique().eq(1).all()
        assert set(excluded[["marker","panel"]].itertuples(index=False,name=None)) == {
            ("proxy_pai1","cardiovascular"),("proxy_adm","cardiovascular")}
        assert set(result.loc[result.marker.eq("proxy_b2m") & result.matched,"nonmatched_panels"]) == {"liver|metabolic"}
        assert result.dominance_p_wild_IUT.notna().sum() == 7
        assert result.loc[result.marker.eq("proxy_b2m") & result.matched,"dominance_q_BH"].lt(.05).all()
        eligible = d.loc[d.qc_pass.eq(1) & d.age.ge(50) & d.time_yr.gt(0)
                         & d.clock_GrimAge2.notna() & d.organ_liver.notna()]
        row = result.loc[result.marker.eq("proxy_cystatin") & result.panel.eq("kidney")].iloc[0]
        assert row.n == len(eligible) and row.households == eligible.ID.str[:6].nunique()
        controls = np.column_stack([np.ones(len(eligible)),eligible[["age","sex","clinical"]]])
        y = eligible[["proxy_cystatin","organ_kidney"]].to_numpy()
        r = y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
        assert np.isclose(pearsonr(r[:,0],r[:,1]).statistic,row.partial_r)
        for raw, adj in [("association_p","association_q_BH"),("dominance_p_wild_IUT","dominance_q_BH")]:
            observed = result[raw].notna()
            np.testing.assert_allclose(result.loc[observed,adj],multipletests(result.loc[observed,raw],method="fdr_bh")[1])
        assert row.min_matched_advantage>0 and row.dominance_q_BH<.05
        assert "ID" not in result
        (output / "run_status.json").write_text(json.dumps({"input_hash":"incorrect"}))
        try:
            run(source, output)
        except ValueError as exc:
            assert "differs" in str(exc)
        else:
            raise AssertionError("A mismatched input hash was accepted")
    print("Synthetic proxy analysis passed: eight markers, seven estimable matches, two unavailable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1]/"synthetic/fixture.csv.gz")
    with threadpool_limits(limits=1):
        main(parser.parse_args().fixture)
