"""Apply the HRS VBS reference analysis unchanged, restricted to each eligible DNAm pair."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"technical"))
from hrs_references import matrix
from shared import sha


def run(path,output):
    status=json.loads((output/"run_status.json").read_text(encoding="utf8"))
    assert sha(path)==status["input_hash"]
    d=pd.read_csv(path,dtype={"ID":"string"}).set_index("ID",verify_integrity=True)
    d=d.loc[d.qc_pass.eq(1)&d.age.ge(50)&d.time_yr.gt(0)].dropna(subset=["clinical","sex","event"])
    scores=[c.removeprefix("organ_") for c in d if c.startswith("organ_")]
    d=d.rename(columns={"organ_"+v:v+"_accel" for v in scores}).rename(columns={"clinical":"full_accel"})
    combos={"cystatin":"kidney","ntprobnp":"cardiovascular","il6":"immune_heme","stnfr1":"immune_heme","cmv":"immune_heme"}
    rows=[];skipped=[]
    for clock in [c for c in d if c.startswith("clock_")]:
        linked=d.loc[d[clock].notna()]
        for marker,matched in combos.items():
            col="ref_"+marker
            if col not in linked:
                skipped.append(dict(clock=clock,marker=marker,reason="reference analyte unavailable"));continue
            if matched not in scores or not linked[matched+"_accel"].notna().any():
                skipped.append(dict(clock=clock,marker=marker,reason="matched organ has no score"))
            frame=linked.assign(age10=(linked.age-65)/10)
            frame["age10sq"]=frame.age10**2
            rows.extend(matrix(frame,col,marker,matched,clock,["age10","age10sq","sex","full_accel"],scores))
    if rows:
        tab=pd.DataFrame(rows)
        for group,ix in tab.groupby("group").groups.items():
            tab.loc[ix,"association_q_BH"]=multipletests(tab.loc[ix,"p"],method="fdr_bh")[1]
            matched=tab.index.isin(ix)&tab.panel.eq(tab.expected)&tab.dominance_p_wild_IUT.notna()
            if matched.any():tab.loc[matched,"dominance_q_BH"]=multipletests(tab.loc[matched,"dominance_p_wild_IUT"],method="fdr_bh")[1]
        tab.to_csv(output/"VBS_reference_matrix.csv",index=False)
    if skipped:pd.DataFrame(skipped).to_csv(output/"VBS_reference_unavailable.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path);parser.add_argument("output",type=Path)
    a=parser.parse_args()
    with threadpool_limits(limits=1):run(a.input,a.output)
