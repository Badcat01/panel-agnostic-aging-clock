"""Assemble same-person incident ADL, optional official Langa–Weir impairment and chronic-condition counts."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tracker import sha

HERE=Path(__file__).resolve().parent
HEALTH=HERE.parent/"healthspan"
CONDS=["hibpe","diabe","cancre","lunge","hearte","stroke","arthre","psyche"]


def binary(visits,base,follow,col):
    risk=visits.loc[visits.wave.eq(visits.wave.min())&visits[col].eq(0),"ID"]
    fb=follow.loc[follow.ID.isin(risk)&follow[col].notna()].copy()
    seen=fb.groupby("ID").time.max()
    pos=fb.loc[fb[col].eq(1)].groupby("ID").time.min()
    b=base.loc[base.index.isin(risk)].copy()
    death=b.death.eq(1)&b.stop.le(b.horizon)
    b["observed"]=seen
    b["first"]=pos
    # A positive interview after the best death date is not treated as a prospective event.
    inc=b["first"].notna()&(~death|b["first"].le(b.stop))
    state=np.where(inc,1,np.where(death,2,0))
    t=np.where(inc,b["first"],np.where(death,b.stop,b.observed))
    x=pd.DataFrame({"time":t,"status":state},index=b.index)
    return x.where(x.time.gt(0))


def run(tracker_path,output,langa=None,visits_path=None,baseline_path=None):
    visits_path=visits_path or HEALTH/"hrs_visits.csv.gz"
    baseline_path=baseline_path or HEALTH/"hrs_baseline.csv.gz"
    d=pd.read_csv(visits_path,dtype={"ID":"string"})
    b=pd.read_csv(baseline_path,dtype={"ID":"string"}).set_index("ID",verify_integrity=True)
    ep=pd.read_csv(tracker_path,dtype={"HHID":"string","PN":"string"})
    ep.index=ep.HHID.str.zfill(6)+"-"+ep.PN.str.zfill(3)
    assert ep.index.is_unique and not d.duplicated(["ID","wave"]).any()
    b=b[["education","horizon"]].join(ep[["event","time_yr","time_upper_yr"]],validate="one_to_one")
    b=b.rename(columns={"event":"death","time_yr":"stop"})
    if langa:
        cog=pd.read_csv(langa,dtype={"ID":"string"})
        assert set(cog)>=set(["ID","wave","lw"]) and set(cog.lw.dropna()).issubset({0,1})
        d=d.merge(cog[["ID","wave","lw"]],on=["ID","wave"],how="left",validate="one_to_one")
    d=d.merge(b,on="ID",validate="many_to_one")
    valid=d.time.gt(0)&d.time.le(d.horizon)&d.death.notna()
    valid&=(d.death.eq(0)&d.time.le(d.stop)) | (d.death.eq(1)&d.time.le(d.time_upper_yr))
    f=d.loc[valid]
    out=b[["education"]].copy()
    for col in ["adl"]+(["lw"] if langa else []):
        data=binary(d,b,f,col)
        out=out.join(data.rename(columns={"time":"ti_"+col,"status":"st_"+col}))
    initial=d[d.wave.eq(d.wave.min())].set_index("ID")[CONDS].dropna()
    f=f.loc[f.ID.isin(initial.index)].dropna(subset=CONDS).copy()
    f=f.loc[f.death.eq(0)|f.time.le(f.stop)]
    for col in CONDS:
        f[col]=f[col].eq(1)&f.ID.map(initial[col]).eq(0)
    summary=f.groupby("ID")[CONDS].max().sum(axis=1).rename("new_conditions")
    out=out.join(summary).join(f.groupby("ID").time.max().rename("years_at_new_conditions"))
    out.insert(0,"PN",out.index.str.split("-").str[1]);out.insert(0,"HHID",out.index.str.split("-").str[0])
    out.to_csv(output,index=False,compression={"method":"gzip","mtime":0})
    info=dict(tracker_sha256=sha(tracker_path),visits_sha256=sha(visits_path),
            baseline_sha256=sha(baseline_path),langa_sha256=sha(langa) if langa else None,
            official_langa_status="included" if langa else "unavailable; no proxy substituted",n=len(out),
            adl_risk_and_followup_n=int(out.st_adl.notna().sum()),n_chronic_count=int(out.new_conditions.notna().sum()),
            valid_chronic_codes=CONDS,output_sha256=sha(output))
    Path(str(output)+".json").write_text(json.dumps(info,indent=2),encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("tracker",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--langa",type=Path,help="Verified official self/proxy 0/1 in ID,wave,lw format")
    p.add_argument("--visits",type=Path)
    p.add_argument("--baseline",type=Path)
    a=p.parse_args();run(a.tracker,a.output,a.langa,a.visits,a.baseline)
