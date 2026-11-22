"""Build baseline age/sex and latest death intervals for the complete 2016 VBS roster."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyreadstat

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"scripts"))
from build_hrs_elsa import month_start
from reference import sha


def derive(vbs,tracker,cutoff,status_cols):
    keys=["HHID","PN"]
    v,_=pyreadstat.read_sav(vbs,usecols=[*keys,"PVBS_N_DAYS"])
    cols=[*keys,"PAGE","PIWYEAR","PIWMONTH","GENDER","BIRTHYR","BIRTHMO","KNOWNDECEASEDYR","KNOWNDECEASEDMO",
          "LASTALIVEYR","LASTALIVEMO",*status_cols]
    t,_=pyreadstat.read_dta(tracker,usecols=cols)
    if not set(cols).issubset(t):raise ValueError("Explicit tracker field mapping required for this version")
    d=v.merge(t,on=keys,how="left",validate="one_to_one")
    out,audit=derive_intervals(d,cutoff,status_cols)
    audit.update(vbs_sha256=sha(vbs),tracker_sha256=sha(tracker))
    return out,audit


def derive_intervals(d,cutoff,status_cols):
    """Apply the frozen death and censoring rules to an already linked VBS roster."""
    keys=["HHID","PN"]
    draw=month_start(d.PIWYEAR,d.PIWMONTH)+pd.to_timedelta(d.PVBS_N_DAYS+14,unit="D")
    born=month_start(d.BIRTHYR,d.BIRTHMO)
    end=pd.Timestamp(cutoff)
    if end!=end.normalize():raise ValueError("Date-only tracker cutoff required")
    age=(draw.dt.year-born.dt.year-(draw.dt.month<born.dt.month).astype(int)).fillna(d.PAGE)
    alive=month_start(d.LASTALIVEYR,d.LASTALIVEMO)
    yr=d.KNOWNDECEASEDYR
    known=yr.between(1900,end.year+5)
    dead_flag=d[status_cols].isin([5,6]).any(axis=1)
    missing_date=~known & dead_flag
    first=month_start(yr,d.KNOWNDECEASEDMO)
    low=first.fillna(month_start(yr,pd.Series(1,index=d.index)))
    high=(first+pd.offsets.MonthEnd(0)).fillna(month_start(yr,pd.Series(12,index=d.index))+pd.offsets.MonthEnd(0))
    after=known & low.gt(end)
    interval_cross=known & low.le(end) & high.gt(end)
    low=pd.concat([low,draw,alive],axis=1).max(axis=1)
    death=known & ~after & ~interval_cross
    conflict=death & (high<low)
    mid=low+(high-low)/2
    stop=(alive+pd.Timedelta(days=14)).clip(upper=end).where(~after,end).where(~death,mid)
    out=d[keys].copy()
    out["age"]=age;out["sex"]=d.GENDER.map({1:1,2:0});out["event"]=death.astype(float)
    unit=pd.Timedelta(days=365.25)
    out["time_yr"]=(stop-draw)/unit
    out["time_lower_yr"]=out.time_yr.where(~death,(low-draw)/unit)
    out["time_upper_yr"]=out.time_yr.where(~death,(high-draw)/unit)
    bad=missing_date|interval_cross|conflict|draw.isna()|out.time_yr.le(0)|out.time_yr.isna()
    out.loc[bad,["time_yr","time_lower_yr","time_upper_yr","event"]]=np.nan
    return out,dict(n_VBS=len(out),valid_time=int((~bad).sum()),death_missing_date=int(missing_date.sum()),
                    death_interval_crosses_cutoff=int(interval_cross.sum()),inconsistent_death_bound=int(conflict.sum()),
                    included_deaths=int(out.event.sum()),cutoff=str(end.date()),
                    status_columns=status_cols)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for x in ["vbs","tracker","output"]:p.add_argument(x,type=Path)
    p.add_argument("--cutoff",required=True);p.add_argument("--status-columns",required=True,nargs="+")
    a=p.parse_args()
    df,audit=derive(a.vbs,a.tracker,a.cutoff,a.status_columns)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.output,index=False,compression={"method":"gzip","mtime":0})
    Path(str(a.output)+".json").write_text(json.dumps(audit,indent=2),encoding="utf8")
