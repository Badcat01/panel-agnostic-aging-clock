"""Generate wholly synthetic families to exercise the preregistered implementation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def run(output_dir=HERE / "synthetic"):
    rng=np.random.default_rng(88913)
    n=320
    family=np.repeat(np.arange(n//2),2)
    age=rng.uniform(46,91,n)
    sex=rng.integers(0,2,n)
    base=rng.normal(size=n)
    clinical=base+rng.normal(size=n)
    dnam=base+rng.normal(size=n)
    death=rng.exponential(4*np.exp(-.45*clinical-.22*dnam-.04*(age-66)),n)
    censor=rng.uniform(.3,7,n)
    t=np.round(np.minimum(death,censor),2).clip(.01)
    events=death<=censor
    data=pd.DataFrame(dict(ID=[f"{i:06d}-{j:03d}" for i in range(n//2) for j in [1,2]],age=age,sex=sex,
        time_yr=t,event=events.astype(int),clinical=clinical,qc_pass=rng.choice([0,1],n,p=[.04,.96]),
        clock_GrimAge2=dnam,clock_Horvath=dnam+rng.normal(size=n),
        clinical_cap80=clinical-np.maximum(age-80,0)*.01,clinical_cap85=clinical-np.maximum(age-85,0)*.01,
        weight=rng.uniform(.6,1.5,n),stratum=(family//4)%8,psu=family%4,
        time_lower_yr=np.where(events,np.maximum(t-.02,.01),t),time_upper_yr=np.where(events,t+.02,t),
        ti_adl=np.minimum(t,rng.uniform(.2,2,n)),st_adl=rng.integers(0,3,n),
        ti_lw=np.minimum(t,rng.uniform(.2,2,n)),st_lw=rng.integers(0,3,n),
        new_conditions=rng.poisson(.7,n),years_at_new_conditions=np.maximum(.2,t),education=rng.integers(1,5,n)))
    for name in ["liver","kidney","immune_heme","metabolic"]:
        data["organ_"+name]=.4*clinical+rng.normal(size=n)
    data["organ_cardiovascular"]=np.nan
    for name in ["cystatin","ntprobnp","il6","stnfr1","cmv"]:
        data["ref_"+name]=rng.normal(size=n)
    vals=rng.dirichlet(np.ones(4),n)
    for i in range(4):data["cell_"+str(i)]=vals[:,i]
    data.loc[data.index[:6],"clock_Horvath"]=np.nan
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir/"fixture.csv.gz",index=False,compression={"method":"gzip","mtime":0})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"synthetic")
    run(parser.parse_args().output_dir)
