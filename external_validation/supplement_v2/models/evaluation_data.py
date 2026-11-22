"""Construct identical simulation cohorts; retain patient-specific missingness externally."""
import numpy as np
import pandas as pd
from design import EXT, FEATS, panels


def external(name):
    if name!="mimic":
        f = pd.read_csv(EXT/"inputs"/f"{name}.csv.gz",dtype={"ID":str})
    else:
        source = EXT/"mimic_iv/formal_v3_1"
        index = pd.read_csv(source/"landmark_index.csv.gz")
        f = index.merge(pd.read_csv(source/"lab_panel.csv.gz"),on="hadm_id",validate="one_to_one")
        vitals = pd.read_csv(source/"vitals.csv.gz", usecols=["subject_id","hadm_id","sbp","dbp","pulse","bmi"])
        f = f.merge(vitals,on=["subject_id","hadm_id"],how="left",validate="one_to_one").rename(columns={"subject_id":"ID"})
    f = f.loc[f.age.ge(20)&f.sex.isin([0,1])&f.event.isin([0,1])&f.time_yr.gt(0)].copy()
    assert f.ID.is_unique
    return f


def scenarios(temporal):
    masks=panels()
    t=temporal.rename(columns={"SEQN":"ID","male":"sex"}).copy()
    core=np.logical_and.reduce([t[feats].notna().any(axis=1) for name,feats in masks.items() if name!="klosa"])
    common=core & t[masks["klosa"]].notna().any(axis=1)
    for name, features in masks.items():
        yield name,"sim",t.loc[common],features
        if name!="klosa":
            yield name,"primary",t.loc[core],features
        if name!="full":
            f=external(name)
            yield name,"external",f.loc[f.reindex(columns=features).notna().any(axis=1)],features
            if name in ("nhanes3","mimic"):
                actual = [c for c in FEATS if c in f and f[c].notna().any()]
                yield name,"actual",f.loc[f[actual].notna().any(axis=1)],actual


def rsf_inputs():
    from design import OUT, develop
    d,t=develop()
    path=OUT/"cache"
    path.mkdir(exist_ok=True)
    d.to_csv(path/"dev.csv.gz",index=False)
    for name, kind, frame, feats in scenarios(t):
        f=frame.reindex(columns=["ID","age","sex","event","time_yr",*feats])
        f.to_csv(path/f"{kind}_{name}.csv.gz",index=False)


if __name__=="__main__":
    rsf_inputs()
