"""Survey-weighted discovery, temporal replication and nested-OOF replication of NHANES exposures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from score_holdout import HERE, TRAIN_RUN, PROJECT, PANELS, load_json, sha, write_json
from organ_references import read_xpt
from nhanes_function import survey_se

SUFFIX=["","_B","_C","_D","_E","_F","_G","_H","_I","_J"]
STAGES=["temporal_discovery","temporal_replication","nested_development_replication"]
OUTCOMES=[p+"_accel" for p in ["full",*PANELS]]
OUT=HERE/"xwas"


def get_sample():
    temporal=TRAIN_RUN/"temporal_predictions.csv.gz"
    assert sha(temporal)==load_json(TRAIN_RUN/"temporal_metrics.json")["predictions_sha256"]
    a=pd.read_csv(temporal,usecols=["SEQN","cycle",*OUTCOMES])
    ids=pd.read_csv(TRAIN_RUN/"development_ids.csv.gz")
    oof=HERE/"xwas_oof"
    parts=[]
    for y in sorted(ids.cycle.unique()):
        audit=load_json(oof/f"audit_{y}.json")
        file=oof/f"held_{y}.csv.gz"
        assert sha(file)==audit["predictions_sha256"] and y not in audit["fit_cycles"] and len(audit["seeds"])==5
        fold=pd.read_csv(file)
        assert set(ids.loc[ids.cycle.eq(y),"SEQN"])==set(fold.SEQN) and fold.cycle.eq(y).all()
        parts.append(fold)
    b=pd.concat(parts)
    assert len(b)==19426 and len(a)==25614 and not set(a.SEQN)&set(b.SEQN)
    return pd.concat([a,b]).set_index("SEQN",verify_integrity=True)


def transform(values,weight,kind):
    mask=values.notna() & (weight>1e-12)
    if kind=="chemical":
        mask &= values.gt(0)
    s=values.loc[mask].astype(float)
    if len(s)<30 or s.nunique()<2:
        return
    if kind=="chemical" and (s.nunique()<10 or s.value_counts(normalize=True).iloc[0]>.5):
        return
    if kind=="chemical": s=np.log(s)
    w=weight.loc[s.index]
    mean=np.average(s,weights=w);sd=np.sqrt(np.average((s-mean)**2,weights=w))
    if sd<=0:return
    return (s-mean)/sd


def collect():
    catalog=load_json(HERE/"sources/exposure_catalog.json")
    man={r["file"][:-4]:r["sha256"] for r in load_json(HERE/"sources/extra_manifest.json")}
    dietary=pd.read_csv(HERE/"hei2015_two_days.csv.gz").set_index("SEQN",verify_integrity=True)
    sampled=get_sample()
    records={};metrics=[]
    for i,(y,suf) in enumerate(zip(range(1999,2019,2),SUFFIX)):
        source=PROJECT/"data/raw/nhanes"
        dem=pd.read_sas(source/f"DEMO{suf}.xpt",format="xport").set_index("SEQN",verify_integrity=True)
        dem=dem.join(sampled[sampled.cycle.eq(y)],how="inner")
        common=dem[["RIDAGEYR","RIAGENDR","RIDRETH1","SDMVSTRA","SDMVPSU","cycle",*OUTCOMES]].copy()
        common.columns=["age","sex","race","strata","psu","cycle",*OUTCOMES]
        early=y in [1999,2001]
        standard=dem.WTMEC4YR*2 if early else dem.WTMEC2YR
        crfile=(["LAB16","L16_B","L16_C"][i] if i<=2 else "ALB_CR"+suf)
        ur=pd.read_sas(source/(crfile+".xpt"),format="xport").set_index("SEQN",verify_integrity=True)
        ucr=np.log(ur.URXUCR.where(ur.URXUCR>1e-12))
        def add(name,family,v,w,kind):
            traits=common.join(v.rename("raw")).join(w.rename("w"))
            if family=="phthalates":
                traits=traits.join(ucr.rename("log_ucr")).dropna(subset=["log_ucr"])
            normalized=transform(traits.raw,traits.w,kind)
            metrics.append(dict(year=y,family=family,exposure=name,raw_n=int(traits.raw.notna().sum()),
                                eligible_n=0 if normalized is None else len(normalized)))
            if normalized is None:return
            traits=traits.loc[normalized.index].drop(columns="raw").assign(exposure_z=normalized)
            records.setdefault((family,name),[]).append(traits)
        add("education_level","social",dem.DMDEDUC2.where(dem.DMDEDUC2.isin(range(1,6))),standard,"social")
        add("poverty_income_ratio","social",dem.INDFMPIR.where(dem.INDFMPIR.between(0,5)),standard,"social")
        if y>=2005:
            f=read_xpt("SLQ"+suf,man)
            sleep=f["SLD012" if "SLD012" in f else "SLD010H"]
            add("weekday_sleep_hours","sleep",sleep.where(sleep.between(2,14)),standard,"survey")
            food=dietary.reindex(dem.index)
            add("two_day_HEI2015","diet",food.HEI2015,food.WTDR2D,"diet")
        if y>=2007:
            q=read_xpt("PAQ"+suf,man)
            values=[]
            for yes,days,minutes,factor in [("PAQ650","PAQ655","PAD660",2),("PAQ665","PAQ670","PAD675",1)]:
                val=(q[days]*q[minutes]*factor).where(q[yes].eq(1)&q[days].between(1,7)&q[minutes].between(10,960))
                values.append(val.mask(q[yes].eq(2),0))
            add("leisure_MVPA_log_weekly_minutes","activity",np.log1p(values[0]+values[1]),standard,"survey")
        families=catalog[str(y)].copy()
        if early: families["cotinine"]=families["metals"]
        for family,items in families.items():
            name=next((n for n,_ in items if not n.startswith("SS") and "VOCMWB" not in n),None)
            # Early PFAS stored-serum release used alternate analyte codes and no routine subsample weights.
            if name is None: continue
            d=read_xpt(name,man)
            candidate_weights=[c for c in d if c.startswith("WT") and c.endswith(("2Y","2YR","4Y","4YR"))]
            four=next((c for c in candidate_weights if "4Y" in c),None)
            two=next((c for c in candidate_weights if "2Y" in c),None)
            w=d[four]*2 if early and four else d[two] if two else standard
            if family=="cotinine": cols=["LBXCOT"]
            elif family=="metals": cols=[c for c in ["LBXBPB","LBXBCD","LBXTHG","LBXBSE","LBXBMN"] if c in d]
            elif family=="phthalates": cols=[c for c in d if c.startswith(("URXM","URXC","URXE","URXH")) and c not in ["URXUCR"]]
            elif family=="PFAS":
                if "LBXNFOA" in d and "LBXBFOA" in d: d["LBXPFOA"]=d.LBXNFOA+d.LBXBFOA
                if "LBXNFOS" in d and "LBXMFOS" in d: d["LBXPFOS"]=d.LBXNFOS+d.LBXMFOS
                cols=[c for c in d if c.startswith("LBX") and c not in ["LBXNFOA","LBXBFOA","LBXNFOS","LBXMFOS"]]
            else: cols=[c for c in d if c.startswith(("LBXV","LBX2DF","LBX4CE"))]
            for col in cols:
                if col in d:add(col,family,d[col],w,"chemical")
    return records,pd.DataFrame(metrics)


def evalue(beta,sd):
    if not np.isfinite(beta) or not np.isfinite(sd) or sd<=0:
        raise ValueError("E-value conversion needs a finite estimate and positive residual SD")
    rr=np.exp(.91*abs(beta)/sd)
    return rr+np.sqrt(rr*(rr-1))


def analyse(data,outcome):
    d=data.dropna(subset=[outcome,"sex","race","age","w","strata","psu"])
    if len(d)<150:return
    a=(d.age.to_numpy()-60)/10;weights=d.w.to_numpy()
    z=np.column_stack([np.ones(len(d)),a,a*a,d.sex,pd.get_dummies(d.race,drop_first=True),pd.get_dummies(d.cycle,drop_first=True)])
    if "log_ucr" in d:z=np.column_stack([z,d.log_ucr])
    z=np.column_stack([z,d.exposure_z]).astype(float);Y=d[outcome].to_numpy()
    B=np.linalg.inv(z.T@(weights[:,None]*z))
    beta=B@(z.T@(weights*Y))
    residual=Y-z@beta
    sd=np.sqrt(np.average(residual**2,weights=weights))
    s,df=survey_se(z*(residual*weights)[:,None],B,d.strata,d.psu)
    low,high=beta[-1]+np.array([-1,1])*student.ppf(.975,df)*s[-1]
    nearest=0 if low<=0<=high else min([low,high],key=abs)
    return dict(n=len(d),beta_accel_years=beta[-1],residual_SD_years=sd,lower=low,upper=high,p=2*student.sf(abs(beta[-1]/s[-1]),df),
                E_value_approx=evalue(beta[-1],sd),E_value_limit=evalue(nearest,sd),n_strata=d.strata.nunique(),
                df=df,cycles=";".join(map(str,sorted(d.cycle.unique()))))


def main():
    OUT.mkdir(exist_ok=True)
    records,catalog=collect()
    catalog.to_csv(OUT/"eligibility_by_cycle.csv",index=False)
    rows=[]
    for (family,name),parts in records.items():
        tab=pd.concat(parts,verify_integrity=True)
        windows=[tab.cycle.between(2009,2013),tab.cycle.between(2015,2017),tab.cycle<2009]
        if not all(w.sum()>=150 for w in windows[:2]): continue
        for stage,mask in zip(STAGES,windows):
            for outcome in OUTCOMES:
                ans=analyse(tab.loc[mask],outcome)
                if ans:rows.append(dict(family=family,exposure=name,panel=outcome,stage=stage,**ans))
    result=pd.DataFrame(rows)
    result["q_BH"]=result.groupby("stage").p.transform(lambda p:multipletests(p,method="fdr_bh")[1])
    result.to_csv(OUT/"all_stage_results.csv",index=False)
    rows=[]
    for (family,exposure,panel),d in result.groupby(["family","exposure","panel"]):
        v=d.set_index("stage")
        directions=np.sign(v.beta_accel_years)
        first=v.reindex(STAGES[:2])
        repeat=(first.q_BH.lt(.05).all() and np.sign(first.beta_accel_years).nunique()==1)
        three=len(v)==3 and v.q_BH.lt(.05).all() and directions.nunique()==1
        rows.append(dict(family=family,exposure=exposure,panel=panel,stages=len(d),two_stage=repeat,three_stage=three,
                    **{s+"_q":v.q_BH.get(s,np.nan) for s in STAGES},
                    **{s+"_beta":v.beta_accel_years.get(s,np.nan) for s in STAGES}))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/"replication.csv",index=False)
    print(summary.groupby("family").agg(tests=("exposure","size"),exposures=("exposure","nunique"),
                                      pass_two=("two_stage","sum"),pass_three=("three_stage","sum")).to_string())


if __name__=="__main__":
    with threadpool_limits(limits=2):main()

