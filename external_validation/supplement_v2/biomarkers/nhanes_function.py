"""NHANES cross-sectional function, cognition and bilateral handgrip in the temporal holdout."""
import numpy as np
import pandas as pd
from functools import lru_cache
import statsmodels.api as sm
from scipy.stats import t as student
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score,roc_auc_score
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from organ_references import read_xpt
from score_holdout import HERE, PROJECT, TRAIN_RUN, SEED, load_json, sha, write_json


def raw_demo(year,i):
    n="DEMO"+(["","_B","_C","_D","_E","_F","_G","_H","_I","_J"][i])
    d=pd.read_sas(PROJECT/"data/raw/nhanes"/(n+".xpt"),format="xport").set_index("SEQN",verify_integrity=True)
    return pd.DataFrame({"education":d.DMDEDUC2.where(d.DMDEDUC2.isin(range(1,6))),"PIR":d.INDFMPIR})


def load():
    man={r["file"][:-4]:r["sha256"] for r in load_json(HERE/"sources/extra_manifest.json")}
    res=TRAIN_RUN/"temporal_predictions.csv.gz"
    assert sha(res)==load_json(TRAIN_RUN/"temporal_metrics.json")["predictions_sha256"]
    pred=pd.read_csv(res).set_index("SEQN",verify_integrity=True)
    kdm_path=HERE.parent/"models/fits/M7_bioage/sim_full.csv.gz"
    kdm=pd.read_csv(kdm_path).rename(columns={"ID":"SEQN"}).set_index("SEQN",verify_integrity=True)
    if not set(kdm.index).issubset(pred.index):
        raise ValueError("KDM contains IDs outside the temporal roster")
    pred=pred.join(kdm[["M7_KDM"]])
    meta=pd.read_csv(PROJECT/"data/processed/nhanes_9918.csv.gz").set_index("SEQN",verify_integrity=True)
    out=[]
    for i,(y,s) in enumerate(zip(range(2009,2019,2),"FGHIJ"),5):
        d=pred[pred.cycle.eq(y)].join(meta[["wt_mec","strata","psu"]]).join(raw_demo(y,i))
        q=read_xpt("PFQ_"+s,man)
        adl=q[["PFQ061"+k for k in "HJKL"]]
        valid=adl.isin([1,2,3,4]).all(axis=1)
        d=d.join(adl.gt(1).any(axis=1).where(valid).rename("difficulty_ADL"))
        if s in "GH":
            c=read_xpt("CFQ_"+s,man)
            c["CERAD_delayed_words"]=c.CFDCSR
            c["CERAD_3_trials_words"]=c[["CFDCST1","CFDCST2","CFDCST3"]].sum(axis=1,min_count=3)
            c["animal_fluency"]=c.CFDAST.where(c.CFDAPP.eq(1))
            c["digit_symbol_correct"]=c.CFDDS.where(c.CFDDPP.eq(1))
            g=read_xpt("MGX_"+s,man)
            d=d.join(c[["CERAD_delayed_words","CERAD_3_trials_words","animal_fluency","digit_symbol_correct"]])
            d=d.join(g.MGDCGSZ.where(g.MGDEXSTS.eq(1)).rename("bilateral_grip_kg"))
        out.append(d)
    return pd.concat(out,verify_integrity=True)


def design(d):
    # Trim pathologic unknown education codes upstream. Flexible age plus sex and education.
    a=(d.age.to_numpy()-60)/10
    return np.column_stack([np.ones(len(d)),a,a**2,d.male,pd.get_dummies(d.education,drop_first=True),
                            pd.get_dummies(d.cycle,drop_first=True)]).astype(float)


@lru_cache(maxsize=1)
def masked_design():
    pairs=[]
    for p in (PROJECT/"data/raw/nhanes").glob("DEMO*.xpt"):
        d=pd.read_sas(p,format="xport")
        pairs.append(d[["SDMVSTRA","SDMVPSU"]].dropna().drop_duplicates())
    return pd.MultiIndex.from_frame(pd.concat(pairs).drop_duplicates())


def survey_se(scores,bread,strata,psu):
    total=pd.DataFrame(scores).groupby([strata.to_numpy(),psu.to_numpy()]).sum()
    domain=masked_design()
    # A PSU with no complete observations from a subdomain contributes zero to its score total.
    domain=domain[domain.get_level_values(0).isin(strata.unique())]
    total=total.reindex(domain,fill_value=0)
    nh=total.groupby(level=0).transform("size")
    if nh.min()<2: raise ValueError("Analytic strata include a singleton PSU")
    centered=(total-total.groupby(level=0).transform("mean")).multiply(np.sqrt(nh/(nh-1)),axis=0).to_numpy()
    return np.sqrt(np.diag(bread@centered.T@centered@bread)),len(total)-total.index.get_level_values(0).nunique()


def examine(df,target,predictor,sd):
    binary=target=="difficulty_ADL"
    data=df.dropna(subset=[target,predictor,"education","wt_mec","psu","strata"])
    if data.empty: return
    if binary or target!="bilateral_grip_kg": data=data[data.age>=60]
    else: data=data[data.age>=20]
    if len(data)<100: return
    x=design(data);v=data[predictor].to_numpy();arr=np.column_stack([x,(v-np.mean(v))/sd])
    y=data[target].to_numpy(dtype=float);w=data.wt_mec.to_numpy()/data.wt_mec.mean()
    if binary:
        fit=sm.GLM(y,arr,family=sm.families.Binomial(),freq_weights=w).fit()
        mu=fit.fittedvalues;beta=fit.params[-1]
        h=w*mu*(1-mu)
    else:
        fit=sm.WLS(y,arr,weights=w).fit();mu=fit.fittedvalues;beta=fit.params[-1];h=w
    bread=np.linalg.inv(arr.T@(h[:,None]*arr))
    se,deg=survey_se(arr*(w*(y-mu))[:,None],bread,data.strata,data.psu)
    s=se[-1];crit=student.ppf(.975,deg)
    convert=np.exp if binary else lambda a:a
    # Paired subject-level folds use the same baseline, score and complete observations.
    split=StratifiedKFold(5,shuffle=True,random_state=SEED) if binary else KFold(5,shuffle=True,random_state=SEED)
    preds=np.empty((len(data),2))
    for tr,va in split.split(x,y):
        for j,mat in enumerate([x,arr]):
            model=LogisticRegression(penalty=None,max_iter=500) if binary else LinearRegression()
            model.fit(mat[tr,1:],y[tr],sample_weight=w[tr])
            preds[va,j]=model.predict_proba(mat[va,1:])[:,1] if binary else model.predict(mat[va,1:])
    score=roc_auc_score if binary else r2_score
    perf=[score(y,p,sample_weight=w) for p in preds.T]
    return dict(endpoint=target,predictor=predictor,n=len(data),events=int(y.sum()) if binary else np.nan,
                years=";".join(map(str,sorted(data.cycle.unique()))),estimate=convert(beta),lower=convert(beta-crit*s),
                upper=convert(beta+crit*s),p=2*student.sf(abs(beta/s),deg),surveydf=deg,
                scale="OR" if binary else "original endpoint units",predictor_unit=sd,metric="AUC" if binary else "R_squared",
                base_oof=perf[0],augmented_oof=perf[1],increment=perf[1]-perf[0])


def main():
    d=load(); targets=["difficulty_ADL","CERAD_3_trials_words","CERAD_delayed_words","animal_fluency","digit_symbol_correct","bilateral_grip_kg"]
    sd=load_json(PROJECT/"external_validation/nhanes3_organ/organ_score_reference.json")["panels"]["full"]["accel_sd_years"]
    rows=[]
    for t in targets:
        for p in ["full_accel","ridge_cox_lp","phenoage","M7_KDM"]:
            scale=sd if p=="full_accel" else d[p].std()
            res=examine(d,t,p,scale)
            if res:rows.append(res)
    tab=pd.DataFrame(rows);tab["q"]=tab.groupby("predictor").p.transform(lambda p:multipletests(p,method="fdr_bh")[1])
    tab.to_csv(HERE/"nhanes_cross_sectional_function.csv",index=False)
    comparisons=[]
    for t in targets:
        available=[p for p in ["full_accel","ridge_cox_lp","phenoage","M7_KDM"] if d.loc[d[t].notna(),p].notna().any()]
        paired=d.dropna(subset=[t,*available])
        for p in available:
            result=examine(paired,t,p,sd if p=="full_accel" else d[p].std())
            if result: comparisons.append(result)
    pd.DataFrame(comparisons).to_csv(HERE/"nhanes_function_same_people.csv",index=False)
    print(tab[tab.predictor.eq("full_accel")][["endpoint","n","estimate","lower","upper","increment"]].to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(limits=2):main()

