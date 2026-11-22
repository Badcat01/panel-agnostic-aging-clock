"""Paired Harrell/Uno C, fixed-horizon decision curves and risk stratification."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.utils import concordance_index
from numba import njit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from shared import HERE,EXT,REFERENCE,TRAIN_RUN,read,formal
from calibration import calibrate
from validate_external import censor_probabilities,weighted_c

@njit(cache=True)
def uno(t,e,ranks,w,h):
    # t descending. At a tied time no pair is comparable. Reverse KM weights
    # use G(T-) and are re-estimated under every set of bootstrap counts.
    n=len(t);G=np.ones(n);g=1.;total=w.sum();i=n-1
    while i>=0:
        k=i
        while k>=0 and t[k]==t[i]: k-=1
        dn=0.;eventn=0.
        for j in range(k+1,i+1):
            G[j]=g;dn+=w[j];eventn+=w[j]*e[j]
        if total>eventn:g*=1-(dn-eventn)/(total-eventn)
        total-=dn;i=k
    tree=np.zeros(n+2);wins=0.;pairs=0.;count=0.;i=0
    while i<n:
        k=i
        while k<n and t[k]==t[i]:k+=1
        for j in range(i,k):
            if not e[j] or w[j]==0 or t[j]>h or G[j]<=0:continue
            ix=ranks[j];a=ix-1;low=0.;high=0.
            while a:
                low+=tree[a];a-=a&-a
            a=ix
            while a:
                high+=tree[a];a-=a&-a
            ww=w[j]/G[j]**2
            wins+=ww*(low+high)/2;pairs+=ww*count
        for j in range(i,k):
            a=ranks[j];count+=w[j]
            while a<len(tree):
                tree[a]+=w[j];a+=a&-a
        i=k
    return wins/pairs if pairs else np.nan

def load(co):
    if co=="temporal":
        d=pd.read_csv(TRAIN_RUN/"temporal_predictions.csv.gz").rename(columns={"male":"sex","ridge_cox_lp":"ridge_lp"})
        return d.set_index("SEQN",verify_integrity=True)
    return formal(co)[1]

def compute(co):
    ref=read(REFERENCE);dest=HERE/"survival";dest.mkdir(exist_ok=True)
    d=load(co).dropna(subset=["full_lp","age_sex_lp","ridge_lp","event","time_yr","sex"])
    d=d[d.time_yr.gt(0)]
    h=1 if co=="mimic" else 3
    if co=="mimic":
        z=pd.read_csv(HERE/"age_calibration.csv");expected=z.query("cohort=='mimic' and model=='clock' and cap=='original' and age_band=='all'").predicted.iloc[0]
        from scipy.optimize import brentq
        lp=ref["risk_lp_slope"]*(d.full_lp.to_numpy()-ref["risk_lp_mean"])
        bh=brentq(lambda val:np.mean(-np.expm1(-val*np.exp(lp)))-expected,1e-8,5)
    else:bh=ref["baseline_hazard"]["clock"][str(h)]
    cols=["full_lp","age_sex_lp","ridge_lp"]
    d=d.sort_values("time_yr",ascending=False,kind="stable")
    t=d.time_yr.to_numpy(dtype=float);e=d.event.to_numpy(dtype=int)
    ranks=[np.unique(d[c],return_inverse=True)[1]+1 for c in cols]
    def scores(w):
        return np.array([[weighted_c(t,e,r,w),uno(t,e,r,w,h)] for r in ranks])
    pt=scores(np.ones(len(d)))
    for j,c in enumerate(cols):np.testing.assert_allclose(pt[j,0],concordance_index(t,-d[c],e),atol=1e-10)
    rng=np.random.default_rng(20261121)
    boots=np.array([scores(np.bincount(rng.integers(0,len(d),len(d)),minlength=len(d)).astype(float)) for _ in range(2000)])
    rr=[]
    for j,c in enumerate(cols):
        for k,label in enumerate(["Harrell","Uno"]):
            ci=np.nanpercentile(boots[:,j,k],[2.5,97.5]); delta=np.nanpercentile(boots[:,0,k]-boots[:,j,k],[2.5,97.5])
            rr.append(dict(cohort=co,n=len(d),events=e.sum(),method=label,horizon=h if k else np.nan,model=c,C=pt[j,k],
                           lower=ci[0],upper=ci[1],clock_minus_model=pt[0,k]-pt[j,k],delta_low=delta[0],delta_high=delta[1],B=2000))
    pd.DataFrame(rr).to_csv(dest/f"{co}_C.csv",index=False)
    lp=ref["risk_lp_slope"]*(d.full_lp.to_numpy()-ref["risk_lp_mean"])
    levels,after,before=censor_probabilities(t,e)
    known_surv=(t>=h) if co=="mimic" else (t>h)
    yy=(e==1)&(t<=h);ctrl=(~yy)&known_surv
    at=np.searchsorted(levels,h,side="right")-1;g=after[at] if at>=0 else 1.
    w=np.zeros(len(t));w[yy]=1/before[yy];w[ctrl]=1. if co=="mimic" else 1/max(g,1e-10)
    p=-np.expm1(-bh*np.exp(lp))
    strata={"all":np.ones(len(d),bool),"female":d.sex.eq(0),"male":d.sex.eq(1),"age20_49":d.age<50,
            "age50_64":d.age.between(50,64),"age65_79":d.age.between(65,79),"age80plus":d.age>=80}
    if co=="temporal":
        detail=pd.read_csv(EXT.parent/"data/processed/nhanes_9918.csv.gz").set_index("SEQN").loc[d.index]
        for race in detail.race.dropna().unique():strata["race_"+str(race)]=detail.race.eq(race)
    if co=="mimic":
        src=EXT/"mimic_iv";idx=pd.read_csv(src/"formal_v3_1/landmark_index.csv.gz",dtype={"subject_id":str})
        admissions=pd.read_csv(src/"local_3_1/hosp/admissions.csv.gz",usecols=["hadm_id","admission_type","race"])
        more=idx.merge(admissions,on="hadm_id",validate="one_to_one").set_index("subject_id").loc[d.index]
        for code in more.admission_type.value_counts().index:strata["admission_"+code]=more.admission_type.eq(code)
        for code in more.race.value_counts().head(8).index:strata["race_"+code]=more.race.eq(code)
        icu=pd.read_csv(HERE/"icu_cache/sources/icustays.csv.gz",parse_dates=["intime","outtime"])
        flag=idx.merge(icu,on="hadm_id",suffixes=("","_icu"));beg=pd.to_datetime(flag.admittime);land=pd.to_datetime(flag.landmark)
        ids=flag.loc[flag.intime.lt(land)&flag.outtime.gt(beg),"subject_id"]
        strata["icu_overlap_hospital_first_day"]=pd.Series(d.index.isin(ids),index=d.index)
        strata["ward_first_day"]=~strata["icu_overlap_hospital_first_day"]
    sub=[]
    for label,keep in strata.items():
        if keep.sum()<40 or e[keep].sum()<10:continue
        s=calibrate(t[keep],e[keep],lp[keep],bh,h)
        s["age_sex_C"]=concordance_index(t[keep],-d.age_sex_lp[keep],e[keep])
        s["ridge_C"]=concordance_index(t[keep],-d.ridge_lp[keep],e[keep]);sub.append(dict(cohort=co,group=label,horizon=h,**s))
    if h>2:
        later=t>2
        if e[later].sum()>10:
            sub.append(dict(cohort=co,group="alive_observed_after_2y",n=later.sum(),harrell_c=concordance_index(t[later]-2,-lp[later],e[later])))
    pd.DataFrame(sub).to_csv(dest/f"{co}_subgroups.csv",index=False)
    dca=[]
    map_lp={"clock":p,"treat_all":np.ones(len(d)),"treat_none":np.zeros(len(d))}
    if co!="mimic":
        for model,key in [("age_sex","age_sex_lp"),("ridge","ridge_lp")]:
            map_lp[model]=-np.expm1(-ref["baseline_hazard"][model][str(h)]*np.exp(d[key]))
    if ctrl.sum()>=20 and (co=="mimic" or g>.05):
        for name,probs in map_lp.items():
            for threshold in [.01,.025,.05,.075,.1,.15,.2,.3]:
                take=probs>=threshold
                net=(w[take&yy].sum()-w[take&ctrl].sum()*threshold/(1-threshold))/len(d)
                dca.append(dict(cohort=co,horizon=h,model=name,threshold=threshold,net_benefit=net))
    pd.DataFrame(dca).to_csv(dest/f"{co}_DCA.csv",index=False)
    qs=pd.qcut(d.full_accel,5,labels=False,duplicates="drop")
    kmrows=[]
    for k in np.unique(qs):
        z=d.loc[qs==k];fit=KaplanMeierFitter().fit(z.time_yr,z.event)
        for tt in np.linspace(0,min(h*2,t.max()),61):
            kmrows.append(dict(cohort=co,quintile=int(k)+1,year=tt,S=fit.predict(tt),at_risk=(z.time_yr>=tt).sum()))
    pd.DataFrame(kmrows).to_csv(dest/f"{co}_km.csv",index=False)
    levels=pd.qcut(d.full_accel,[0,.1,.4,.6,.9,1],labels=False,duplicates="drop")
    ext=d.loc[levels.isin([2,4]),["time_yr","event","age","sex"]].copy()
    ext["top10"]=levels[levels.isin([2,4])].eq(4).astype(int)
    fit=CoxPHFitter().fit(ext,"time_yr","event")
    pd.DataFrame([dict(cohort=co,n=len(ext),contrast="upper decile vs 40–60th percentile",adjusted_HR=np.exp(fit.params_.top10),
                  equivalent_years=fit.params_.top10/ref["age_sex"]["coef"][0],
                  lower=fit.confidence_intervals_.loc["top10"].iloc[0]/ref["age_sex"]["coef"][0],
                  upper=fit.confidence_intervals_.loc["top10"].iloc[1]/ref["age_sex"]["coef"][0])]).to_csv(dest/f"{co}_equivalent.csv",index=False)
    print(co,len(d),"complete",flush=True)

def plots():
    src=HERE/"survival"
    with PdfPages(src/"KM_and_DCA.pdf") as pdf:
        for p in sorted(x for x in src.glob("*_km.csv") if x.name!="mhas_km.csv"):
            co=p.name.split("_")[0];km=pd.read_csv(p);dca=pd.read_csv(src/f"{co}_DCA.csv")
            fig,axes=plt.subplots(1,2,figsize=(8.4,3.6),constrained_layout=True)
            for k,g in km.groupby("quintile"):axes[0].step(g.year,g.S,where="post",label=f"Q{k}")
            for name,g in dca.groupby("model"):axes[1].plot(g.threshold,g.net_benefit,label=name)
            axes[0].set(xlabel="Years",ylabel="Survival");axes[1].set(xlabel="Threshold probability",ylabel="Net benefit")
            for ax in axes:ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.2)
            fig.suptitle(co);pdf.savefig(fig);plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("cohorts",nargs="+");args=ap.parse_args()
    with threadpool_limits(limits=1):
        for name in args.cohorts:
            if name=="plots":plots()
            else:compute(name)
