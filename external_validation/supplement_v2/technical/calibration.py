"""Age top-coding, fixed-horizon risk calibration, and HRS assay substitution."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from patsy import dmatrix
from shared import Ensemble, HERE, REFERENCE, TRAIN_RUN, read, sha, get_cohorts, formal, h0
from validate_external import censor_probabilities

def calibrate(t,e,lp,base,h):
    km=KaplanMeierFitter().fit(t,e)
    observed=float(1-km.predict(h))
    q=-np.expm1(-base*np.exp(lp))
    cases=(e==1)&(t<=h)
    levels,g,before=censor_probabilities(t,e)
    fixed=bool(np.all((t>=h)|(e==1)))
    controls=(~cases)&((t>=h) if fixed else (t>h))
    before_h=g[np.searchsorted(levels,h,side="right")-1] if np.any(levels<=h) else 1.
    w=np.zeros(len(t)); w[cases]=1/before[cases]
    if fixed: w[controls]=1.
    elif before_h>0: w[controls]=1/before_h
    row=dict(n=len(t),events_to_h=int(cases.sum()),observed_KM=observed,predicted=q.mean(),
             E_over_O=q.mean()/observed if observed else np.nan,
             harrell_c=concordance_index(t,-lp,e) if min(e.sum(), len(t)-e.sum()) else np.nan)
    sel=w>0
    if min(cases.sum(),controls.sum())<10 or ((not fixed) and before_h<.05):
        return row
    eta=np.log(base)+lp
    family=sm.families.Binomial(link=sm.families.links.CLogLog())
    def fit(design,offset=None):
        return sm.GLM(cases[sel],design[sel],freq_weights=w[sel],family=family,
                      offset=None if offset is None else offset[sel]).fit()
    slope=fit(sm.add_constant(eta))
    intercept=fit(np.ones((len(t),1)),eta)
    spline=np.asarray(dmatrix("cr(z, df=4, constraints='center')",{"z":eta}))
    curve=fit(spline)
    row.update(calibration_in_large=intercept.params[0],calibration_slope=slope.params[1],
               ICI=np.abs(curve.predict(spline)-q).mean(),ipcw_brier=(w*(cases-q)**2).mean())
    return row

def main():
    net=Ensemble()
    ref=read(REFERENCE)
    assert ref["freeze_sha256"]==sha(TRAIN_RUN/"model_freeze.json")
    dev,_,_=get_cohorts()
    ages=dev.groupby("cycle").age.agg(["min","max","count"])
    ages.to_csv(HERE/"training_age_ranges.csv")
    devlp=ref["risk_lp_slope"]*(net.score(dev)-ref["risk_lp_mean"])
    t=dev.time_yr.to_numpy(); e=dev.event.to_numpy()
    assert abs(h0(t,e,devlp,3)-ref["baseline_hazard"]["clock"]["3"])<1e-7
    h1=h0(t,e,devlp,1)
    specs=ref["age_sex"]
    demog=lambda age,sex:(age-specs["center"][0])*specs["coef"][0]+(sex-specs["center"][1])*specs["coef"][1]
    hd1=h0(t,e,demog(dev.age.to_numpy(),dev.male.to_numpy()),1)
    group=lambda age: np.where(age<70,"<70",np.where(age<80,"70–79",np.where(age<85,"80–84","85+")))
    rows=[]; attribution=[]
    for cohort in ["hrs","elsa","mimic"]:
        raw,pred=formal(cohort)
        sel=pred.full_lp.notna() & pred.sex.notna() & pred.event.notna() & pred.time_yr.gt(0)
        raw,pred=raw.loc[sel],pred.loc[sel]
        idx=np.linspace(0,len(pred)-1,120,dtype=int)
        np.testing.assert_allclose(net.score(raw.iloc[idx]),pred.full_lp.iloc[idx],atol=1e-4)
        horizon=1 if cohort=="mimic" else 3
        base=h1 if horizon==1 else ref["baseline_hazard"]["clock"][str(horizon)]
        hd=hd1 if horizon==1 else ref["baseline_hazard"]["age_sex"][str(horizon)]
        t=pred.time_yr.to_numpy(); e=pred.event.to_numpy(dtype=int)
        by_age=group(pred.age); bands=["all","<70","70–79","80–84","85+"]
        for cap in [None,85,80]:
            age=raw.age.copy(); scored=pred.full_lp.to_numpy(copy=True)
            if cap is not None:
                affected=age.gt(cap)
                age=age.clip(upper=cap)
                if affected.any():
                    scored[affected]=net.score(raw.loc[affected].assign(age=age[affected]))
            lp=ref["risk_lp_slope"]*(scored-ref["risk_lp_mean"])
            for model,ranking,bh in [("clock",lp,base),("age_sex",demog(age,raw.sex).to_numpy(),hd)]:
                for band in bands:
                    sel=np.ones(len(t),bool) if band=="all" else by_age==band
                    if not sel.any(): continue
                    rows.append(dict(cohort=cohort,horizon=horizon,model=model,cap=cap or "original",age_band=band,
                                     n_above_cap=int((raw.age[sel]>(cap or np.inf)).sum()),
                                     **calibrate(t[sel],e[sel],ranking[sel],bh,horizon)))
            if cap is None:
                update=np.empty(len(t))
                for tr,te in StratifiedKFold(5,shuffle=True,random_state=20261121).split(t,e):
                    update[te]=np.log(h0(t[tr],e[tr],lp[tr],horizon))+lp[te]
                rows.append(dict(cohort=cohort,horizon=horizon,model="clock_local_baseline_OOF",cap="original",
                                 age_band="all",n_above_cap=0,**calibrate(t,e,update,1.,horizon)))
        if cohort=="hrs":
            risks=lambda lp:-np.expm1(-base*np.exp(ref["risk_lp_slope"]*(lp-ref["risk_lp_mean"])))
            initial=risks(pred.full_lp.to_numpy())
            # Match on fixed 5-year age bands and sex. Pool oldest into the 80+ reference bin.
            bin_for=lambda a:np.floor(np.minimum(a,80)/5).astype(int)
            reference=dev.groupby([bin_for(dev.age),dev.male])[net.bundle["features"]].median()
            matched=reference.reindex(pd.MultiIndex.from_arrays([bin_for(raw.age),raw.sex])).set_axis(raw.index)
            for name in net.bundle["features"]:
                if name not in raw: continue
                keep=raw[name].notna() & matched[name].notna()
                if not keep.any(): continue
                changed=raw.loc[keep].copy(); changed[name]=matched.loc[keep,name]
                shift=(risks(net.score(changed))-initial[keep]).sum()/len(raw)
                attribution.append(dict(feature=name,n_replaced=int(keep.sum()),cohort_n=len(raw),
                                        original_mean=initial.mean(),replacement_mean=initial.mean()+shift,
                                        change_percentage_points=100*shift))
        pd.DataFrame(rows).to_csv(HERE/"age_calibration.csv",index=False)
        if attribution: pd.DataFrame(attribution).to_csv(HERE/"hrs_marker_substitution.csv",index=False)
        print(cohort,len(pred),"complete",flush=True)
    table=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),constrained_layout=True)
    for ax,cohort in zip(axes,["hrs","elsa","mimic"]):
        s=table[(table.cohort==cohort)&(table.model=="clock")&(table.age_band!="all")]
        n=s[s.cap=="original"]
        ax.plot(n.age_band,n.observed_KM,"ko--",label="Observed (KM)")
        for cap,color in [("original","#184b78"),(85,"#1d856b"),(80,"#c66527")]:
            z=s[s.cap==cap]
            ax.plot(z.age_band,z.predicted,"o-",color=color,label=str(cap))
        ax.set_title(cohort.upper()+(" · 1 year" if cohort=="mimic" else " · 3 years"))
        ax.set_xlabel("Age at baseline"); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Mortality probability")
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(HERE/"age_calibration.png",dpi=200)

if __name__=="__main__": main()
