"""External mortality with survey weights and additional baseline case-mix strata."""
import numpy as np
import pandas as pd
import pyreadstat
from lifelines import KaplanMeierFitter
from threadpoolctl import threadpool_limits
from shared import EXT,HERE,REFERENCE,read,formal
from survival_standard import load
from validate_external import weighted_c
from calibration import calibrate
from build_nhanes3 import load_layout,read_fwf

from study_paths import COMMUNITY_ROOT

FILES = COMMUNITY_ROOT
def stata(pattern,cols):
    p=next(FILES.rglob(pattern));return pyreadstat.read_dta(p,usecols=cols)[0]

def indicators(co,d):
    design=pd.DataFrame(index=d.index)
    info=pd.DataFrame(index=d.index)
    if co=="temporal":
        frame=pd.read_csv(EXT.parent/"data/processed/nhanes_9918.csv.gz").set_index("SEQN").loc[d.index]
        design=frame[["wt_mec","strata","psu"]].rename(columns={"wt_mec":"wt"})
        design["wt"]/=5
        baseline=[]
        for p in sorted((EXT.parent/"data/raw/nhanes").glob("DEMO_*.xpt")):
            x=pd.read_sas(p,format="xport")
            if {"SEQN","DMDEDUC2","INDFMPIR"}<=set(x):
                baseline.append(x[["SEQN","DMDEDUC2","INDFMPIR"]])
        extra=pd.concat(baseline).set_index("SEQN").reindex(d.index)
        info["education"]=extra.DMDEDUC2.where(extra.DMDEDUC2.between(1,5))
        info["PIR"]=pd.cut(extra.INDFMPIR,[-.01,1,2,5],labels=["0_1","1_2","2_5"])
    elif co=="nhanes3":
        raw,_=formal(co)
        design=raw.loc[d.index,["WTPFEX6","SDPSTRA6","SDPPSU6"]].set_axis(["wt","strata","psu"],axis=1)
        layout=load_layout("adult.sas",["SEQN","DMARETHN","DMPPIR","HFA8R"])
        # The source adult record length is specified by the source file rather than inferred data values.
        adult,_=read_fwf("adult.dat",layout,{3345,3346},20050,missing_codes=False)
        adult.index=adult.index.astype(str);adult=adult.reindex(d.index)
        info["race"]=adult.DMARETHN.where(adult.DMARETHN.between(1,4))
        edu=adult.HFA8R.where(adult.HFA8R.between(0,17))
        info["education"]=pd.cut(edu,[-1,11,12,17],labels=["lt12","12","13plus"])
        # DMPPIR contains its decimal point in the released ASCII values.
        pir=adult.DMPPIR
        info["PIR"]=pd.cut(pir.where(pir.between(0,20)),[-.01,1,2,20],labels=["0_1","1_2","gt2"])
    elif co=="hrs":
        h=stata("trk2020tr_r.dta",["HHID","PN","PVBSWGTR","SECU","STRATUM","RACE","HISPANIC"])
        h=h.set_index(h.HHID.str.strip()+"-"+h.PN.str.strip()).reindex(d.index)
        design=h[["PVBSWGTR","STRATUM","SECU"]].set_axis(["wt","strata","psu"],axis=1)
        info["race"]=h.RACE.where(h.RACE.isin([1,2,3]))
    elif co=="elsa":
        nurse=stata("wave_2_nurse_data_v2.dta",["idauniq","w2wtbld"])
        core=stata("wave_2_core_data_v4.dta",["idauniq","astratif","hseclst"])
        fields=nurse.merge(core,on="idauniq",validate="one_to_one").set_index("idauniq")
        fields.index=fields.index.astype(int).astype(str)
        design=fields.reindex(d.index)[["w2wtbld","astratif","hseclst"]].set_axis(["wt","strata","psu"],axis=1)
        design=design.where(design.ge(0))
    elif co=="charls":
        blood=stata("Blood_20140429.dta",["ID","bloodweight"])
        harm=stata("H_CHARLS_D_Data.dta",["ID","ID_w1","communityID"]).rename(columns={"ID":"current","ID_w1":"ID"})
        f=blood.merge(harm[harm.ID.ne("")],on="ID",validate="one_to_one").set_index("current").reindex(d.index)
        design=pd.DataFrame({"wt":f.bloodweight,"psu":f.communityID.replace("",np.nan),"strata":"unstratified_community_approximation"})
    if co in ("hrs","elsa","charls"):
        hs=pd.read_csv(HERE.parent/"healthspan"/f"{co}_baseline.csv.gz",dtype={"ID":str}).set_index("ID").reindex(d.index)
        info["education"]=hs.education
        info["comorbidities"]=pd.cut(hs.base_disease_count,[-1,0,1,np.inf],labels=["none","one","two_plus"])
    return design,info

def main():
    refr=read(REFERENCE);output=[];strata=[]
    for co in ["temporal","hrs","elsa","charls","nhanes3"]:
        d=load(co).dropna(subset=["full_lp","age_sex_lp","ridge_lp","sex","event","time_yr"])
        survey,info=indicators(co,d)
        order=survey[survey.wt.gt(0)].dropna().index
        d2=d.loc[order].sort_values("time_yr",ascending=False); w=survey.loc[d2.index]
        t=d2.time_yr.to_numpy();e=d2.event.to_numpy(dtype=int)
        columns=["full_lp","ridge_lp","age_sex_lp"]
        ranks=[np.unique(d2[c],return_inverse=True)[1]+1 for c in columns]
        weights=w.wt.to_numpy(dtype=float);weights/=weights.mean()
        cl=w[["strata","psu"]].drop_duplicates().reset_index(drop=True).reset_index(names="cluster")
        which=w.reset_index().merge(cl,on=["strata","psu"],validate="many_to_one",sort=False).cluster.to_numpy()
        blocks=list(cl.groupby("strata").cluster.apply(list)); groups=np.array([len(x) for x in blocks])
        def cstat(wt):return np.array([weighted_c(t,e,r,wt) for r in ranks])
        point=cstat(weights); boot=[]; rng=np.random.default_rng(20261121)
        for _ in range(2000):
            rep=np.ones(len(cl))
            for block in blocks:
                n=len(block)
                if n>1:rep[block]=np.bincount(rng.integers(0,n,n-1),minlength=n)*n/(n-1)
            boot.append(cstat(weights*rep[which]))
        sims=np.array(boot)
        bh=refr["baseline_hazard"]["clock"]["3"]
        lp=refr["risk_lp_slope"]*(d2.full_lp-refr["risk_lp_mean"])
        expected=np.average(-np.expm1(-bh*np.exp(lp)),weights=weights)
        actual=1-KaplanMeierFitter().fit(t,e,weights=weights).predict(3)
        for k,key in enumerate(columns):
            ci=np.nanpercentile(sims[:,k],[2.5,97.5]);delta=np.nanpercentile(sims[:,0]-sims[:,k],[2.5,97.5])
            output.append(dict(cohort=co,model=key,n=len(d2),events=e.sum(),strata=len(blocks),clusters=len(cl),singleton_strata=(groups==1).sum(),
                               survey_C=point[k],lower=ci[0],upper=ci[1],delta_C=point[0]-point[k],delta_lower=delta[0],delta_upper=delta[1],
                               predicted_3y_clock=expected,weighted_KM_3y=actual,design="weights + community cluster approximation" if co=="charls" else "design weights/PSU/strata"))
        pd.DataFrame(output).to_csv(HERE/"survey_validation.csv",index=False)
        for field in info:
            for label,idx in info.groupby(field,observed=True).groups.items():
                sub=d.loc[idx]
                if len(sub)<60 or sub.event.sum()<15:continue
                lp=refr["risk_lp_slope"]*(sub.full_lp.to_numpy()-refr["risk_lp_mean"])
                result=calibrate(sub.time_yr.to_numpy(),sub.event.to_numpy(int),lp,bh,3)
                strata.append(dict(cohort=co,variable=field,group=label,**result))
        pd.DataFrame(strata).to_csv(HERE/"social_clinical_subgroups.csv",index=False)
        print(co,len(d2),len(cl),flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=1):main()
