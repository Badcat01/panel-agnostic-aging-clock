"""Case-mix-standardized shifts, mortality associations and CRP sensitivity."""
import json
import sys
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from threadpoolctl import threadpool_limits
from shared import EXT,HERE,Ensemble,formal,get_cohorts,read

METHODS={"crp":"immunoturbidimetry","hba1c":"boronate affinity HPLC","glucose":"enzymatic colorimetry",
         "cholesterol":"enzymatic colorimetry","hdl":"enzymatic colorimetry","triglycerides":"enzymatic colorimetry",
         "bun":"enzymatic urease UV","creatinine":"compensated Jaffe","uric_acid":"UA Plus"}

def strata(d):
    return pd.MultiIndex.from_arrays([np.minimum(d.age,80)//5,d.sex],names=["ageband","male"])

def charls_check(raw,pred):
    import build_charls as bc
    markers=bc.LABS
    b,meta=pyreadstat.read_dta(bc.BLOOD)
    h=pd.read_stata(bc.HARM,columns=["ID","ID_w1"],convert_categoricals=False)
    subset=h[h.ID_w1.ne("")].merge(b,left_on="ID_w1",right_on="ID",suffixes=("","_old"),validate="one_to_one").set_index("ID")
    subset=subset.loc[raw.index]
    for name,(col,unit) in markers.items():
        np.testing.assert_allclose(raw[name],subset[col]/(10 if name=="crp" else 1),equal_nan=True)
    lab=meta.variable_value_labels.get("qc1_va003",{})
    rows=[]
    for status,idx in subset.groupby(subset.qc1_va003.fillna(-999)).groups.items():
        d=pred.loc[idx].dropna(subset=["sex","event","time_yr","ridge_lp","age_sex_lp","full_lp"])
        row=dict(code=status,label=lab.get(status,"unlabelled"),n=len(d),events=d.event.sum())
        if d.event.sum()>5:
            for mod in ["full_lp","ridge_lp","age_sex_lp"]: row[mod]=concordance_index(d.time_yr,-d[mod],d.event)
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/"charls_fasting.csv",index=False)

def main():
    net=Ensemble()
    b=net.bundle
    dictionary = pd.read_csv(EXT.parent / "docs" / "input_dictionary.csv").set_index("variable", verify_integrity=True)
    units = dictionary.unit.to_dict()
    dev,test,_=get_cohorts();dev=dev.rename(columns={"male":"sex"})
    feats=b["features"]
    convert=lambda v,f: np.log(v.clip(lower=.001)) if f in b["log_features"] else v
    bins=strata(dev)
    baselines={}; table=[]; changes=[]
    for feat in feats:
        y=convert(dev[feat],feat)
        baselines[feat]=y.groupby([bins.get_level_values(0),bins.get_level_values(1)]).agg(["mean","var","count"])
    for cohort in ["development","temporal","charls","hrs","elsa","nhanes3","share","klosa","lasi","mimic"]:
        if cohort=="development":
            raw=dev.copy();pred=None; metadata={}
        elif cohort=="temporal":
            raw=test.rename(columns={"male":"sex"}).copy();pred=None;metadata={}
        else:
            raw,pred=formal(cohort)
            metadata=read(EXT/"audit"/f"{cohort}.json") if cohort!="mimic" else {}
            if cohort=="charls":charls_check(raw,pred)
        if pred is not None and "crp" in raw and raw.crp.notna().any():
            updated=pred.full_lp.to_numpy(copy=True)
            measured=raw.crp.notna()
            updated[measured]=net.score(raw.loc[measured].drop(columns="crp"))
            dd=pred.assign(mask_crp=updated).dropna(subset=["mask_crp","full_lp","event","time_yr","sex"])
            if len(dd) and dd.event.sum():
                changes.append(dict(cohort=cohort,n=len(dd),events=int(dd.event.sum()),
                                    original_C=concordance_index(dd.time_yr,-dd.full_lp,dd.event),
                                    mask_C=concordance_index(dd.time_yr,-dd.mask_crp,dd.event)))
        for i,f in enumerate(feats):
            source=next((metadata[k].get(f) for k in ["features","feature_columns_and_labels","feature_columns_units","feature_mapping"] if k in metadata and f in metadata[k]),None)
            row=dict(cohort=cohort,feature=f,target_unit=units.get(f),raw_mapping=json.dumps(source,ensure_ascii=False),
                     present=f in raw and raw[f].notna().any(),n=0,source_metadata=str(EXT/"audit"/f"{cohort}.json") if metadata else cohort)
            if not row["present"]:table.append(row);continue
            temp=raw.loc[raw[f].notna()&raw.sex.notna(),["age","sex",f,"time_yr","event"]].copy()
            y=convert(temp[f],f); temp["marker"]=((y-b["mean"][i])/b["std"][i]).clip(-5,5)
            s=strata(temp); ref=baselines[f].reindex(s).set_axis(temp.index)
            mask=ref["count"].ge(2); dmean=y[mask].mean()-ref["mean"][mask].mean()
            mixture_var=(ref["var"][mask]+ref["mean"][mask]**2).mean()-ref["mean"][mask].mean()**2
            denom=np.sqrt((y[mask].var()+mixture_var)/2)
            row.update(n=len(y),matched_n=int(mask.sum()),median=temp[f].median(),age_sex_SMD=dmean/denom,
                       platform=METHODS.get(f,"see original cohort documentation; exact instrument unverified") if cohort=="charls" else metadata.get("panel_type","see source documentation; instrument unverified"),
                       fasting="individual qc1_va003" if cohort=="charls" else str(metadata.get("fasting_status_included","not uniformly documented")),
                       LOD_handling=("NHANES III released CRP includes 0.21 mg/dL code" if (cohort,f)==("nhanes3","crp") else "retained released values; assay-specific substitution unconfirmed"))
            temp=temp.dropna(subset=["event","time_yr","marker"])
            temp=temp[temp.time_yr>0]
            if len(temp)>100 and temp.event.sum()>=20 and temp.marker.std()>1e-7:
                temp["age_c"]=(temp.age-60)/10
                design=temp.assign(age_sq=temp.age_c**2)[["age_c","age_sq","sex","marker","time_yr","event"]]
                fit=CoxPHFitter().fit(design,"time_yr","event",batch_mode=True,fit_options={"step_size":0.55})
                row.update(n_mortality=len(temp),deaths=temp.event.sum(),HR_per_dev_SD=np.exp(fit.params_.marker),
                           ci_low=np.exp(fit.confidence_intervals_.loc["marker"].iloc[0]),ci_high=np.exp(fit.confidence_intervals_.loc["marker"].iloc[1]))
            table.append(row)
        pd.DataFrame(table).to_csv(HERE/"assay_audit.csv",index=False);pd.DataFrame(changes).to_csv(HERE/"crp_mask.csv",index=False)
        print(cohort,"complete",flush=True)
    df=pd.DataFrame(table)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(HERE/"marker_HR_forest.pdf") as pdf:
        for start in range(0,len(feats),6):
            fig,axes=plt.subplots(2,3,figsize=(11,7.5),constrained_layout=True)
            for ax,f in zip(axes.flat,feats[start:start+6]):
                z=df[df.feature.eq(f)].dropna(subset=["HR_per_dev_SD"])
                ax.errorbar(z.HR_per_dev_SD,z.cohort,xerr=[z.HR_per_dev_SD-z.ci_low,z.ci_high-z.HR_per_dev_SD],fmt=".",color="#235878")
                ax.axvline(1,color="gray",linewidth=.7);ax.set_xscale("log");ax.set_title(f);ax.set_xlabel("HR per NHANES SD")
            for ax in list(axes.flat)[len(feats[start:start+6]):]:ax.set_visible(False)
            pdf.savefig(fig);plt.close(fig)
    from assay_metadata import enrich
    enrich()

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
