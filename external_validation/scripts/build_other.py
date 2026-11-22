"""Build independent, person-level external inputs from the supplied releases."""
import argparse
import hashlib
import json
from pathlib import Path

from study_paths import COMMUNITY_ROOT

import numpy as np
import pandas as pd
import pyreadstat


OUT = Path(__file__).resolve().parents[1]
BASE = COMMUNITY_ROOT
SOURCES = []


def source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(pattern, cols=None):
    matches = list(BASE.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one source for {pattern}, got {matches}")
    p = matches[0]
    SOURCES.append(p)
    return pyreadstat.read_dta(p, usecols=list(cols) if cols else None)[0]


def num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def ym(year, month):
    y, m = num(year), num(month)
    return (12*y + m).where(y.between(1900, 2030) & m.between(1, 12))


def date_ym(series):
    d = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    return ym(d.dt.year, d.dt.month)


def wave_max(values):
    return pd.concat(values, axis=1).max(axis=1)


def endpoints(out, base, last, died, death, conflict, meta):
    """Count only positive observed follow-up. Retain unknowns as missing."""
    fail = died & ((death < last) | conflict | death.isna())
    end = death.where(died, last)
    valid = (end > base) & ~fail & base.notna()
    out["event"] = died.astype(float).where(valid)
    out["time_yr"] = ((end-base)/12).where(valid)
    meta["followup_quality_before_baseline_filter"] = {
        "missing_baseline_month": int(base.isna().sum()),
        "death_reported_missing_complete_year_month": int((died & death.isna()).sum()),
        "death_conflict_or_falls_outside_alive_to_informant_interview_interval": int((died & ((death < last) | conflict)).sum()),
        "no_positive_followup_and_no_reported_death": int((~died & ~(last>base)).sum())}


def share():
    mapping = dict(cholesterol="cho_x4", hdl="hdl_x4", triglycerides="trg_x4",
                   hgb="thb_x4", crp="crp_x4", hba1c="a1c_x4")
    dbs = source("sharew6_rel1-0-0_dbs.dta", ["mergeid", *mapping.values()])
    waves = ("6","7","8","8ca","9ca","9")
    cols = [f"{c}_w{w}" for c in ("interview","int_year","int_month","deadoralive") for w in waves]
    cv = source("sharewX_rel9-0-0_gv_allwaves_cv_r.dta", ["mergeid","country","gender","age_int_w6",
                "deceased_year","deceased_month",*cols])
    assert set(dbs.mergeid) <= set(cv.mergeid)
    d = dbs.merge(cv,on="mergeid",validate="one_to_one")
    out = pd.DataFrame(dict(ID=d.mergeid,age=num(d.age_int_w6),sex=d.gender.map({1:1,2:0}),country=d.country))
    for feature, col in mapping.items():
        v = num(d[col])
        out[feature] = v.where(v>0) / (10 if feature=="crp" else 1)
    base = ym(d.int_year_w6,d.int_month_w6).where(d.interview_w6.eq(1) & d.deadoralive_w6.eq(1))
    living = [ym(d[f"int_year_w{w}"],d[f"int_month_w{w}"]).where(
        d[f"interview_w{w}"].eq(1) & d[f"deadoralive_w{w}"].eq(1)) for w in waves]
    death = ym(d.deceased_year,d.deceased_month)
    signals = [d[f"deadoralive_w{w}"].eq(2) | d[f"interview_w{w}"].eq(2) for w in waves[1:]]
    died = pd.concat(signals,axis=1).any(axis=1) | num(d.deceased_year).gt(0)
    reports=[ym(d[f"int_year_w{w}"],d[f"int_month_w{w}"]).where(d[f"interview_w{w}"].eq(2)) for w in waves[1:]]
    after_report=death.gt(pd.concat(reports,axis=1).min(axis=1))
    m = dict(cohort="SHARE",baseline="Wave 6 DBS in-home interview; collection during interview; x4 final specimen-adjusted measurements, 12 countries",
        panel_type="objective_blood_DBS_adjusted",feature_mapping=mapping,
        units="HbA1c %, tHb g/dL, lipids mg/dL, CRP mg/L divided by 10 to mg/dL. Cystatin C is a different analyte from creatinine and is excluded.",
        references=["https://www.nber.org/system/files/working_papers/w32764/w32764.pdf (table 5-7)",
                    "https://share-eric.eu/data/data-set-details/share-dbs"],
        survival="Reported death year/month in all-waves coverscreen. Censor at last dated participant main interview with confirmed alive status in w6/7/8/8ca/9ca/9. Positive elapsed months only.",
        limitations="DBS specimen and assay differ from NHANES venous tests, high missingness by assay batch and country. Country pooling and unweighted analysis do not estimate a population death rate.",
        raw_baseline_file_rows=len(d))
    endpoints(out,base,wave_max(living),died,death,after_report,m)
    return out, m, list(mapping)


def mhas():
    mapping=dict(bmi="r3mbmi",waist="r3mwaist",sbp="r3systo",dbp="r3diasto",pulse="r3pulse",
                 hba1c="reshg2_12",hgb="resh2_12",hdl="hdlc_12",cholesterol="chol_rev_12",crp="crp_12")
    bio=source("sect_l_biomarkers_2012.dta",["cunicah","np","date_12","reshg1_12","resh1_12",
                "reshg2_12","resh2_12","hdlc_12","chol_rev_12","crp_12"])
    def person(d):
        return (100*num(d.cunicah)+num(d.np)).astype("int64")
    bio["key"] = person(bio)
    h=source("H_MHAS_c2.dta",["unhhidnp","ragender","r3agey","r3iwstat","r3mbmi","r3mwaist","r3systo",
              "r3diasto","r3pulse","r4iwstat","r5iwstat","radyear","radmonth"])
    h["key"]=num(h.unhhidnp).astype("int64")
    waves=["15","18","21"]
    track=source("master_follow_up_file_2021.dta",["cunicah","np","UNHHIDNP",*[f"{field}_{w}" for w in waves
                 for field in ("fallecido","tipent","int_date")]])
    track["key"]=person(track)
    # The master string pads the person number to three digits. The older
    # harmonized numeric ID concatenates the same household and two-digit NP.
    assert np.array_equal(1000*num(track.cunicah)+num(track.np),num(track.UNHHIDNP))
    eol=source("sect_sa_sb_sc_sd_se_sh_si_2021.dta",["cunicah","np","sa8_1_21","sa8_2_21"])
    eol["key"]=person(eol)
    assert set(bio.key)<=set(h.key) and set(bio.key)<=set(track.key)
    d=bio.merge(h,on="key",validate="one_to_one").merge(track.drop(columns=["cunicah","np"]),
        on="key",validate="one_to_one").merge(eol[["key","sa8_1_21","sa8_2_21"]],on="key",how="left",validate="one_to_one")
    out=pd.DataFrame(dict(ID=d.UNHHIDNP,age=num(d.r3agey),sex=d.ragender.map({1:1,2:0})))
    for f,c in mapping.items():
        out[f]=num(d[c]).where(num(d[c]).ge(0) if f=="crp" else num(d[c]).gt(0))
    out["hba1c"]=out.hba1c.where(d.reshg1_12.eq(1))
    out["hgb"]=out.hgb.where(d.resh1_12.eq(1))
    base=date_ym(d.date_12).where(pd.to_datetime(d.date_12,errors="coerce").dt.year.eq(2012) & d.r3iwstat.eq(1))
    living=[base]+[date_ym(d[f"int_date_{w}"]).where(d[f"tipent_{w}"].isin([1,2,3,4])) for w in waves]
    death1=ym(d.radyear,d.radmonth)
    death2=ym(d.sa8_2_21,d.sa8_1_21)
    disagree=death1.notna() & death2.notna() & death1.ne(death2)
    death=death1.fillna(death2)
    died=(num(d.radyear).between(1990,2023) | num(d.sa8_2_21).between(1990,2023) |
          d.r4iwstat.isin([5,6]) | d.r5iwstat.isin([5,6]))
    for w in waves:
        died |= num(d[f"fallecido_{w}"]).isin([1,2]) | num(d[f"tipent_{w}"]).eq(5)
    reports=[date_ym(d[f"int_date_{w}"]).where(num(d[f"tipent_{w}"]).eq(5)) for w in waves]
    disagree |= death.gt(pd.concat(reports,axis=1).min(axis=1))
    m=dict(cohort="MHAS",baseline="2012 wave 3 objective biomarkers/anthropometrics subsample, one visit per person, actual dated blood/anthropometry collection month",
       panel_type="objective_blood_and_physical",feature_mapping=mapping,
       units="HbA1c %, capillary Hb g/dL, venous HDL and revised total cholesterol mg/dL, CRP mg/dL exactly as codebook and laboratory protocol specify, BMI kg/m2, measured waist cm, BP mmHg, pulse/min. MHAS median CRP is high on the published scale; independent lab harmonization not established.",
       references=["https://mhasweb.org/Resources/DOCUMENTS/2012/Codebook/Section_L_2012_Anthropometric_Measurements_and_Biomarkers.pdf",
                   "https://enasem.org/Resources/DOCUMENTS/2012/MHAS_Procedures_for_Biomarkers_2012.pdf"],
       survival="All-cause death from harmonized radyear/radmonth through 2018 or 2021 next-of-kin sa8_2_21/sa8_1_21. Compare against harmonized status, master fallecido and next-of-kin type. Last alive at positively dated interview types 1-4, including proxy for living participant. Month-scale; unverified, contradictory and zero follow-up unscored for survival.",
       limitations="2012 biomarker subsample, assay bias possible. Blood visit dates absent or 1960 sentinel retained only for scoring; main interview dates never substituted for missing draw dates.",
       raw_baseline_file_rows=len(bio))
    endpoints(out,base,wave_max(living),died,death,disagree,m)
    return out,m,list(mapping)


def klosa():
    waves=range(1,9)
    d=source("H_KLoSA_e2.dta",["pid","ragender","r1agey","r1bmi","radyear","radmonth",
                  *[f"r{w}{f}" for w in waves for f in ["iwstat","iwy","iwm"]]])
    out=pd.DataFrame(dict(ID=d.pid,age=num(d.r1agey),sex=d.ragender.map({1:1,2:0}),
                          bmi=num(d.r1bmi).where(d.r1iwstat.eq(1))))
    dates=[ym(d[f"r{w}iwy"],d[f"r{w}iwm"]).where(d[f"r{w}iwstat"].eq(1)) for w in waves]
    death=ym(d.radyear,d.radmonth)
    died=pd.concat([d[f"r{w}iwstat"].isin([5,6]) for w in waves],axis=1).any(axis=1)|num(d.radyear).gt(0)
    m=dict(cohort="KLoSA",baseline="Wave 1 (2006) interviewed respondents",panel_type="self_reported_BMI_sensitivity_only",
       feature_mapping={"bmi":"r1bmi, self-reported height and weight, kg/m²"},raw_baseline_file_rows=len(d),
       survival="Harmonized radyear/radmonth and iwstat 5/6 deaths; last participant interview month iwstat=1 in waves 1-8 (through 2020).",
       limitations="Only self-reported BMI overlaps the model, a measurement-mode sensitivity analysis. Grip strength is objectively measured but has no matching model input.",
       references=["Local H_KLoSA_e2.dta and Harmonized KLoSA E.2 2006-2020.pdf, body mass index and interview status sections"])
    endpoints(out,dates[0],wave_max(dates),died,death,pd.Series(False,index=d.index),m)
    return out,m,["bmi"]


def lasi():
    mapping={"hgb":"haemoglobin__hb2","platelets":"platelet_count2","rbc":"erythrocyte__rbc__count2",
         "rdw":"rdw__red_cell_distribut2","wbc":"total_leucocytes__wbc2","neut_pct":"neutrophils2",
         "eos_pct":"eosinophils2","lymph_pct":"lymphocytes2","mono_pct":"monocytes2","mch":"mch__mean_corpuscular_h2",
         "mcv":"mcv__mean_corpuscular_v2","hba1c":"hba1c__glycated_haemogl2","cholesterol":"cholesterol__total2",
         "hdl":"hdl_cholesterol2","triglycerides":"triglycerides_level2","bilirubin":"bilirubin_total2",
         "total_protein":"total_protein2","albumin":"albumin2","globulin":"globulin2","alt":"sgpt__alt2",
         "ast":"sgot__ast2","alp":"alkaline_phosphatase2","ggt":"gamma_gt__ggtp2","bun":"bun_blood_urea_nitrogen2",
         "creatinine":"creatinine2","uric_acid":"uric_acid2","calcium":"calcium2",
         "bmi":"r1mbmi","sbp":"r1systo","dbp":"r1diasto","pulse":"r1pulse"}
    physical=["r1mbmi","r1systo","r1diasto","r1pulse"]
    bio=source("dad_vbs_w1a.dta",["prim_key",*[n for n in mapping.values() if n not in physical]])
    h=source("h_dad_w1a3.dta",["prim_key","r1agey","ragender","r1ht_flag","r1wt_flag",*physical])
    assert set(bio.prim_key)<=set(h.prim_key)
    d=bio.merge(h,on="prim_key",validate="one_to_one")
    out=pd.DataFrame(dict(ID=d.prim_key,age=num(d.r1agey),sex=d.ragender.map({1:1,2:0}),event=np.nan,time_yr=np.nan))
    for f,c in mapping.items():
        v=num(d[c])
        out[f]=v.where(v.ge(0)) / (1000 if f=="wbc" else 1)
    measured_here = num(d.r1ht_flag).eq(0) & num(d.r1wt_flag).eq(0)
    earlier_bmi = int((out.bmi.notna() & ~measured_here).sum())
    out["bmi"] = out.bmi.where(measured_here)
    m=dict(cohort="LASI-DAD",baseline="Wave 1 DAD visit 2017–2020, venous subsample; age and physical measures from the same DAD visit.",
       panel_type="objective_blood_and_physical_scoring_only",feature_mapping=mapping,
       bmi_source_check={"rule":"r1ht_flag=0 and r1wt_flag=0 (both measured at DAD visit), Harmonized LASI-DAD A.3 pp. 221-223","previous_visit_bmi_values_removed":earlier_bmi},
       raw_baseline_file_rows=len(bio),survival="The supplied LASI-DAD folder contains wave 1 only. National wave-1 files, 2010 pilot and studies reporting later outcomes do not supply linked dates of death or last known alive for these participants.",
       units="VBS protocol Tables 4,5,7: WBC raw cells/µL /1000 gives 10^9/L. Platelet numeric values are thousands/µL as confirmed by published Table 7 summary (232.18 mean, 33 to 1368). RBC million/µL, differential %; Hb g/dL, RDW %, MCH pg, MCV fL, HbA1c %, enzymes U/L, proteins g/dL, lipids/creatinine/BUN/uric acid/calcium/bilirubin mg/dL. BP mmHg, pulse/min, BMI kg/m².",
       limitations="No mortality performance estimate possible. Estimated average glucose is derived from HbA1c and was excluded. CRP excluded because wave 1 VBS protocol and subsequent wave-1 analysis conflict on units. Additional analytes without verified release-specific units excluded.",
       references=["https://lasi-dad.org/codebooks/LASI-DAD%20VBS%20Data%20Document.pdf",
                   "https://pubmed.ncbi.nlm.nih.gov/42466419/"])
    return out,m,list(mapping)


def build(name):
    SOURCES.clear()
    out,meta,features=globals()[name]()
    out=out.loc[out.age.ge(20)&out.age.le(120)&out[features].notna().any(axis=1)].copy()
    if out.ID.duplicated().any() or out.ID.isna().any() or not (out.time_yr.isna()==out.event.isna()).all():
        raise RuntimeError("Invalid identity or endpoint")
    meta.update(n_baseline_scored=len(out),n_survival=int(out.event.notna().sum()),deaths=int(out.event.sum()),
        positive_followup_median_years=float(out.time_yr.median()) if out.event.notna().any() else None,
        feature_counts={c:int(out[c].notna().sum()) for c in features},n_missing_survival=int(out.event.isna().sum()),
        sources=[{"path":str(p.relative_to(BASE)), "sha256":source_digest(p)} for p in SOURCES])
    if (out.time_yr.dropna()<=0).any() or not set(out.event.dropna()).issubset({0,1}):
        raise RuntimeError("Invalid survival data")
    for folder in ["inputs","audit"]:
        (OUT/folder).mkdir(exist_ok=True,parents=True)
    out.to_csv(OUT/"inputs"/f"{name}.csv.gz",index=False)
    (OUT/"audit"/f"{name}.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    print(name,len(out),meta["n_survival"],meta["deaths"],len(features))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("cohorts",nargs="*",default=["share","mhas","klosa","lasi"])
    for name in parser.parse_args().cohorts:
        if name not in {"share","mhas","klosa","lasi"}:
            raise ValueError(name)
        build(name)
