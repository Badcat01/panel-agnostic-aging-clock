"""Rebuild one baseline per person, with independent mortality follow-up, from local source releases."""
import hashlib
import json
from pathlib import Path

from study_paths import COMMUNITY_ROOT, community_dir

import numpy as np
import pandas as pd
import pyreadstat

OUT = Path(__file__).resolve().parents[1]
DATA = COMMUNITY_ROOT
HRS = community_dir("HRS")
ELSA = community_dir("ELSA")


def jdump(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metadata(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2**20), b""):
            h.update(chunk)
    return {"file": str(path.relative_to(DATA)), "sha256": h.hexdigest()}


def month_start(year, month):
    return pd.to_datetime(pd.DataFrame({"year": year, "month": month.where(month.between(1, 12)), "day": 1}),
                          errors="coerce")


def save(name, rows, features, audit):
    if rows.ID.isna().any() or rows.ID.duplicated().any():
        raise ValueError(f"{name} duplicated or missing participant IDs")
    assert rows.event.isin([0, 1]).all() and rows.age.ge(20).all() and rows.time_yr.gt(0).all()
    assert rows[features].notna().any(axis=1).all()
    assert (rows.time_lower_yr <= rows.time_yr + 1.e-12).all()
    assert (rows.time_upper_yr >= rows.time_yr - 1.e-12).all()
    audit["included"] = len(rows)
    audit["deaths"] = int(rows.event.sum())
    audit["censored"] = int((rows.event == 0).sum())
    audit["followup_median_years"] = float(rows.time_yr.median())
    audit["date_quality_counts"] = rows.date_quality.value_counts().to_dict()
    audit["observed_feature_counts"] = {s:int(rows[s].notna().sum()) for s in features}
    audit["observed_features_per_person_median"] = float(rows[features].notna().sum(axis=1).median())
    cols = ["ID", "age", "sex", "time_yr", "event", *features, "time_lower_yr", "time_upper_yr", "date_quality"]
    rows[cols].to_csv(OUT / "inputs" / (name + ".csv.gz"), index=False, compression=dict(method="gzip",mtime=0))
    jdump(OUT / "audit" / (name + ".json"), audit)
    print(name, len(rows), audit["deaths"], audit["date_quality_counts"],flush=True)


def build_hrs():
    vbs_path = next(HRS.rglob("hrs2016vbs.sav"))
    track_path = next(HRS.rglob("trk2020tr_r.dta"))
    assays, ameta = pyreadstat.read_sav(vbs_path)
    tcols = ("HHID PN PAGE BIRTHMO BIRTHYR PIWYEAR PIWMONTH GENDER KNOWNDECEASEDYR KNOWNDECEASEDMO "
             "KNOWNDECEASEDSOURCE LASTALIVEYR LASTALIVEMO LASTALIVESOURCE RALIVE QALIVE").split()
    tracker, _ = pyreadstat.read_dta(track_path, usecols=tcols)
    d = assays.merge(tracker, on=["HHID","PN"], how="left", validate="one_to_one", indicator=True)
    out = pd.DataFrame({"ID":d.HHID.str.strip()+"-"+d.PN.str.strip(), "sex":d.GENDER.map({1:1,2:0})})
    index_month = month_start(d.PIWYEAR,d.PIWMONTH)
    delay = pd.to_timedelta(d.PVBS_N_DAYS,unit="D")
    day = pd.Timedelta(days=1)
    draw = index_month + 14*day + delay
    born = month_start(d.BIRTHYR,d.BIRTHMO)
    out["age"] = (draw.dt.year-born.dt.year - (draw.dt.month < born.dt.month).astype(int)).fillna(d.PAGE)
    fields = {
        "albumin":"PALB","alp":"PALKP2","ast":"PAST","alt":"PALT","bilirubin":"PBILT","total_protein":"PTP",
        "creatinine":"PCR","bun":"PBUN","potassium":"PK","sodium":"PNA","chloride":"PCL","bicarbonate":"PCO2",
        "calcium":"PCA","glucose":"PGLUFF","triglycerides":"PTGF","cholesterol":"PCHOL","hdl":"PHDLD",
        "wbc":"PWBC","lymph_pct":"PLYMP","mono_pct":"PMONO","neut_pct":"PNEUT","eos_pct":"PEOS","crp":"PCRP",
        "rbc":"PRBC","hgb":"PHGB","mcv":"PMCV","mch":"PMCH","rdw":"PRDW","platelets":"PPLT","mpv":"PMPV",
    }
    for col, source in fields.items():
        out[col] = d[source].where(d[source] >= 0)
    out["crp"] /= 10
    out["globulin"] = (out.total_protein-out.albumin).where(out.total_protein>out.albumin)
    features = list(fields) + ["globulin"]
    known_dead = d.KNOWNDECEASEDYR.between(2016, 2021)
    unlocated_death = ~known_dead & (d.RALIVE.isin([5,6]) | d.QALIVE.isin([5,6]))
    alive = month_start(d.LASTALIVEYR,d.LASTALIVEMO)
    death_month = month_start(d.KNOWNDECEASEDYR,d.KNOWNDECEASEDMO)
    death_first = death_month.fillna(month_start(d.KNOWNDECEASEDYR, pd.Series(1,index=d.index)))
    death_end = (death_month + pd.offsets.MonthEnd(0)).fillna(
        month_start(d.KNOWNDECEASEDYR, pd.Series(12,index=d.index)) + pd.offsets.MonthEnd(0))
    # Last-known-alive can bound a year-only death estimate. Dead participants
    # never cross to the living/censored group when a death date is unusable.
    death_low = pd.concat([death_first,draw,alive],axis=1).max(axis=1)
    death_mid = death_low + (death_end-death_low)/2
    stop = (alive+14*day).where(~known_dead,death_mid)
    out["time_yr"] = (stop-draw)/pd.Timedelta(days=365.25)
    out["time_lower_yr"] = out.time_yr.where(~known_dead, (death_low-draw)/pd.Timedelta(days=365.25))
    out["time_upper_yr"] = out.time_yr.where(~known_dead, (death_end-draw)/pd.Timedelta(days=365.25))
    out["event"] = known_dead.astype(int)
    source_label = d.KNOWNDECEASEDSOURCE.map({1:"exit_report",2:"spouse_report",3:"field_report",4:"tracker_imputed"})
    out["date_quality"] = source_label.where(known_dead,"last_known_alive_month")
    year_only = known_dead & death_month.isna()
    out.loc[year_only,"date_quality"] = out.loc[year_only,"date_quality"]+"_year_only"
    candidate = draw.notna() & out.age.ge(20) & out[features].notna().any(axis=1)
    conflict = known_dead & (death_end < death_low)
    included = candidate & ~conflict & ~unlocated_death & out.time_yr.gt(0)
    audit = {
        "baseline":"Single 2016 HRS VBS serum/whole-blood draw (clinical results at that draw only).",
        "sources":[metadata(vbs_path),metadata(track_path)], "vbs_release":list(map(str,assays.VERSION.drop_duplicates())),
        "official_sources":[
            "https://hrs.isr.umich.edu/sites/default/files/biblio/HRS2016VBSDD_1.pdf",
            "https://hrsdata.isr.umich.edu/sites/default/files/documentation/data-descriptions/1704996077/trk20_Final_V1.pdf"],
        "ID":"HHID + PN, one-to-one, leading zeros preserved",
        "age":"Completed years at approximated draw month minus birth month, ignoring birthday day. Fallback PAGE if birth month unknown.",
        "time_origin":"Midmonth (15th) of PIWYEAR/PIWMONTH + exact VBS lag PVBS_N_DAYS. Interview day unavailable.",
        "endpoint":"Known all-cause death from 2020 Tracker (through fieldwork 2021), KNOWNDECEASEDYR/MO. Midpoint of death month (or permitted year), constrained by draw and last verified alive. With no reported death, censor at the 15th day of last confirmed alive month, LASTALIVEYR/MO. Explicit detected-death flag without date cannot be censored.",
        "limitations":["About-month precision of index and follow-up dates; tracker includes field reports and source-imputed dates (see date_quality). Source 3/4 month bounds describe recorded precision, not actual uncertainty of those imputed dates.",
                       "US VBS venous lab methods and fasting status differ from NHANES; no cross-assay recalibration. Physical measurements from a different-day core interview are omitted. No DNAm measurements entered."],
        "vbs_baseline_people":len(assays),"unmatched_tracker":int(d._merge.eq("left_only").sum()),
        "eligible_baseline":int(candidate.sum()),"excluded_known_death_without_year":int((candidate & unlocated_death).sum()),
        "excluded_death_date_conflict":int((candidate & conflict).sum()),
        "excluded_no_positive_followup_or_missing_end":int((candidate & ~conflict & ~unlocated_death & ~out.time_yr.gt(0)).sum()),
        "age_uses_interview_fallback":int((included & born.isna()).sum()),
        "feature_columns_and_labels":{f:{v:ameta.column_names_to_labels[v]} for f,v in fields.items()},
        "conversions":{"crp":"PCRP mg/L / 10 to mg/dL","globulin":"PTP minus PALB in g/dL (calculated serum globulin)"},
        "fasting_status_included":d.loc[included,"PFASTYN"].map({1:"yes",5:"no"}).fillna("unknown").value_counts().to_dict(),
    }
    save("hrs",out.loc[included].copy(),features,audit)


def build_elsa():
    nur_path = ELSA/"Raw_data"/"wave2"/"wave_2_nurse_data_v2.dta"
    harm_path = next(ELSA.rglob("h_elsa_g3.dta"))
    idx_path = next(ELSA.rglob("index_file_wave_0-wave_5_v2.dta"))
    nurses, nm = pyreadstat.read_dta(nur_path)
    cols = ["idauniq","radyear"]+[f"r{w}iw{v}" for w in range(2,10) for v in ("stat","indy","indm")]
    harm,_ = pyreadstat.read_dta(harm_path,usecols=cols)
    tracking,_ = pyreadstat.read_dta(idx_path,usecols=["idauniq","mortwave"])
    # Wave end is the final observed individual interview month in the full
    # release (W6 is the final wave with new death statuses in h_elsa_g3).
    ends = {}
    for w in range(2,7):
        contact = month_start(harm[f"r{w}iwindy"],harm[f"r{w}iwindm"]).where(harm[f"r{w}iwstat"].eq(1))
        ends[w] = contact.max()+pd.offsets.MonthEnd(0)
    cutoff = ends[6]
    d = nurses.merge(harm,on="idauniq",how="left",validate="one_to_one")
    d = d.merge(tracking,on="idauniq",how="left",validate="one_to_one")
    base = month_start(d.visyear,d.vismon)+pd.Timedelta(days=14)
    out = pd.DataFrame({"ID":d.idauniq.astype(str),"age":d.confage.replace({99:90}), "sex":d.sex.map({1:1,2:0})})
    sourcecols = {"cholesterol":("chol",1/0.0259), "hdl":("hdl",1/0.0259), "triglycerides":("trig",1/0.0113),
                  "glucose":("fglu",1/0.0555), "hba1c":("hba1c",1), "hgb":("hgb",1), "crp":("hscrp",0.1),
                  "bmi":("bmival",1), "waist":("wstval",1), "sbp":("sysval",1), "dbp":("diaval",1)}
    for col,(source,multiplier) in sourcecols.items():
        out[col] = d[source].where(d[source]>0)*multiplier
    pulse = d[["pulse1","pulse2","pulse3"]]
    out["pulse"] = pulse.where(pulse>0).mean(axis=1).where(out.sbp.notna() & out.dbp.notna())
    features = list(sourcecols)+["pulse"]
    blood = ["cholesterol","hdl","triglycerides","glucose","hba1c","hgb","crp"]
    # Use own, dated interviews only. A later respondent interview can support
    # known survival up to the administrative end, even if W6 was skipped.
    alive_dates = pd.DataFrame({
        w:(month_start(d[f"r{w}iwindy"],d[f"r{w}iwindm"])+pd.Timedelta(days=14)).where(d[f"r{w}iwstat"].eq(1))
        for w in range(2,10)
    })
    last = alive_dates.max(axis=1).clip(upper=cutoff)
    # mortwave 31/32/33, etc. mean deaths found between issues of waves
    # or during the incoming wave. Its last fieldwork date gives a conservative
    # upper limit; no wave-assigned calendar month of death is claimed.
    deathw = (d.mortwave//10).where(d.mortwave.between(31,63))
    for w in range(3,7):
        newly_dead = d[f"r{w}iwstat"].isin([5,6])
        deathw = deathw.mask(newly_dead & (deathw.isna() | (deathw>w)),w)
    high = pd.to_datetime(deathw.map(ends))
    year = d.radyear.where(d.radyear.between(2004,cutoff.year))
    ystart = month_start(year,pd.Series(1,index=d.index))
    yend = month_start(year,pd.Series(12,index=d.index))+pd.offsets.MonthEnd(0)
    dead = deathw.notna() | year.notna()
    low = pd.concat([base,last,ystart],axis=1).max(axis=1)
    high = pd.concat([high,yend],axis=1).min(axis=1).clip(upper=cutoff)
    mid = low + (high-low)/2
    stop = last.where(~dead,mid)
    out["event"] = dead.astype(int)
    yr = pd.Timedelta(days=365.25)
    out["time_yr"] = (stop-base)/yr
    out["time_lower_yr"] = out.time_yr.where(~dead,(low-base)/yr)
    out["time_upper_yr"] = out.time_yr.where(~dead,(high-base)/yr)
    out["date_quality"] = np.select([dead & year.notna(),dead],["death_report_year_interval","death_wave_interval"],
                                    default="last_interview_or_admin_censor")
    eligible = base.notna() & out.age.ge(20) & out[blood].notna().any(axis=1)
    conflict = dead & ((high<=low) | high.isna())
    included = eligible & ~conflict & out.time_yr.gt(0)
    audit = {
        "baseline":"ELSA Wave 2 nurse (first blood collection wave), one person/one visit; require at least one frozen-model laboratory value.",
        "sources":[metadata(nur_path),metadata(harm_path),metadata(idx_path)],
        "official_sources":["https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_index_file_user_guide_wave_0-wave_5_v5.pdf",
                            "https://www.elsa-project.ac.uk/data-and-documentation"],
        "time_origin":"Day 15 of Wave 2 nurse visit month (visyear/vismon).",
        "endpoint":"All-cause death independently recorded by mortwave (31–63), or harmonic interview status 5/6 during Waves 3–6, or harmonized death year. Wave 6 end is the administrative cut-off. Broad event intervals are bounded below by the latest dated participant interview and nurse baseline and above by the end of first wave reporting death. Documented death years additionally constrain jan-dec bounds. Excluded irreconcilable date sequences. Primary time is each permitted death interval midpoint. Survivors censored at last individual interview or administrative end, whichever earlier. Unknown status alone is not a death and provides no censor date.",
        "fieldwork_upper_dates":{str(k):str(v.date()) for k,v in ends.items()},
        "age":"confage at nurse visit; 99 is the public file label for age 90+, recoded to its lower limit 90.",
        "age_topcoded_90plus":int((included & d.confage.eq(99)).sum()),
        "limitations":["Surveillance based on wave status and interval-censored death reports. Many dates are imprecise. Evaluate interval estimates and fixed-horizon sensitivity; statewide registry mortality not established.",
                       "No dates after W6 were treated as new deaths; later individual interviews only confirm survival through the cut-off.",
                       "Baseline lab-participants only; up to 12 markers; no hepatic or renal panel. Fasting glucose available only in fasting participants. ELSA-specific assay effects are uncalibrated."],
        "nurse_baseline_people":len(nurses),"eligible_baseline_with_lab":int(eligible.sum()),
        "excluded_no_lab_or_age":int((~eligible).sum()),"excluded_date_conflict":int((eligible&conflict).sum()),
        "excluded_no_positive_followup_or_missing_end":int((eligible&~conflict&~out.time_yr.gt(0)).sum()),
        "event_interval_width_median_years":float((out.time_upper_yr-out.time_lower_yr)[included & dead].median()),
        "feature_columns_units":{"source_variable_label_and_multiplier":{f:{"column":v,"label":nm.column_names_to_labels[v],"multiplier_to_NHANES_unit":m} for f,(v,m) in sourcecols.items()},
                                 "pulse":"Mean of available positive pulse1,2,3 (bpm) for a nurse visit with valid SBP and DBP; pulval denotes pulse pressure and is excluded."},
    }
    save("elsa",out.loc[included].copy(),features,audit)


if __name__=="__main__":
    for folder in ("inputs","audit"):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    build_hrs()
    build_elsa()
