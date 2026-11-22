"""Rebuild landmark measurements for a distinct index admission preceded by a finished MIMIC stay."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from study_paths import MIMIC_ARCHIVE
import json
import hashlib
import zipfile

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
SUP=OUT.parents[1]
EXT=SUP.parent
MIM=EXT/"mimic_iv"
sys.path[:0]=[str(MIM)]
import duckdb
from mimic_iv_validate import MAP, DIFF, DAY, YEAR, digest, one_per_patient
from mimic_iv_vitals import compose_features, ITEMS
from mimic_iv_run import write_json, write_csv

HOSP=MIM/"local_3_1/hosp"
ICU=SUP/"technical/icu_cache/sources"
CACHE=OUT/"cache"
CACHE.mkdir(exist_ok=True)
KEYS=["subject_id","hadm_id"]


def sqlpath(path):
    return "'"+str(path.resolve()).replace("\\","/").replace("'","''")+"'"


def roster(adm,pat):
    d=adm.sort_values(["subject_id","admittime","hadm_id"]).copy()
    # At least one real, positive-length stay made it to discharge strictly before this admission.
    validprior=d.dischtime.where(d.dischtime.ge(d.admittime)&d.hospital_expire_flag.eq(0))
    first_prior_end=validprior.groupby(d.subject_id).min()
    d=d.join(first_prior_end.rename("first_live_discharge"),on="subject_id",validate="many_to_one")
    patient=d.merge(pat,on="subject_id",validate="many_to_one")
    dk=d[d.hospital_expire_flag.eq(1)].copy();dk["death"]=dk.deathtime.fillna(dk.dischtime)
    death=dk.sort_values(["subject_id","death"]).drop_duplicates("subject_id").set_index("subject_id").death
    p=patient.join(death,on="subject_id")
    contacts=d.groupby("subject_id").agg(last_admit=("admittime","max"),last_discharge=("dischtime","max"))
    p=p.join(contacts,on="subject_id")
    p["age"]=(p.anchor_age+p.admittime.dt.year-p.anchor_year).where(p.anchor_age.lt(91),91)
    p["sex"]=p.gender.map({"M":1,"F":0});p["landmark"]=p.admittime+DAY;p["horizon"]=p.landmark+YEAR
    id_bad=d.loc[d.admittime.isna(),"subject_id"]
    index_death=p.deathtime.fillna(p.dischtime).where(p.hospital_expire_flag.eq(1))
    uncertain=p.dod.notna() & p.dod.eq(p.landmark.dt.normalize()) & p.death.isna()
    state_conflict=(p.dod-p.death.dt.normalize()).abs().gt(DAY)
    dateonly=p.dod+pd.Timedelta(hours=12)
    dateonly=dateonly.mask(p.hospital_expire_flag.eq(0)&p.dod.eq(p.dischtime.dt.normalize())&dateonly.lt(p.dischtime),p.dischtime)
    all_death=p.death.fillna(dateonly)
    bad=((p.hospital_expire_flag.eq(0)&p.deathtime.notna()) |
         (index_death.notna() & (index_death.lt(p.admittime)|index_death.gt(p.dischtime+DAY))) |
         uncertain | state_conflict | (p.hospital_expire_flag.eq(0)&all_death.lt(p.dischtime)) |
         (p.death.notna() & p.last_admit.gt(p.death)) |
         (p.death.isna() & p.dod.notna() & p.last_admit.ge(p.dod+DAY)) |
         (p.subject_id.isin(dk.loc[dk.death.isna(),"subject_id"])) |
         (all_death.notna() & all_death.le(p.landmark)) | (p.last_discharge+YEAR).lt(p.horizon))
    p["event"]=(all_death.notna()&all_death.le(p.horizon)).astype(int)
    p["time_yr"]=(all_death.where(p.event.eq(1),p.horizon)-p.landmark)/YEAR
    p["hospital_event"]=p.hospital_expire_flag.astype(int)
    p["hospital_time_yr"]=(index_death.fillna(p.dischtime)-p.landmark)/YEAR
    eligible=(~p.subject_id.isin(id_bad)&p.sex.notna()&p.age.between(20,120)&p.anchor_age.notna()&
              p.anchor_year.notna()&p.dischtime.gt(p.landmark)&p.hospital_time_yr.gt(0)&~bad)
    original=p[p.hadm_id.isin(d.drop_duplicates("subject_id").hadm_id)&eligible].sort_values("hadm_id")
    gold=one_per_patient(adm,pat,{}).sort_values("hadm_id")
    for field in ["hadm_id","subject_id","age","sex","event","hospital_event","time_yr","hospital_time_yr"]:
        np.testing.assert_allclose(original[field].astype(float),gold[field].astype(float))
    out=p[eligible&p.first_live_discharge.lt(p.admittime)][KEYS+["admittime","dischtime","landmark","age","sex","event","time_yr","hospital_event","hospital_time_yr"]]
    assert out.hadm_id.is_unique and out.time_yr.between(0,1,inclusive="right").all()
    return out


def source_audit():
    with zipfile.ZipFile(MIMIC_ARCHIVE) as z:
        txt=z.read("mimic-iv-3.1/SHA256SUMS.txt").decode()
        manifest={p.lstrip("*"):h for h,p in (v.split(maxsplit=1) for v in txt.splitlines())}
    files={}
    for name in ["admissions","patients","d_labitems","labevents","diagnoses_icd","icustays","chartevents"]:
        scope="icu" if name in ["icustays","chartevents"] else "hosp"
        p=(HOSP if name in ["admissions","patients","d_labitems","labevents"] else ICU)/(name+".csv.gz")
        assert digest(p)==manifest[f"{scope}/{p.name}"]
        files[name]=manifest[f"{scope}/{p.name}"]
    return files


def labs(db,index):
    defs=pd.DataFrame([(item,*vals) for item,vals in MAP.items()],columns=["itemid","feature","label","unit"])
    lut=pd.read_csv(HOSP/"d_labitems.csv.gz").set_index("itemid").loc[defs.itemid]
    assert np.array_equal(lut.label,defs.label) and lut.fluid.str.casefold().eq("blood").all()
    db.register("roster",index);db.register("lookup",defs)
    path=sqlpath(HOSP/"labevents.csv.gz")
    ids=",".join(map(str,DIFF))
    rows=db.execute(f"""
    with selected as (
       select r.subject_id,r.hadm_id,z.feature,z.itemid,v.charttime,v.storetime,v.labevent_id,
              case when z.itemid=50889 then v.valuenum/10 else v.valuenum end as value
       from read_csv({path},header=true,sample_size=100000,types={{'value':'VARCHAR','valuenum':'DOUBLE'}}) v
       join roster r using (hadm_id,subject_id) join lookup z on z.itemid=v.itemid
       where v.charttime>=r.admittime and v.charttime<r.landmark and v.storetime between v.charttime and r.landmark
         and (lower(trim(v.valueuom))=z.unit or (z.unit='iu/l' and lower(trim(v.valueuom))='u/l')
              or (z.unit='meq/l' and lower(trim(v.valueuom))='mmol/l'))
         and isfinite(v.valuenum) and (v.valuenum>0 or (v.valuenum=0 and z.itemid in (50889,{ids})))
         and (z.itemid not in ({ids}) or v.valuenum<=100))
    select subject_id,hadm_id,feature,value from selected
    qualify row_number() over (partition by hadm_id,feature order by charttime,storetime,labevent_id)=1
    """).df()
    return rows.pivot(index=KEYS,columns="feature",values="value").reset_index()


def vitals(db,index):
    if "chosen" in db.execute("show tables").df().name.tolist():db.unregister("chosen")
    db.register("chosen",index)
    d=pd.DataFrame([(item,*values[1:]) for item,values in ITEMS.items()],columns=["itemid","unit","lo","hi"])
    db.register("vitals_items",d)
    path=sqlpath(ICU/"chartevents.csv.gz")
    icu=sqlpath(ICU/"icustays.csv.gz")
    sql=f"""
    with stays as (
      select c.*,s.stay_id,s.intime,s.outtime from chosen c
      join read_csv({icu},header=true) s using(hadm_id,subject_id)
      where s.intime<c.landmark and s.outtime>=c.admittime and s.outtime>s.intime)
    select s.subject_id,s.hadm_id,s.stay_id,v.itemid,v.charttime,v.storetime,v.valuenum
    from read_csv({path},header=true,types={{'value':'VARCHAR','valuenum':'DOUBLE'}}) v
    join stays s using(subject_id,hadm_id,stay_id) join vitals_items i on v.itemid=i.itemid
    where v.charttime>=s.admittime and v.charttime<s.landmark and v.charttime between s.intime and s.outtime
      and v.storetime between v.charttime and s.landmark and v.valuenum between i.lo and i.hi
      and lower(trim(v.valueuom))=i.unit
    """
    return compose_features(db.execute(sql).df())


def main():
    hashes=source_audit()
    admissions=pd.read_csv(HOSP/"admissions.csv.gz",usecols=KEYS+["admittime","dischtime","deathtime","hospital_expire_flag"],
                            parse_dates=["admittime","dischtime","deathtime"])
    patients=pd.read_csv(HOSP/"patients.csv.gz",parse_dates=["dod"])
    candidates=roster(admissions,patients)
    print("history-eligible admissions",len(candidates),"people",candidates.subject_id.nunique(),flush=True)
    conn=duckdb.connect()
    conn.execute("set threads=4"); conn.execute("set memory_limit='6GB'")
    ls=CACHE/"candidate_labs.csv.gz"
    if not ls.exists():
        print("scanning hospital laboratory records",flush=True);write_csv(ls,labs(conn,candidates))
    panel=pd.read_csv(ls)
    assert panel.hadm_id.is_unique
    valid=candidates.merge(panel,on=KEYS,validate="one_to_one")
    chosen=valid.sort_values(["subject_id","admittime","hadm_id"]).drop_duplicates("subject_id")
    vi=CACHE/"vitals.csv.gz"
    if not vi.exists():
        print("scanning eligible ICU monitors",len(chosen),flush=True);write_csv(vi,vitals(conn,chosen[candidates.columns]))
    wide=chosen.merge(pd.read_csv(vi),on=KEYS,how="left",validate="one_to_one")
    assert wide.subject_id.is_unique and wide.hadm_id.is_unique and len(wide)==len(chosen)
    save=CACHE/"history_index_lab_panel.csv.gz";write_csv(save,wide)
    history=admissions.merge(wide[KEYS+["admittime"]].rename(columns={"hadm_id":"index_hadm_id","admittime":"index_admittime"}),
                             on="subject_id",validate="many_to_one")
    prior=history[history.dischtime.lt(history.index_admittime)&history.dischtime.ge(history.admittime)&history.hospital_expire_flag.eq(0)].copy()
    assert set(prior.subject_id)==set(wide.subject_id) and not prior.hadm_id.isin(wide.hadm_id).any()
    prior_codes=pd.read_csv(ICU/"diagnoses_icd.csv.gz",usecols=KEYS+["icd_version","icd_code"],dtype={"icd_code":"string"})
    icd=prior[KEYS].merge(prior_codes,on=KEYS,validate="one_to_many")
    assert np.isin(icd.icd_version,[9,10]).all()
    prior_file=CACHE/"prior_codes.csv.gz";write_csv(prior_file,icd)
    current=OUT/"cohort_audit.json"
    write_json(current,dict(source_sha256=hashes,n_with_pre_landmark_eligibility=candidates.subject_id.nunique(),
               n_admissions_with_pre_landmark_eligibility=len(candidates),n_with_qualifying_lab=len(wide),
               n_with_prior_coded_stay=icd.subject_id.nunique(),index_hospital_deaths=int(wide.hospital_event.sum()),
               deaths_by_365d=int(wide.event.sum()),n_prior_admissions=len(prior),prior_icd_records=len(icd),
               n_people_with_icu_measurements=len(pd.read_csv(vi)),first_admissions_checked_against_formal_algorithm=165661,
               all_past_stays_discharge_before_index_admission=True,panel_sha256=digest(save),prior_codes_sha256=digest(prior_file),
               candidate_lab_sha256=digest(ls),vitals_sha256=digest(vi)))
    print("completed index labs, vitals and strictly prior ICD extraction",len(wide),flush=True)


if __name__=="__main__":main()

