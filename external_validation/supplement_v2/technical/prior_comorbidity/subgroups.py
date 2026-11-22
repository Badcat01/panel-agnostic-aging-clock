"""Repeat same-patient paired increments within prespecified admission, ICU and age categories."""
from compare import oof_comparison, threadpool_limits
from cohort import CACHE, OUT, ICU, HOSP, np, pd
from statsmodels.stats.multitest import multipletests


def run():
    d=pd.read_csv(CACHE/"all_scores.csv.gz").rename(columns={"ID":"subject_id"})
    a=pd.read_csv(HOSP/"admissions.csv.gz",usecols=["subject_id","hadm_id","admission_type","admittime"],parse_dates=["admittime"])
    d=d.merge(a,on=["subject_id","hadm_id"],validate="one_to_one")
    s=pd.read_csv(ICU/"icustays.csv.gz",usecols=["subject_id","hadm_id","intime","outtime"],parse_dates=["intime","outtime"])
    s=s.merge(d[["subject_id","hadm_id","admittime"]],on=["subject_id","hadm_id"],validate="many_to_one")
    active=s.intime.lt(s.admittime+pd.Timedelta(days=1))&s.outtime.ge(s.admittime)&s.intime.lt(s.outtime)
    d["early_ICU"]=np.where(d.hadm_id.isin(s.loc[active,"hadm_id"]),"ICU within 24h","no ICU within 24h")
    d["age_group"]=pd.cut(d.age,bins=[20,50,65,80,np.inf],labels=["20–49","50–64","65–79","80+"],right=False)
    rows=[]
    for field in ["admission_type","early_ICU","age_group"]:
        for level,sub in d.groupby(field,observed=True):
            age=(sub.age.to_numpy()-60)/10
            demo=np.column_stack([age,age**2,sub.sex])
            for endpoint in ["hospital_event","event"]:
                y=sub[endpoint].to_numpy(dtype=bool)
                for score in ["charlson","elixhauser"]:
                    row=dict(stratum=field,category=level,endpoint=endpoint,comorbidity=score,n=len(y),events=int(y.sum()))
                    if min(y.sum(),(~y).sum())>=20 and sub[score].nunique()>1:
                        row.update(oof_comparison(np.column_stack([demo,sub[score]]),sub.full_accel.to_numpy(),y,20261121))
                    else:
                        row["reason"]="fewer than 20 cases/controls or no score variation"
                    rows.append(row)
    tab=pd.DataFrame(rows)
    eligible=tab.incremental_auc_oof_p.notna()
    tab.loc[eligible,"q_BH_all_estimable"]=multipletests(tab.loc[eligible,"incremental_auc_oof_p"],method="fdr_bh")[1]
    tab.to_csv(OUT/"prior_score_subgroups.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        run()
