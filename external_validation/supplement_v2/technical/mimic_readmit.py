"""Readmission diagnoses and marker-count thresholds for the frozen MIMIC scores."""
import io
import json
import zipfile
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from study_paths import MIMIC_ARCHIVE
from shared import HERE,EXT,read,sha,verify
MIM=EXT/"mimic_iv"
sys.path.insert(0,str(MIM/"incident_v3_1"/"scripts"))
from evaluate_incident import evaluate, ORGAN_REFERENCE, ENDPOINTS
from mimic_iv_panels import oof_comparison,compare_auc,evaluate_panel,flatten

# Principal diagnosis on any distinct subsequent hospitalization. A broad infection
# proxy is intentionally restricted to coded sepsis/severe sepsis.
CODES={
 "liver":(r"^(570|572[24]|4560$|45620$)",r"^(K72|K767|I8501|I8511)"),
 "kidney":(r"^(584|V56)",r"^(N17|Z49)"),
 "metabolic":(r"^250[12]",r"^E(08|09|10|11|13)[01]"),
 "immune_heme":(r"^(038|9959[12]|78552)",r"^(A40|A41|R652)"),
 "cardiovascular":(r"^(410|428|43[0126]|43[34][0-9]1$)",r"^(I21|I22|I50|I6[01234])")}

def diagonal(base,organs,y,match):
    folds=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=20261121).split(base,y):
        scores=[]
        for organ in organs.T:
            x=np.column_stack([base,organ]);x=(x-x[tr].mean(0))/x[tr].std(0)
            fit=LogisticRegression(penalty=None,max_iter=500).fit(x[tr],y[tr])
            scores.append(fit.predict_proba(x[te])[:,1])
        folds.append(compare_auc(np.stack(scores,axis=1),y[te]))
    weights=np.array([f[2] for f in folds],float);weights/=weights.sum()
    means=sum(w*f[0] for w,f in zip(weights,folds))
    cov=sum(w**2*f[1] for w,f in zip(weights,folds))
    i=list(CODES).index(match);rows=[]
    for j,other in enumerate(CODES):
        if j==i:continue
        se=np.sqrt(max(0,cov[i,i]+cov[j,j]-2*cov[i,j]));delta=means[i]-means[j]
        rows.append(dict(endpoint=match,other=other,n=len(y),events=y.sum(),matched_AUC=means[i],other_AUC=means[j],
                         delta=delta,lo=delta-1.96*se,hi=delta+1.96*se,p=norm.sf(delta/se) if se else 1.))
    return rows

def main():
    b=verify();ref=read(ORGAN_REFERENCE)
    scores_path=MIM/"formal_v3_1"/"predictions.csv.gz"
    assert sha(scores_path)==read(MIM/"formal_v3_1"/"audit.json")["predictions_sha256"]
    scores=pd.read_csv(scores_path).set_index("ID",verify_integrity=True)
    src=MIM/"local_3_1"/"hosp"
    a=pd.read_csv(src/"admissions.csv.gz",usecols=["subject_id","hadm_id","admittime","dischtime","hospital_expire_flag"],
                  parse_dates=["admittime","dischtime"])
    pat=pd.read_csv(src/"patients.csv.gz",usecols=["subject_id","anchor_year","anchor_year_group"])
    archive=MIMIC_ARCHIVE
    with zipfile.ZipFile(archive) as z:
        member="hosp/diagnoses_icd.csv.gz"; prefix="mimic-iv-3.1/"
        checksum={p.lstrip("*"):s for s,p in (x.split(maxsplit=1) for x in z.read(prefix+"SHA256SUMS.txt").decode().splitlines())}
        for nm in ["hosp/admissions.csv.gz","hosp/patients.csv.gz"]:
            assert sha(src/nm.split("/")[-1])==checksum[nm]
        content=z.read(prefix+member); assert hashlib.sha256(content).hexdigest()==checksum[member]
        dx=pd.read_csv(io.BytesIO(content),compression="gzip",dtype={"icd_code":"string"})
    idx=pd.read_csv(MIM/"formal_v3_1"/"landmark_index.csv.gz",usecols=["subject_id","hadm_id","landmark"],parse_dates=["landmark"])
    idx=idx[idx.subject_id.isin(scores.index)].merge(a,on=["subject_id","hadm_id"],validate="one_to_one").merge(pat,on="subject_id",validate="one_to_one")
    idx["horizon"]=idx.landmark+pd.Timedelta(days=365)
    group_last=idx.anchor_year_group.str.extract(r"(\d{4})\s*$")[0].astype(int)
    # One calendar-year safety margin for deidentified dates; not registry-wide readmissions.
    covered=group_last+(idx.horizon.dt.year-idx.anchor_year)<=2021
    risk=idx[covered&idx.hospital_expire_flag.eq(0)&idx.dischtime.lt(idx.horizon)].copy()
    future=a.merge(risk[["subject_id","hadm_id","dischtime","horizon"]],on="subject_id",suffixes=("","_index"))
    future=future[future.hadm_id.ne(future.hadm_id_index)&future.admittime.gt(future.dischtime_index)&future.admittime.le(future.horizon)]
    assert risk.subject_id.is_unique and future.hadm_id.is_unique
    obs=dx[dx.hadm_id.isin(set(future.hadm_id)|set(risk.hadm_id))].copy()
    code=obs.icd_code.str.upper().str.strip(); v9=obs.icd_version.eq(9);v10=obs.icd_version.eq(10)
    results=[];matrix=[];counts=[];diag=[]
    for panel,(c9,c10) in CODES.items():
        flags=obs[(v9&code.str.match(c9))|(v10&code.str.match(c10))]
        previous=set(flags.hadm_id)&set(risk.hadm_id)
        nosame=risk.loc[~risk.hadm_id.isin(previous),"subject_id"]
        for kind in ["principal","any"]:
            identified=flags[flags.seq_num.eq(1)] if kind=="principal" else flags
            cases=set(future.loc[future.hadm_id.isin(identified.hadm_id),"subject_id"])
            d=scores.loc[nosame].copy();d["event"]=d.index.isin(cases).astype(int)
            age=(d.age-60)/10
            base=np.column_stack([age,age**2,d.sex,d.full_accel/ref["panels"]["full"]["accel_sd_years"]])
            all5=d[[p+"_accel" for p in CODES]].notna().all(axis=1)
            if kind=="principal" and d.loc[all5,"event"].sum()>=10:
                diag.extend(diagonal(base[all5],d.loc[all5,[p+"_accel" for p in CODES]].to_numpy(),
                                     d.loc[all5,"event"].to_numpy(bool),panel))
            for size,gate in [("one_or_more",d[panel+"_n_observed"].ge(1)),("at_least_two",d[panel+"_n_observed"].ge(2)),
                              ("at_least_40pct",d[panel+"_n_observed"].ge(np.ceil(.4*len(b["groups"][panel])))),("all_five_scored",all5)]:
                if (d.event[gate].sum()<10):continue
                for candidate in (list(CODES) if size=="all_five_scored" and kind=="principal" else [panel]):
                    y=d.event[gate].to_numpy(bool)
                    score=d.loc[gate,candidate+"_accel"].to_numpy()/ref["panels"][candidate]["accel_sd_years"]
                    pair=oof_comparison(base[gate],score,y,20261121)
                    row=dict(endpoint=panel,organ=candidate,code_position=kind,eligibility=size,n=len(y),events=int(y.sum()),
                             **{k:v for k,v in pair.items() if not isinstance(v,list)},
                             ci_low=pair["incremental_auc_oof_ci95"][0],ci_high=pair["incremental_auc_oof_ci95"][1])
                    (matrix if size=="all_five_scored" else results).append(row)
            counts.append(dict(panel=panel,code_position=kind,index_excluded=len(previous),eligible_before_scores=len(nosame),
                               any_local_readmission=nosame.isin(future.subject_id).sum(),any_target=nosame.isin(cases).sum()))
        print(panel,"readmissions complete",flush=True)
    df=pd.DataFrame(results)
    for _,s in df.groupby(["code_position","eligibility"]):df.loc[s.index,"q_five"]=multipletests(s.incremental_auc_oof_p,method="fdr_bh")[1]
    df.to_csv(HERE/"readmission_matched.csv",index=False)
    pd.DataFrame(matrix).to_csv(HERE/"readmission_common_matrix.csv",index=False)
    diagonals=pd.DataFrame(diag)
    p_all=diagonals.groupby("endpoint").p.max().reindex(CODES,fill_value=1.)
    diagonals["dominance_q_five"]=diagonals.endpoint.map(dict(zip(p_all.index,multipletests(p_all,method="fdr_bh")[1])))
    diagonals.to_csv(HERE/"readmission_diagonal.csv",index=False)
    pd.DataFrame(counts).to_csv(HERE/"readmission_flow.csv",index=False)
    meta=dict(definitions=CODES,index_n=len(idx),eligible_era=int(covered.sum()),survived_discharge_and_within_year=len(risk),
              note="First observed admission design has zero previous local inpatient Charlson or Elixhauser history. No current-discharge diagnosis entered as a predictor.")
    (HERE/"readmission_design.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    threshold=[]
    for panel in CODES:
        inp=MIM/"incident_v3_1"/"inputs"/f"{panel}.csv.gz"
        assert sha(inp)==read(MIM/"incident_v3_1"/"audit"/f"{panel}.json")["input_sha256"]
        end=pd.read_csv(inp).set_index("ID",verify_integrity=True)
        d=scores.drop(columns="event").loc[end.index].join(end[["event"]],validate="one_to_one")
        for tag,k in [("one_or_more",1),("at_least_two",2),("at_least_40pct",int(np.ceil(.4*len(b["groups"][panel]))))]:
            sub=d[d[panel+"_n_observed"]>=k]
            if sub.empty:continue
            a=evaluate(sub,panel,ref,"primary")
            threshold.append(dict(panel=panel,rule=tag,minimum=k,n=len(sub),events=int(sub.event.sum()),**{k:v for k,v in a.items() if k.startswith("incremental") or k.endswith("auc_oof")}))
    pd.DataFrame(threshold).to_csv(HERE/"incident_marker_thresholds.csv",index=False)
    mortality=[]
    for panel in CODES:
        for label,count in [("one_or_more",1),("at_least_two",2),("at_least_40pct",int(np.ceil(.4*len(b["groups"][panel]))))]:
            eligible=scores[scores[panel+"_n_observed"]>=count]
            for endpoint in ["index_hospital_mortality","post_landmark_365d_mortality"]:
                row=flatten(evaluate_panel(eligible,panel,endpoint,ref,20261121))
                mortality.append(dict(rule=label,minimum=count,**row))
    pd.DataFrame(mortality).to_csv(HERE/"mortality_marker_thresholds.csv",index=False)

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
