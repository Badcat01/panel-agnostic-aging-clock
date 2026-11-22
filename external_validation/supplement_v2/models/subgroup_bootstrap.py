"""Patient bootstrap intervals for subgroup discrimination and calibration."""
import sys
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from numba import njit, prange, set_num_threads
from threadpoolctl import threadpool_limits
from design import OUT, EXT, RUN, original, read

TECH = OUT.parent / "technical"
sys.path.insert(0, str(TECH))
sys.path.insert(0, str(EXT/"scripts"))
from shared import Ensemble, REFERENCE, get_cohorts, h0
from validate_external import weighted_c

METRICS = ["harrell_C", "observed_KM", "mean_predicted", "E_over_O"]
COHORTS = ["temporal","nhanes3","hrs","charls","elsa","mimic"]

def write_summary(table):
    labels={"temporal":"NHANES 时间留出","nhanes3":"NHANES III","hrs":"HRS","charls":"CHARLS","elsa":"ELSA","mimic":"MIMIC-IV"}
    rows=["# 冻结时钟在年龄和性别亚组中的死亡风险预测","",
          "纳入原正式分析中可获得冻结全身分数、年龄、性别和生存结局的个体。五个社区样本观察三年死亡，MIMIC-IV 从入院满 24 小时观察一年死亡。Harrell C 使用各队列全部可用随访。E/O 是个人预测死亡概率的均值除以同组 Kaplan–Meier 估计的死亡概率，1 代表两者相同。","",
          "| 队列 | 人数 / 时限内死亡 | Harrell C（95% CI） | 实际死亡率 %（95% CI） | 平均预测 %（95% CI） | E/O（95% CI） |",
          "|---|---:|---:|---:|---:|---:|"]
    def ci(row,key,pct=False):
        digits=1 if pct else (3 if key=="harrell_C" else 2)
        factor=100 if pct else 1
        fmt=lambda x:f"{x*factor:.{digits}f}"
        return f"{fmt(row[key])}（{fmt(row[key+'_low'])}–{fmt(row[key+'_high'])}）"
    for _,s in table[table.group=="all"].iterrows():
        cells=[labels[s.cohort],f"{s.n:,} / {s.events_to_h:,}",ci(s,"harrell_C"),ci(s,"observed_KM",True),ci(s,"mean_predicted",True),ci(s,"E_over_O")]
        rows.append("| "+" | ".join(cells)+" |")
    keyed=table.set_index(["cohort","group"])
    rows.extend(["","- HRS 的三年风险在三个可评估年龄组均被高估。50–64、65–79 及 ≥80 岁组的 E/O 依次为 "+
                 "、".join(ci(keyed.loc["hrs",g],"E_over_O") for g in ["age50_64","age65_79","age80plus"])+"。",
                 "- NHANES III 的总体 E/O 接近 1，但女性为 "+ci(keyed.loc["nhanes3","female"],"E_over_O")+"，男性为 "+ci(keyed.loc["nhanes3","male"],"E_over_O")+"。ELSA 的女性和男性 E/O 分别为 "+ci(keyed.loc["elsa","female"],"E_over_O")+" 和 "+ci(keyed.loc["elsa","male"],"E_over_O")+"。",
                 "- MIMIC-IV 的四个年龄段均低估一年风险。20–49 岁 E/O 为 "+ci(keyed.loc["mimic","age20_49"],"E_over_O")+"，≥80 岁为 "+ci(keyed.loc["mimic","age80plus"],"E_over_O")+"。",
                 "","每个性别或年龄亚组分别按患者有放回抽样 2,000 次。每一次重抽同时重算 C、KM、预测均值及二者的比，取分布的 2.5% 和 97.5% 分位数。冻结权重及 NHANES 开发集基线风险保持固定。年龄用模型原输入，按 <50、50–64、65–79、≥80 岁分层。依照既有生存亚组分析，仅计算人数至少 40 且全随访死亡至少 10 例的组。42 个总体或亚组中可估计 40 个，另外两组是 HRS 的 <50 岁（147 人、3 年内死亡 2 例）与 ELSA 的 <50 岁（无受试者）。全部 40 个估计的每项指标均有 2,000 个有效重抽样值。区间反映各亚组内抽样不确定性，组间差异尚需正式交互检验。",
                 "","[全部亚组及 95% 区间](subgroup_bootstrap.csv) · [来源校验和](subgroup_bootstrap_audit.json) · [计算代码](subgroup_bootstrap.py)"])
    (OUT/"年龄性别亚组结果.md").write_text("\n".join(rows)+"\n",encoding="utf8")

@njit(cache=True)
def score(t, e, rank, q, wt, h):
    n = len(t)
    at_risk = wt.sum()
    survival = 1.0
    i = n - 1
    while i >= 0 and t[i] <= h:
        last = i
        removed = 0.0
        deaths = 0.0
        while i >= 0 and t[i] == t[last]:
            removed += wt[i]
            deaths += wt[i] * e[i]
            i -= 1
        if deaths:
            survival *= 1.0 - deaths / at_risk
        at_risk -= removed
    observed = 1.0 - survival
    predicted = np.sum(wt * q) / wt.sum()
    return np.array([weighted_c(t,e,rank,wt), observed, predicted,
                     predicted / observed if observed > 0 else np.nan])

@njit(cache=True, parallel=True)
def bootstrap(t,e,rank,q,h,seed):
    estimates = np.empty((2000,4))
    n = len(t)
    for rep in prange(2000):
        np.random.seed(seed+rep)
        wt = np.bincount(np.random.randint(0,n,n),minlength=n)
        estimates[rep] = score(t,e,rank,q,wt,h)
    return estimates

def hazard():
    ref = read(REFERENCE)
    assert ref["freeze_sha256"] == original.sha(RUN/"model_freeze.json")
    net = Ensemble("cpu")
    torch.set_num_threads(2)
    dev, _, _ = get_cohorts()
    lp = ref["risk_lp_slope"] * (net.score(dev) - ref["risk_lp_mean"])
    base = {h:h0(dev.time_yr.to_numpy(),dev.event.to_numpy(),lp,h) for h in [1,3]}
    np.testing.assert_allclose(base[3],ref["baseline_hazard"]["clock"]["3"],rtol=1e-6)
    base[3] = ref["baseline_hazard"]["clock"]["3"]
    return ref,base

def main():
    ref,base=hazard()
    rows=[]; checks={}
    for co_index,co in enumerate(COHORTS):
        src=RUN/"temporal_predictions.csv.gz" if co=="temporal" else (
            EXT/"mimic_iv/formal_v3_1/predictions.csv.gz" if co=="mimic" else EXT/"results"/co/"predictions.csv.gz")
        idcol="SEQN" if co=="temporal" else "ID"
        d=pd.read_csv(src,dtype={idcol:str},usecols=[idcol,"time_yr","event","full_lp","age","male" if co=="temporal" else "sex"]).rename(columns={"male":"sex"})
        assert d[idcol].is_unique
        d=d.loc[d.time_yr.gt(0)&d.event.isin([0,1])&d.sex.isin([0,1])&d.age.notna()&d.full_lp.notna()].copy()
        h=1 if co=="mimic" else 3
        d["prob"]=-np.expm1(-base[h]*np.exp(ref["risk_lp_slope"]*(d.full_lp-ref["risk_lp_mean"])))
        d=d.sort_values("time_yr",ascending=False,kind="stable")
        subsets={"all":np.ones(len(d),bool),"female":d.sex.eq(0),"male":d.sex.eq(1),
                 "age20_49":d.age.lt(50),"age50_64":d.age.ge(50)&d.age.lt(65),
                 "age65_79":d.age.ge(65)&d.age.lt(80),"age80plus":d.age.ge(80)}
        assert sum(int(subsets[g].sum()) for g in list(subsets)[3:])==len(d)
        old=pd.read_csv(TECH/"survival"/f"{co}_subgroups.csv").set_index("group")
        checks[co]={"sha256":original.sha(src),"n":len(d),"development_h0":base[h]}
        for k,(name,sel) in enumerate(subsets.items()):
            f=d.loc[sel]
            t=f.time_yr.to_numpy();e=f.event.to_numpy(dtype=np.int64);p=f.prob.to_numpy()
            cases=(e==1)&(t<=h)
            row={"cohort":co,"group":name,"horizon_y":h,"n":len(f),"all_events":int(e.sum()),
                 "events_to_h":int(cases.sum()),"n_at_or_beyond_h":int((t>=h).sum())}
            if len(f)<40 or e.sum()<10:
                row["status"]="below_n40_or_total_events10"
            else:
                rank=np.unique(f.full_lp,return_inverse=True)[1]+1
                point=score(t,e,rank,p,np.ones(len(t),dtype=np.int64),h)
                np.testing.assert_allclose(point[0],concordance_index(t,-f.full_lp,e),atol=1e-10)
                np.testing.assert_allclose(point[1],1-KaplanMeierFitter().fit(t,e).predict(h),atol=1e-10)
                np.testing.assert_allclose(point,old.loc[name,["harrell_c","observed_KM","predicted","E_over_O"]].to_numpy(float),rtol=3e-5,atol=1e-7)
                sums=bootstrap(t,e,rank,p,h,20261121+100000*co_index+3000*k)
                row["status"]="estimated" if cases.sum()>=10 else "estimated_fewer_than_10_horizon_events"
                row["B"]=len(sums)
                for ix,metric in enumerate(METRICS):
                    vals=sums[:,ix];finite=np.isfinite(vals)
                    lo,hi=np.percentile(vals[finite],[2.5,97.5]) if finite.any() else [np.nan,np.nan]
                    row.update({metric:point[ix],metric+"_low":lo,metric+"_high":hi,metric+"_valid_B":int(finite.sum())})
            rows.append(row)
        tmp=OUT/"subgroup_bootstrap.tmp"
        pd.DataFrame(rows).to_csv(tmp,index=False,encoding="utf-8-sig")
        tmp.replace(OUT/"subgroup_bootstrap.csv")
        print(co,len(d),"done",flush=True)
    original.save_json(OUT/"subgroup_bootstrap_audit.json",{"freeze":ref["freeze_sha256"],"cohorts":checks,
                       "replicates":2000,"tech_threshold":"n>=40 and total deaths>=10","outputs_sha256":original.sha(OUT/"subgroup_bootstrap.csv")})
    write_summary(pd.DataFrame(rows))

if __name__=="__main__":
    set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
