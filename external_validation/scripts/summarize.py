"""Build cohort tables and risk calibration plots from verified outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from reference import ROOT, REFERENCE, sha, verify

OUT = ROOT / "results"
NAMES = {"charls":"CHARLS","hrs":"HRS","elsa":"ELSA","mhas":"MHAS","share":"SHARE","klosa":"KLoSA","lasi":"LASI-DAD"}
TYPES = ["预定","预定","预定","补充","补充","探索","仅评分"]
LOCATIONS = ["中国","美国","英国","墨西哥","欧洲与以色列","韩国","印度"]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def tables():
    verify()
    metrics, horizon, sensitivity = [], [], []
    for (key,title),kind,place in zip(NAMES.items(),TYPES,LOCATIONS):
        d = read(OUT/key/"metrics.json")
        src = ROOT/"inputs"/(key+".csv.gz")
        assert d["source_sha256"] == sha(src) and d["reference_sha256"] == sha(REFERENCE)
        inp = pd.read_csv(src,dtype={"ID":"string"})
        pred = pd.read_csv(OUT/key/"predictions.csv.gz",dtype={"ID":"string"})
        assert inp.ID.is_unique and inp.ID.equals(pred.ID)
        assert len(pred)==d["scored"] and int(pred.event.sum())==d["deaths"]
        assert pred.time_yr.gt(0).sum()==d["n"]
        for group in ["full","liver","kidney","metabolic","immune_heme","cardiovascular"]:
            assert pred[group+"_lp"].notna().equals(pred[group+"_n_observed"].gt(0))
        m = {"cohort":title,"region":place,"role":kind,"scored":d["scored"],"survival_n":d["n"],
             "deaths":d["deaths"] if d["n"] else None,"median_features":d["median_features"],
             "feature_types":d["feature_types"],"followup_median":d.get("observed_followup_median_years")}
        if d["n"]:
            c = d["same_sex_available"]
            m.update(paired_n=d["n_sex"], c_clock=c["full_lp"]["C"],c_low=c["full_lp"]["CI95"][0],
                     c_high=c["full_lp"]["CI95"][1],c_age=c["age"]["C"],c_agesex=c["age_sex_lp"]["C"],c_ridge=c["ridge_lp"]["C"])
            for name,col in [("agesex","age_sex_lp"),("ridge","ridge_lp")]:
                m["delta_"+name]=c[col]["delta_clock_minus_comparator"]
                m[name+"_low"],m[name+"_high"]=c[col]["delta_CI95"]
            h = d["accel_association"]
            m.update(hr_sd=h["HR"],hr_low=h["CI95"][0],hr_high=h["CI95"][1],sd_years=h["SD_years"])
            for k,y in d["sensitivities"].items():
                sensitivity.append(dict(cohort=title,analysis=k,n=y["n"],events=y["deaths"],
                                    c_clock=y.get("full_lp",{}).get("C"),c_age=y.get("age",{}).get("C")))
            for years,row in d["fixed_times"].items():
                for key_model in ["clock","age_sex","ridge"]:
                    model = row.get(key_model,{})
                    horizon.append(dict(cohort=title,years=int(years),method=key_model,supported=row["supported"],
                         n=row["n"],cases=row["cases"],controls=row["controls"],censor_survival=row["G"],
                         auc=model.get("auc"),brier=model.get("brier"),predicted=model.get("predicted_mean"),
                         observed_KM=row.get("observed_km"),continuous_nri_vs_agesex=row.get("continuous_nri_vs_age_sex") if key_model=="clock" else None,
                         idi_vs_agesex=row.get("idi_vs_age_sex") if key_model=="clock" else None))
        metrics.append(m)
    metrics, horizon = pd.DataFrame(metrics), pd.DataFrame(horizon)
    metrics.to_csv(OUT/"cohort_summary.csv",index=False,encoding="utf-8-sig")
    horizon.to_csv(OUT/"fixed_horizons.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(OUT/"sensitivity_summary.csv",index=False,encoding="utf-8-sig")
    return metrics, horizon


def figures(m):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,
                        "axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    v = m.dropna(subset=["c_clock"]).reset_index(drop=True)
    fig,ax = plt.subplots(figsize=(7.3,3.8),layout="constrained")
    for off,field,name,color in [(0.12,"agesex","对照为年龄和性别 Cox","#196a80"),(-0.12,"ridge","对照为相同输入的线性 Cox","#b4662c")]:
        center = v["delta_"+field].to_numpy()
        xerr = np.array([center-v[field+"_low"],v[field+"_high"]-center])
        ax.errorbar(center,np.arange(len(v))+off,xerr=xerr,fmt="o",markersize=5,capsize=3,color=color,label=name)
    ax.axvline(0,color="#585858",lw=1,ls="--")
    ax.set_yticks(np.arange(len(v)),v.cohort)
    ax.invert_yaxis()
    ax.set_xlabel("冻结时钟 C-index 减去对照 C-index（配对 95% 区间）")
    ax.grid(axis="x",alpha=.18)
    ax.legend(frameon=False,loc="lower right")
    fig.savefig(OUT/"discrimination.png",dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6.6),layout="constrained")
    for ax,key in zip(axes.flat,["charls","hrs","elsa","mhas","share","klosa"]):
        cal = pd.read_csv(OUT/key/"calibration.csv")
        hi = max(cal[["predicted","observed_km"]].max())*1.08
        ax.plot([0,hi],[0,hi],lw=1,color="#777",ls="--")
        for h,g in cal.groupby("horizon"):
            color = {3:"#196a80",5:"#b4662c",10:"#77669d"}[h]
            ax.plot(g.predicted,g.observed_km, color=color,marker="o",ms=3.5,label=f"{h}年")
        ax.set(xlim=(0,hi),ylim=(0,hi),title=NAMES[key])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.legend(fontsize=8,frameon=False,loc="upper left")
        ax.grid(alpha=.12)
    fig.supxlabel("NHANES 模型预测的平均死亡风险")
    fig.supylabel("外部随访观察死亡率（Kaplan–Meier）")
    fig.savefig(OUT/"calibration.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    metrics, _ = tables()
    figures(metrics)
