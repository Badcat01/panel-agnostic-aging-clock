"""Refresh aggregate comparative results, heterogeneity, figures, and coverage."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

from design import OUT, original


def meta(frame):
    rows=[]
    ext=frame[(frame.setting=="external")&frame.panel.isin(["nhanes3","hrs","charls","elsa"])]
    for model,sub in ext.groupby("model"):
        sub=sub.dropna(subset=["paired_se"])
        if len(sub)<3 or not sub.paired_se.gt(0).all():
            continue
        y=sub.delta_M0_minus_comparator.to_numpy()
        w=1/sub.paired_se.to_numpy()**2
        df=len(y)-1
        q=np.dot(w,(y-np.average(y,weights=w))**2)
        tau2=max(0,(q-df)/(w.sum()-np.square(w).sum()/w.sum()))
        weights=1/(1/w+tau2)
        effect=np.average(y,weights=weights)
        inflation=max(1,np.dot(weights,(y-effect)**2)/df)
        se=np.sqrt(inflation/weights.sum())
        gap=t.ppf(.975,df)*se
        rows.append(dict(model=model,k=len(sub),cohorts=",".join(sub.panel),delta=effect,low=effect-gap,
                         high=effect+gap,tau2=tau2,I2=100*max(0,1-df/q) if q>0 else 0))
    return pd.DataFrame(rows)


def heatmap(sim,title,destination):
    order=sim[["panel","panel_size"]].drop_duplicates().sort_values("panel_size").panel.tolist()
    table=sim.pivot(index="panel",columns="model",values="delta_M0_minus_comparator").reindex(order)
    fig,ax=plt.subplots(figsize=(max(8,len(table.columns)*1.02),4.7),layout="constrained")
    vmax=min(.025,np.nanmax(np.abs(table)))
    h=ax.imshow(table,cmap="RdBu",vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=32,ha="right")
    ax.set_yticks(range(len(table)),[x.upper() for x in table.index])
    for i in range(len(table)):
        for j in range(len(table.columns)):
            v=table.iat[i,j]
            if np.isfinite(v):
                ax.text(j,i,f"{v:+.3f}",ha="center",va="center",fontsize=8,
                        color="white" if abs(v)>.65*vmax else "black")
    ax.set_title(title)
    fig.colorbar(h,ax=ax,shrink=.8,label="Frozen FT-Mask minus comparator")
    fig.savefig(OUT/destination,dpi=180)
    plt.close(fig)

def main():
    d=pd.read_csv(OUT/"comparisons.csv")
    if "paired_se" not in d:
        return
    estimates=meta(d)
    estimates.to_csv(OUT/"community_random_effects.csv",index=False)
    sim=d[d.setting==("primary" if (d.setting=="primary").any() else "sim")]
    heatmap(sim,"NHANES temporal holdout: paired Harrell C difference","temporal_paired_heatmap.png")
    heatmap(d[d.setting=="external"],"External cohorts on matched panels: paired Harrell C difference","external_paired_heatmap.png")
    if len(estimates):
        fig,ax=plt.subplots(figsize=(7,max(3.3,.48*len(estimates)+1.3)),layout="constrained")
        ix=np.arange(len(estimates))
        ax.errorbar(estimates.delta,ix,xerr=np.array([estimates.delta-estimates.low,
                    estimates.high-estimates.delta]),fmt="o",color="#225e75",capsize=3)
        ax.set_yticks(ix,[f"{r.model}  (I² {r.I2:.0f}%)" for r in estimates.itertuples()])
        ax.axvline(0,ls="--",color="0.5",lw=1)
        ax.set_xlabel("FT-Mask minus comparator, Harrell C (95% CI)")
        ax.set_title("Community external cohorts: random-effects summary")
        fig.savefig(OUT/"external_random_effects.png",dpi=180);plt.close(fig)
    datafile=OUT/"random_panel_richness.csv"
    if datafile.exists():
        z=pd.read_csv(datafile).pivot(index=["available_assays","seed"],columns="model",values="c_index")
        fig,ax=plt.subplots(figsize=(7,4.3),layout="constrained")
        for m in z:
            if m=="M0":continue
            delta=(z.M0-z[m]).groupby(level=0)
            avg=delta.mean()
            ax.plot(avg.index,avg,"o-",label=m)
            ax.fill_between(avg.index,delta.min(),delta.max(),alpha=.12)
        ax.axhline(0,ls="--",color="0.5",lw=1)
        ax.set_xticks(sorted(z.index.get_level_values(0).unique()))
        ax.set_xlabel("Exactly this many observed assays per patient")
        ax.set_ylabel("Frozen FT-Mask minus comparator, Harrell C")
        ax.set_title("Same NHANES patients, three random nested panel assignments")
        ax.legend(frameon=False)
        fig.savefig(OUT/"random_panel_richness.png",dpi=180);plt.close(fig)
    availability=[]
    for p in sorted((OUT/"predictions").glob("*.csv.gz")):
        f=pd.read_csv(p)
        for k in f.columns[4:]:
            ok=np.isfinite(f[k])
            availability.append({"scenario":p.name[:-7],"model":k,"eligible_n":len(f),
                                 "scorable_n":int(ok.sum()),"scorable_events":int(f.event[ok].sum())})
    pd.DataFrame(availability).to_csv(OUT/"scoring_coverage.csv",index=False)
    from write_readout import make_readout
    make_readout(d,estimates)


if __name__=="__main__":
    main()
