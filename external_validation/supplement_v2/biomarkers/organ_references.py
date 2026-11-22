"""Prespecified within-NHANES organ specificity contrasts on frozen-score holdouts."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from score_holdout import (HERE, PROJECT, TRAIN_RUN, CLOCKS, PROXIES, CELLS, PANELS, SEED,
                           load_json, sha, write_json)

ACC = [f"{p}_accel" for p in PANELS]


def read_xpt(stem, manifest):
    path=HERE/"sources"/(stem+".xpt")
    assert sha(path)==manifest[stem]
    d=pd.read_sas(path,format="xport")
    numeric=d.select_dtypes(include="number")
    d[numeric.columns]=numeric.mask(np.isclose(numeric,16.0**-65,atol=0,rtol=1e-12),0)
    return d.set_index("SEQN",verify_integrity=True)


def sources():
    manifest={r["file"][:-4]:r["sha256"] for r in load_json(HERE/"sources/manifest.json")}
    hd=HERE/"dnam_frozen_scores.csv.gz"
    assert sha(hd)==load_json(HERE/"roster_audit.json")["output_sha256"]
    dnam=pd.read_csv(hd).set_index("SEQN",verify_integrity=True)
    tp=TRAIN_RUN/"temporal_predictions.csv.gz"
    assert sha(tp)==load_json(TRAIN_RUN/"temporal_metrics.json")["predictions_sha256"]
    temporal=pd.read_csv(tp).set_index("SEQN",verify_integrity=True)
    meta=pd.read_csv(PROJECT/"data/processed/nhanes_9918.csv.gz").set_index("SEQN",verify_integrity=True)
    temporal=temporal.join(meta[["wt_mec","strata","psu"]])
    dev=set(pd.read_csv(TRAIN_RUN/"development_ids.csv.gz").SEQN)
    assert not (set(dnam.index)&(dev|set(temporal.index)) or set(temporal.index)&dev)
    frames=[]
    for marker,matched in PROXIES.items():
        frames.append((dnam, marker, matched, "dnam_proxy", "WTDN4YR", "identity"))
    liver=temporal.join(read_xpt("LUX_J",manifest)).query("LUAXSTAT==1")
    for field in ["LUXSMED","LUXCAPM"]:
        frames.append((liver,field,["liver"],"temporal","wt_mec","log" if field=="LUXSMED" else "identity"))
    all_ins=[]
    for year,suffix in zip(range(2009,2019,2),"FGHIJ"):
        d=read_xpt(("GLU_" if year<2013 else "INS_")+suffix,manifest)
        if year>=2013:
            d=d.join(read_xpt("GLU_"+suffix,manifest)[["LBXGLU"]])
        d=d[d.WTSAF2YR.gt(0)].copy()
        if "PHAFSTHR" in d:
            hrs=d.PHAFSTHR+d.PHAFSTMN/60
            d=d[hrs.ge(8)&hrs.lt(24)]
        d["HOMA_IR"]=d.LBXIN*d.LBXGLU/405
        all_ins.append(d[["LBXIN","HOMA_IR","WTSAF2YR"]])
    ins=temporal.join(pd.concat(all_ins,verify_integrity=True),how="inner")
    for field in ["LBXIN","HOMA_IR"]:
        frames.append((ins,field,["metabolic"],"temporal","WTSAF2YR","log"))
    cys=dnam.join(pd.concat([read_xpt("SSCYST_A",manifest),read_xpt("SSCYST_B",manifest)],verify_integrity=True))
    frames.append((cys,"SSCYPC",["kidney"],"dnam_holdout","WTDN4YR","log"))
    for tab,cols in [("SSBNP_A",["SSBNP"]),("SSTROP_A",["SSTNT","SSTNIA","SSTNIS","SSTNIO"])]:
        d=dnam.join(read_xpt(tab,manifest))
        for field in cols:
            frames.append((d,field,["cardiovascular"],"dnam_holdout","WTDN4YR","log1p"))
    return dnam,frames


def regress_residuals(x,z,w):
    root=np.sqrt(w)[:,None]
    coef=np.linalg.lstsq(root*z,root*x,rcond=None)[0]
    return x-z@coef


def control_design(d,cells):
    blocks=[np.ones(len(d)),d.age,d.male,d.full_accel]
    if d.cycle.nunique()>1:
        blocks.extend(pd.get_dummies(d.cycle,drop_first=True).T.to_numpy())
    if cells:
        blocks.extend(d[CELLS].T.to_numpy())
    return np.column_stack(blocks).astype(float)


def matrix(frame,marker,matches,tier,weight,transform,mode):
    keep=[marker,*ACC,"age","male","cycle","full_accel",weight,"psu","strata"]
    if "cells" in mode:
        keep+=CELLS
    d=frame.dropna(subset=keep).loc[lambda f:f[weight].gt(0)].copy()
    y=d[marker].to_numpy()
    if transform=="log":
        assert (y>0).all()
        y=np.log(y)
    elif transform=="log1p":
        assert (y>=0).all()
        y=np.log1p(y)
    x=np.column_stack([d[ACC],y])
    w=np.ones(len(d)) if mode in ["primary","cells"] else d[weight].to_numpy()
    w=w/w.mean()
    z=control_design(d,"cells" in mode)
    residual=regress_residuals(x,z,w)
    standardized=residual/np.sqrt(np.mean(w[:,None]*residual**2,axis=0))
    x,y=standardized[:,:-1],standardized[:,[-1]]
    r=np.mean(w[:,None]*x*y,axis=0)
    # Influence function incorporates estimation of the covariance and both marginal variances.
    infl=w[:,None]*(x*y-.5*r*(x*x+y*y))
    if mode.startswith("weighted"):
        by_psu=pd.DataFrame(infl).groupby([d.strata.to_numpy(),d.psu.to_numpy()]).sum()
        # Center the total score per PSU within each masked stratum, finite-PSU correction.
        nh=by_psu.groupby(level=0).transform("size")
        if nh.min()<2:
            raise ValueError("Single-PSU analytic stratum")
        infl=(by_psu-by_psu.groupby(level=0).transform("mean")).multiply(np.sqrt(nh/(nh-1)),axis=0).to_numpy()
    se=np.sqrt((infl**2).sum(axis=0))/len(d)
    rng=np.random.default_rng(SEED)
    # Rademacher multiplier null distribution for paired correlation differences.
    max_p={}
    for group in matches:
        i=PANELS.index(group)
        contrasts=[(i,j,sgn) for j in range(5) if PANELS[j] not in matches for sgn in (1,-1)]
        D=np.column_stack([infl[:,i]-sgn*infl[:,j] for i,j,sgn in contrasts])
        delta=np.array([r[i]-sgn*r[j] for i,j,sgn in contrasts])
        sim=np.empty((2000,len(contrasts)))
        for k in range(0,2000,100):
            signs=rng.integers(0,2,(100,len(infl)),dtype=np.int8)*2-1
            sim[k:k+100]=signs@D/len(d)
        p=((sim>=delta).sum(axis=0)+1)/2001
        max_p[group]=[float(delta.min()),float(p.max())]
    rows=[]
    for j,panel in enumerate(PANELS):
        rows.append(dict(marker=marker,tier=tier,analysis=mode,panel=panel,matched=panel in matches,
                         n=len(d),r=r[j],lower=max(-1,r[j]-1.96*se[j]),upper=min(1,r[j]+1.96*se[j]),
                         p=2*norm.sf(abs(r[j])/se[j]),match_margin=max_p.get(panel,[np.nan,np.nan])[0],
                         p_diagonal=max_p.get(panel,[np.nan,np.nan])[1]))
    return rows


def other_dnam(d):
    from scipy.stats import t as td
    age=np.column_stack([np.ones(len(d)),d.age])
    xs=d[CLOCKS+["full_accel"]+ACC].copy(); xs["HorvathTelo"]*=-1
    cov=np.ones(len(d))
    residual=regress_residuals(xs.to_numpy(),age,cov)
    pd.DataFrame(residual,columns=xs.columns).corr().to_csv(HERE/"dnam_acceleration_correlation.csv")
    rows=[]
    for mode in ["primary","cells","weighted","weighted_cells"]:
        cols=["age","male"]+ACC+(CELLS if "cells" in mode else [])
        e=d.dropna(subset=cols+CLOCKS).copy()
        subset=e[cols].copy()
        for name in ACC:
            subset[name]=(subset[name]-subset[name].mean())/subset[name].std()
        z=np.column_stack([np.ones(len(e)),subset])
        w=e.WTDN4YR.to_numpy() if mode.startswith("weighted") else np.ones(len(e))
        w=w/w.mean()
        for clock in CLOCKS:
            y=e[clock].to_numpy()*(-1 if clock=="HorvathTelo" else 1)
            y=(y-y.mean())/y.std(ddof=1)
            A=np.linalg.inv(z.T@(w[:,None]*z))
            beta=A@(z.T@(w*y))
            err=y-z@beta
            scores=(w*err)[:,None]*z
            if mode.startswith("weighted"):
                psu=pd.DataFrame(scores).groupby([e.strata.to_numpy(),e.psu.to_numpy()]).sum()
                nh=psu.groupby(level=0).transform("size")
                scores=(psu-psu.groupby(level=0).transform("mean")).multiply(np.sqrt(nh/(nh-1)),axis=0).to_numpy()
            se=np.sqrt(np.diag(A@(scores.T@scores)@A))
            for panel,col in zip(PANELS,ACC):
                j=1+cols.index(col)
                rows.append(dict(analysis=mode,clock=clock,panel=panel,n=len(e),beta=beta[j],lower=beta[j]-1.96*se[j],
                                 upper=beta[j]+1.96*se[j],p=2*norm.sf(abs(beta[j])/se[j])))
    out=pd.DataFrame(rows); out["q"]=out.groupby("analysis").p.transform(lambda p:multipletests(p,method="fdr_bh")[1])
    out.to_csv(HERE/"dnam_clock_on_five_organs.csv",index=False)


def main():
    dnam,frames=sources()
    rows=[]
    for d,marker,match,tier,weight,transform in frames:
        modes=["primary","weighted"]
        if tier=="dnam_proxy": modes+=["cells","weighted_cells"]
        for mode in modes:
            rows+=matrix(d,marker,match,tier,weight,transform,mode)
    tab=pd.DataFrame(rows)
    # Two families, matching sequence-derived proxies versus independently measured laboratory traits.
    tab["family"]=np.where(tab.tier.eq("dnam_proxy"),"methylation_proxies","measured_references")
    for _,sub in tab.groupby(["family","analysis"]):
        tab.loc[sub.index,"q_matrix_BH"]=multipletests(sub.p,method="fdr_bh")[1]
        diag=sub[sub.matched]
        tab.loc[diag.index,"q_diagonal_BH"]=multipletests(diag.p_diagonal,method="fdr_bh")[1]
    tab.to_csv(HERE/"organ_reference_matrix.csv",index=False)
    other_dnam(dnam)
    print(tab[tab.analysis.eq("primary")&tab.matched][["marker","panel","n","r","match_margin","q_diagonal_BH"]].to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(limits=2): main()

