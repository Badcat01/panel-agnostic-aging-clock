"""HRS measured reference proteins and ancestry-specific polygenic scores."""
import json
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from shared import HERE,EXT,formal,sha

PANELS=["liver","kidney","metabolic","immune_heme","cardiovascular"]
from study_paths import community_dir

ROOT = community_dir("HRS")
SOURCES=[]

def source(name):
    f=next(ROOT.rglob(name)); SOURCES.append({"file":name,"sha256":sha(f)})
    x=(pyreadstat.read_sav(f) if f.suffix==".sav" else pyreadstat.read_dta(f))[0]
    x.columns=x.columns.str.upper()
    return x.set_index(x.HHID.str.strip()+"-"+x.PN.str.strip(),verify_integrity=True)

def residual(x,controls):
    return x-controls@np.linalg.lstsq(controls,x,rcond=None)[0]

def matrix(d,target,label,matched,group,controls,panels,sign=1):
    eligible=[p for p in panels if d[p+"_accel"].notna().any()]
    features=[p+"_accel" for p in eligible]
    f=d[[target,*controls,*features]].dropna()
    if len(f)<40:return []
    X=np.column_stack([np.ones(len(f)),f[controls]])
    r=residual(f[[target]+features].to_numpy(),X)
    r=r/r.std(axis=0)
    y=r[:,0]; z=r[:,1:]; rho=np.mean(y[:,None]*z,axis=0)
    influence=y[:,None]*z - .5*rho*(y[:,None]**2+z**2)
    clusters=pd.DataFrame(influence,index=f.index.str.split("-").str[0]).groupby(level=0).sum().to_numpy()/len(f)
    se=np.sqrt((clusters**2).sum(axis=0)*len(clusters)/(len(clusters)-1))
    result=[]
    for p,rr,ss in zip(eligible,rho,se):
        result.append(dict(group=group,target=label,expected=matched,panel=p,n=len(f),households=len(clusters),
                           partial_r=rr,ci95_low=rr-1.96*ss,ci95_high=rr+1.96*ss,p=2*norm.sf(abs(rr/ss))))
    if matched in eligible and len(eligible)>1:
        i=eligible.index(matched)
        # Is the matched, correctly directed correlation stronger than every
        # alternative in absolute magnitude? Infer all signed contrasts jointly
        # by intersection-union. Household scores receive shared wild signs.
        draws=np.random.default_rng(20261121)
        estimates=np.array([sign*rho[i]-s*rho[j] for j in range(len(rho)) if j!=i for s in (1,-1)])
        effects=np.column_stack([sign*clusters[:,i]-s*clusters[:,j] for j in range(len(rho)) if j!=i for s in (1,-1)])
        count=np.ones(len(estimates))
        for _ in range(20):
            null=draws.choice([-1,1],size=(100,len(clusters)))@effects
            count+=(null>=estimates).sum(axis=0)
        for row in result:
            row["min_matched_advantage"]=estimates.min()
            row["dominance_p_wild_IUT"]=max(count/2001)
    return result

def main():
    _,scores=formal("hrs")
    scores["age_c"]=(scores.age-65)/10;scores["age_sq"]=scores.age_c**2
    v=source("hrs2016vbs.sav")[["PCYSC","PCMVGE","PNTBNPE"]]
    v=v.join(source("vbs16aa.dta")[["PIL6","PIL1RA","PTNFR1"]],how="outer",validate="one_to_one")
    d=scores.join(np.log(v.where(v>0)),validate="one_to_one")
    basic=["age_c","age_sq","sex"]
    assays=[("PCYSC","kidney"),("PNTBNPE","cardiovascular"),("PCMVGE","immune_heme"),
            ("PIL6","immune_heme"),("PIL1RA","immune_heme"),("PTNFR1","immune_heme")]
    rows=[]
    for col,matched in assays:
        rows+=matrix(d,col,col,matched,"VBS_clinical",basic+["full_accel"],PANELS)
    for suffix,group in [("e","EA"),("a","AA")]:
        g=source(f"pgenscore4{suffix}_r.dta");pcs=[f"PC{x}{letter}" for x in ["1_5","6_10"] for letter in "ABCDE"]
        data=scores.join(g[[c for c in g if c in pcs or c.startswith(suffix.upper()+"4_")]],how="inner",validate="one_to_one")
        for col,p,sign in [("T2D_DIAGRAM12","metabolic",1),("CKD_CKDGEN19","kidney",1),
                           ("EGFR_CKDGEN19","kidney",-1),("CD_CARDIOGRAM11","cardiovascular",1),
                           ("LONG_CHARGE15","full",-1)]:
            field=suffix.upper()+"4_"+col
            targets=PANELS+["full"] if p=="full" else PANELS
            rows+=matrix(data,field,col,p,group,basic+pcs+([] if p=="full" else ["full_accel"]),targets,sign)
    out=pd.DataFrame(rows);out["matched"]=out.panel.eq(out.expected)
    for group,ix in out.groupby("group").groups.items():
        out.loc[ix,"association_q_BH"]=multipletests(out.loc[ix,"p"],method="fdr_bh")[1]
        keep=out.index.isin(ix)&out.matched&out.dominance_p_wild_IUT.notna()
        if keep.any():out.loc[keep,"dominance_q_BH"]=multipletests(out.loc[keep,"dominance_p_wild_IUT"],method="fdr_bh")[1]
    out.to_csv(HERE/"hrs_reference_pgs.csv",index=False)
    (HERE/"hrs_reference_source_hashes.json").write_text(json.dumps(SOURCES,indent=2),encoding="utf8")
    print(out[out.matched][["group","target","n","partial_r","dominance_q_BH"]].to_string(index=False))

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
