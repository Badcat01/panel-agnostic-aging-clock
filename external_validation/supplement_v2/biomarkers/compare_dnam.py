"""Paired NHANES DNAm comparisons; no new clock training and no sample reuse for Cox predictions."""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import StatisticalWarning
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from numba import njit

from score_holdout import HERE, CLOCKS, CELLS, SEED, load_json, write_json, sha
from validate_external import weighted_c
from lifelines.utils import concordance_index


def adjusted_scores(df, columns, weighted, age_center):
    extras = ["time_yr", "event"] + (["WTDN4YR"] if weighted else [])
    fit_input = df[columns+extras].copy()
    if weighted:
        fit_input.WTDN4YR /= fit_input.WTDN4YR.mean()
    fit = CoxPHFitter().fit(fit_input, "time_yr", "event", weights_col="WTDN4YR" if weighted else None,
                            robust=weighted)
    return fit


@njit
def uno10(t, e, ranks):
    # Time-sorted ascending, compare events <= 10y with strictly later observed follow-up.
    g = 1.
    num = den = 0.
    i = 0
    while i < len(t) and t[i] <= 10:
        j = i+1
        while j < len(t) and t[j] == t[i]:
            j += 1
        events = e[i:j].sum()
        if events and g > 0:
            for k in range(i, j):
                if e[k]:
                    num += ((ranks[k] > ranks[j:]).sum() + 0.5*(ranks[k] == ranks[j:]).sum())/g**2
                    den += (len(t)-j)/g**2
        g *= 1 - (j-i-events)/(len(t)-i)
        i = j
    return num/den


def run():
    path = HERE/"dnam_frozen_scores.csv.gz"
    assert sha(path)==load_json(HERE/"roster_audit.json")["output_sha256"]
    df = pd.read_csv(path)
    # All adjusted models retain actual age; age residualization and positive scaling leave rankings unchanged.
    for col in CLOCKS + ["full_accel"]:
        df[col] = (df[col]-df[col].mean())/df[col].std()
    df["HorvathTelo"] *= -1  # shorter telomere length = older biological direction
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df,df.event))
    mode_settings = [("primary",False,False),("cells",False,True),("weighted",True,False),("weighted_cells",True,True)]
    benefits, joint = [], []
    likelihood = {}
    for mode, weighted, cells in mode_settings:
        demographics = ["age","male"] + (CELLS if cells else [])
        specs = {"clinical": ["full_accel"], "age_sex": [],
                 **{c:[c] for c in CLOCKS}, "joint_GrimAge2":["full_accel","GrimAge2Mort"]}
        scores = pd.DataFrame(index=df.index)
        for label, terms in specs.items():
            covars = demographics+terms
            reference = df[covars].mean().to_numpy()
            values = np.full(len(df),np.nan)
            for training, test in folds:
                fit = adjusted_scores(df.iloc[training], covars, weighted, reference)
                values[test] = (df[covars].iloc[test].to_numpy()-reference) @ fit.params_.to_numpy()
            scores[label] = values
            if mode == "primary":
                model = adjusted_scores(df,covars,False,reference)
                likelihood[label]=model.log_likelihood_
                if terms:
                    for term in terms:
                        joint.append(dict(model=label,term=term,HR=np.exp(model.params_[term]),
                                          lower=np.exp(model.confidence_intervals_.loc[term].iloc[0]),
                                          upper=np.exp(model.confidence_intervals_.loc[term].iloc[1]),p=model.summary.loc[term,"p"]))
        assert np.isfinite(scores).all().all()
        order = np.argsort(-df.time_yr.to_numpy(),kind="stable")
        t = df.time_yr.to_numpy()[order]
        e = df.event.to_numpy(dtype=int)[order]
        ranks = [np.unique(scores[c].to_numpy()[order],return_inverse=True)[1]+1 for c in scores]
        w = (df.WTDN4YR/df.WTDN4YR.mean()).to_numpy()[order] if weighted else np.ones(len(t))
        estimates=np.array([weighted_c(t,e,r,w) for r in ranks])
        if not weighted:
            for i,c in enumerate(scores):
                assert np.isclose(estimates[i],concordance_index(t,-scores[c].to_numpy()[order],e))
        units = df.iloc[order][["strata","psu"]].reset_index(drop=True)
        unit_idx, unit_names = pd.factorize(pd.MultiIndex.from_frame(units))
        strata = [np.where(unit_names.get_level_values(0)==h)[0] for h in np.unique(units.strata)]
        assert min(map(len,strata))>=2
        rng=np.random.default_rng(SEED)
        sim=np.empty((2000,len(ranks)))
        for b in range(len(sim)):
            if weighted:
                rep=np.zeros(len(unit_names))
                for ids in strata:
                    counts=rng.multinomial(len(ids)-1,[1/len(ids)]*len(ids))
                    rep[ids]=counts*len(ids)/(len(ids)-1)
                rw=w*rep[unit_idx]
            else:
                rw=np.bincount(rng.integers(0,len(t),len(t)),minlength=len(t))
            sim[b]=[weighted_c(t,e,r,rw) for r in ranks]
        for j,label in enumerate(scores):
            lo,hi=np.quantile(sim[:,j],[.025,.975])
            dlo,dhi=np.quantile(sim[:,0]-sim[:,j],[.025,.975])
            benefits.append(dict(analysis=mode,model=label,n=len(df),deaths=int(df.event.sum()),C=estimates[j],
                                 C_lower=lo,C_upper=hi,clinical_minus_C=estimates[0]-estimates[j],
                                 delta_lower=dlo,delta_upper=dhi,
                                 uno_C_10y=uno10(t[::-1],e[::-1],ranks[j][::-1]) if not weighted else np.nan,
                                 bootstrap_reps=len(sim)))
        print(mode, "clinical/Grim2",estimates[0],estimates[list(scores).index("GrimAge2Mort")],flush=True)
    pair_lrt=[]
    for col in CLOCKS:
        if col=="GrimAge2Mort":
            ll=likelihood["joint_GrimAge2"]
        else:
            model=adjusted_scores(df,["age","male","full_accel",col],False,None)
            ll=model.log_likelihood_
        for reduced,added in [(col,"clinical"),("clinical",col)]:
            lr=2*(ll-likelihood[reduced])
            pair_lrt.append(dict(dnam=col,addition=added,LR=lr,p=chi2.sf(lr,1)))
    results=pd.DataFrame(pair_lrt); results["q_BH"]=multipletests(results.p,method="fdr_bh")[1]
    results.to_csv(HERE/"dnam_lrt.csv",index=False)
    pd.DataFrame(joint).to_csv(HERE/"dnam_hr.csv",index=False)
    pd.DataFrame(benefits).to_csv(HERE/"dnam_discrimination.csv",index=False)
    write_json(HERE/"dnam_methods.json",dict(n=len(df),events=int(df.event.sum()),folds=5,seed=SEED,
               holdout_fit="Cox outcome associations calibrated in DNAm-only training folds; same folds for all scores",
               uncertainty="2,000 paired bootstrap conditional on fitted out-of-fold scores, not refitting the Cox models",
               weighted="WTDN4YR normalized, weighted Cox; survey stratum-PSU rescaled bootstrap within NHANES masked strata",
               design_strata=len(strata),design_units=len(unit_names),clocks=CLOCKS,
               cells="Five of six estimated cell proportions to account for composition; omit NeuPP",
               telomere_direction="shorter telomere length oriented as higher risk in association and correlation outputs",
               uno_horizon_years=10,uno="Marginal censoring KM; compare strictly later follow-up within time horizon",
               source="https://wwwn.cdc.gov/nchs/nhanes/dnam/"))


if __name__ == "__main__":
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=StatisticalWarning)
        run()

