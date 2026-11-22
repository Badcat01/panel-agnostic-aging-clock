"""Transport development-anchored absolute mortality risks for the fitted LOCO MLPs."""
import hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from loco_survival import EXT,OUT,TRAIN_RUN,SurvivalMLP,load_data,rank,TARGETS
from reference import BUNDLE,REFERENCE,get_cohorts,score_frame
from validate_external import censor_probabilities

def cumulative_hazard(t,e,lp,h):
    idx=np.argsort(t)
    at_risk=np.cumsum(np.exp(lp[idx])[::-1])[::-1]
    tt,nn=np.unique(t[(e==1)&(t<=h)],return_counts=True)
    return float(np.sum(nn/at_risk[np.searchsorted(t[idx],tt)]))

def load_model(name,d):
    m=SurvivalMLP(d.x.shape[1]);path=next(OUT.glob(f'mlp_{name}_*.pt'))
    m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
    m.eval();return m

def main():
    ds,bundle=load_data()
    d0=ds['development'];single=load_model('development_only',d0)
    dev,_,_=get_cohorts()
    fr=score_frame(dev,BUNDLE,'cpu')
    lp1=dev[['SEQN','permth_exm']].merge(fr[['SEQN','full_lp']],on='SEQN',validate='one_to_one').sort_values('permth_exm',ascending=False,kind='stable').full_lp.to_numpy()
    ref=json.loads(REFERENCE.read_text(encoding='utf8'))
    slope,center=ref['risk_lp_slope'],ref['risk_lp_mean']
    lp1=(lp1-center)*slope
    t0=d0.t;e0=d0.e.numpy()
    source_single=rank(single,d0);rows=[]
    assert np.isclose(cumulative_hazard(t0,e0,lp1,3),ref['baseline_hazard']['clock']['3'],rtol=1e-5)
    for held in [*TARGETS,'temporal']:
        d=ds[held];multi=load_model('exclude_'+held,d0)
        source_multi=rank(multi,d0); models={'v1':(lp1,(d.v1-center)*slope),'single_mlp':(source_single,rank(single,d)),'multi_mlp':(source_multi,rank(multi,d))}
        t=d.t;e=d.e.numpy().astype(int); km=KaplanMeierFitter().fit(t,e)
        lev,after,before=censor_probabilities(t,e)
        for h in ([1] if held=='mimic_iv' else [3,5,10]):
            cases=(t<=h)&(e==1); alive=(t>h)
            if held=='mimic_iv':
                alive=(e==0)&np.isclose(t,1.)
                assert (cases|alive).all()
                w=np.ones(len(e))
            else:
                previous=np.searchsorted(lev,h,side='right')-1
                survival=after[previous] if previous>=0 else 1
                if min(cases.sum(),alive.sum())<20 or survival<=.05:continue
                w=np.zeros(len(e)); w[cases]=1/before[cases];w[alive]=1/survival
            for name,(source,lp) in models.items():
                # The learned log-hazard has coefficient 1 in the fitted Cox model.
                baseline=cumulative_hazard(t0,e0,source,h)
                risk=-np.expm1(-baseline*np.exp(lp))
                observed=1-km.predict(h); pred=risk.mean()
                rows.append(dict(target=held,model=name,years=h,n=len(e),cases=cases.sum(),
                              observable_controls=alive.sum(),predicted_mean=pred,observed_KM=observed,E_over_O=pred/observed,
                              ipcw_brier=np.sum(w*(cases-risk)**2)/len(e)))
    pd.DataFrame(rows).to_csv(OUT/'absolute_risk_transport.csv',index=False)
if __name__=='__main__':
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    with threadpool_limits(limits=3):main()

