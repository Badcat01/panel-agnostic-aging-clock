"""Check the held-out ranks using an independent concordance implementation."""
import json
import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from loco_survival import OUT,TARGETS,load_data,rank
from transport_risk import load_model
def main():
    ds,_=load_data()
    table=pd.read_csv(OUT/'leave_one_cohort_out.csv').set_index('target')
    spec=json.loads((OUT/'analysis_spec.json').read_text(encoding='utf8'))
    assert set(spec['targets'])==set(TARGETS)
    assert 'mhas' not in set(ds) and all('mhas' not in source.lower() for source in spec['sources'])
    checked=[]
    assert set(table.index)==set([*TARGETS,'temporal'])
    for k,r in table.iterrows():
        d=ds[k]
        assert k not in r.fit_cohorts.split(',') and 'temporal' not in r.fit_cohorts.split(',')
        assert set(r.fit_cohorts.split(','))==({'development',*TARGETS}-{k})
        assert d.n==r.n and d.e.sum().item()==r.events
        for key,vals in [('v1',d.v1),('single',rank(load_model('development_only',ds['development']),d)),('multi',rank(load_model('exclude_'+k,ds['development']),d))]:
            val=concordance_index(d.t,-vals,d.e.numpy())
            np.testing.assert_allclose(val,r[key+'_C'],rtol=0,atol=2e-7)
        checked.append(k)
    risk=pd.read_csv(OUT/'absolute_risk_transport.csv')
    assert risk[['observed_KM','predicted_mean','ipcw_brier']].apply(lambda x:x.between(0,1)).all().all()
    assert np.isclose(risk.query("target=='hrs' and years==3 and model=='v1'").predicted_mean.item(),.1513,atol=5e-5)
    (OUT/'verification.json').write_text(json.dumps(dict(independent_rank_checks=checked,held_out_disjoint=True,fresh_seven_cohort_scope=True,
                 exact_HRS_v1_3y_reproduced_to_4_decimals=True),indent=2),encoding='utf8')
if __name__=='__main__':
    torch.set_num_threads(3)
    main()

