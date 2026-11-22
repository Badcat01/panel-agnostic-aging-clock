"""Recalculate selected partial correlations and verify the ELSA result rosters."""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from prepare_healthspan import OUT
from repeat_biomarkers import elsa

def main():
    checks=[]
    base=pd.read_csv(OUT/'elsa_baseline.csv.gz',dtype={'ID':str}).set_index('ID')
    early=elsa(2)
    late=pd.concat([elsa(8),elsa(9)]).sort_values('origin',kind='stable')
    late=late[~late.index.duplicated(keep='first')]
    for prefix,dataset,last in [('elsa_organ_change',elsa(4),4),('elsa_pooled_w89',late,89)]:
        dfs=pd.read_csv(OUT/(prefix+'_matrix.csv'))
        match=dfs[dfs.match&dfs.marker.ne('ferritin')]
        assert len(match)==12
        np.testing.assert_allclose(match.diagonal_q,multipletests(match.matched_stronger_than_all_p,method='fdr_bh')[1])
        groups=['metabolic','immune_heme','cardiovascular']
        for marker in ['fibrinogen','hba1c','sbp']:
            participants=base[['age','sex','full_z',*groups]].join(early[marker].rename('baseline')).join(dataset[marker].rename('follow'))
            participants['years']=(dataset.origin-early.origin)/pd.Timedelta(days=365.25)
            d=participants[participants.years.gt(0)].dropna()
            cov=np.c_[np.ones(len(d)),d.age,d.age**2,d.sex,d.full_z,rankdata(d.years),rankdata(d.baseline)]
            def resid(x):
                y=rankdata(x)
                return y-cov@np.linalg.lstsq(cov,y,rcond=None)[0]
            y=resid(d.follow)
            for group in groups:
                r=dfs[dfs['first'].eq(2)&dfs['last'].eq(last)&dfs.marker.eq(marker)&dfs.organ.eq(group)].iloc[0]
                assert r.n==len(d)
                rho=np.corrcoef(y,resid(d[group]))[0,1]
                np.testing.assert_allclose(rho,r.rho,atol=1e-10)
                checks.append(dict(wave_start=2,wave_end=last,marker=marker,group=group,n=len(d),rho=rho))
    aux=pd.read_csv(OUT/'elsa_observed_incidence.csv')
    assert len(aux)==39 and aux.n.ge(aux.events).all() and aux.observed_intervals.ge(aux.n).all()
    ten=pd.read_csv(OUT/'elsa_pooled_w89_to_w10.csv')
    assert len(ten)==8 and ten.n.le(3365).all()
    pace=pd.read_csv(OUT/'elsa_pooled_w89_pace_to_w10.csv')
    assert len(pace)==6 and pace.n.ge(pace.events).all()
    (OUT/'elsa_extension_checks.json').write_text(json.dumps(dict(partial_rho_recalculated_using_OLS=checks,
          observed_intervals_count_checked=len(aux),last_core_models_checked=len(ten),pace_models_checked=len(pace)),
          indent=2),encoding='utf8')
if __name__=='__main__':main()
