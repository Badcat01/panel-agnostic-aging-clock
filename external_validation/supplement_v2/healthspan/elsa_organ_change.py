"""Directionally specified ELSA organ/marker change comparisons on identical rosters."""
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
import torch
from prepare_healthspan import OUT,EXT,DATA,META,ORGANS,read
from repeat_biomarkers import elsa,scores,diagonal,verify,repeat

def main(late=False,pooled=False):
    bundle=verify()
    panels=json.loads((EXT/'nhanes3_organ'/'organ_score_reference.json').read_text(encoding='utf8'))['panels']
    harmonic=read(next(DATA.rglob('h_elsa_g3.dta')),['idauniq','ragender'])
    harmonic.index=harmonic.idauniq.astype('int64').astype(str)
    draws={w:elsa(w) for w in ((2,4,6,8,9) if late else (2,4,6))}
    if pooled:
        late_data=pd.concat([draws[8],draws[9]],axis=0).sort_values('origin',kind='stable')
        assert int(late_data.index.duplicated().sum())==2
        draws[89]=late_data[~late_data.index.duplicated(keep='first')]
    labs='cholesterol hdl triglycerides glucose hba1c hgb crp wbc mch'.split()
    bases={2:pd.read_csv(OUT/'elsa_baseline.csv.gz',dtype={'ID':str}).set_index('ID')}
    for w in ([4,6] if late else [4]):
        a=draws[w];a=a[a[labs].notna().any(axis=1)&a.age.ge(20)&a.origin.notna()]
        score=scores(a,bundle['features'])
        roster=pd.DataFrame({'age':a.age,'sex':harmonic.ragender.map({1:1,2:0}).reindex(a.index),
                             'full_z':score.full_accel/panels['full']['accel_sd_years']})
        for organ in ORGANS:roster[organ]=score[organ+'_accel']/panels[organ]['accel_sd_years']
        bases[w]=roster
    rows=[]
    marked={'fibrinogen':'immune_heme','ferritin':'immune_heme','hba1c':'metabolic','sbp':'cardiovascular','dbp':'cardiovascular'}
    audits=[];stability=[]
    late_occasions=(89,) if pooled else (8,9)
    pairs=[(first,last) for first in (2,4,6) for last in late_occasions] if late else [(2,4),(2,6),(4,6)]
    for first,last in pairs:
        a=draws[first];b=draws[last]
        if late:
            r,s=repeat('elsa',last,a.loc[bases[first].index],b,bundle['features'],frozen_baseline=first==2)
            if pooled:s.to_csv(OUT/f'elsa_pooled_w89_pace_from_w{first}.csv.gz',index_label='ID',compression={'method':'gzip','mtime':0})
            for item in r:item['baseline_wave']=first
            stability.extend(r)
        elapsed=((b.origin-a.origin)/pd.Timedelta(days=365.25)).rename('years')
        for mark,match in marked.items():
            cov=pd.concat([elapsed.where(elapsed.gt(0)),a[mark].rename('baseline_marker')],axis=1)
            result=diagonal(bases[first],b[mark],match,'elsa',mark,cov,directional=mark!='ferritin')
            for item in result:
                item.update(first=first,last=last,direct_model_input=mark in ('sbp','dbp','hba1c'))
                se=(item['hi']-item['lo'])/3.92
                item['two_sided_p']=2*norm.sf(abs(item['rho']/se))
            rows.extend(result)
            organ_names=[s for s in ORGANS if bases[first][s].notna().any()]
            m=bases[first][['age','sex','full_z',*organ_names]].join(cov).join(b[mark].rename('follow')).dropna()
            assert len(m)==result[0]['n']
            audits.append(dict(first=first,last=last,marker=mark,n=len(m),years_median=m.years.median(),
                               baseline_median=m.baseline_marker.median(),followup_median=m.follow.median(),
                               paired_difference_median=(m.follow-m.baseline_marker).median()))
    matrix=pd.DataFrame(rows)
    family=matrix[matrix.match&matrix.marker.ne('ferritin')][['marker','first','last','matched_stronger_than_all_p']].copy()
    assert len(family)==len(pairs)*4
    family['diagonal_q']=multipletests(family.matched_stronger_than_all_p,method='fdr_bh')[1]
    matrix=matrix.merge(family.drop(columns='matched_stronger_than_all_p'),on=['first','last','marker'],how='left',validate='many_to_one')
    ferr=matrix.marker.eq('ferritin')
    matrix.loc[ferr,'ferritin_2s_q']=multipletests(matrix.loc[ferr,'two_sided_p'],method='fdr_bh')[1]
    prefix='elsa_pooled_w89' if pooled else ('elsa_late_nurse' if late else 'elsa_organ_change')
    matrix.to_csv(OUT/(prefix+'_matrix.csv'),index=False)
    pd.DataFrame(audits).to_csv(OUT/(prefix+'_levels.csv'),index=False)
    if late:
        pd.DataFrame(stability).to_csv(OUT/(prefix+'_repeat.csv'),index=False)
    (OUT/(prefix+'_sources.json')).write_text(json.dumps(META,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
    with threadpool_limits(limits=3):main(late='--late' in sys.argv or '--pooled' in sys.argv,pooled='--pooled' in sys.argv)
