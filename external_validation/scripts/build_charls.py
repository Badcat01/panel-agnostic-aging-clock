"""Build participant-level CHARLS validation cohorts from baseline blood and follow-ups."""

import hashlib
import json
from pathlib import Path

from study_paths import COMMUNITY_ROOT, community_dir

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
WORKSPACE = COMMUNITY_ROOT
RAW = community_dir('CHARLS') / 'Raw_data'
HARM = RAW / 'Harmonized_CHARLS_D' / 'H_CHARLS_D_Data.dta'
BLOOD = RAW / '2011charls' / 'Blood_20140429.dta'
EXIT20 = RAW / '2020charls' / 'Exit_Module.dta'
LABS = {
    'bun': ('newbun','mg/dL'), 'creatinine': ('newcrea','mg/dL'),
    'uric_acid': ('newua','mg/dL'), 'glucose': ('newglu','mg/dL'),
    'hba1c': ('newhba1c','%'), 'cholesterol': ('newcho','mg/dL'),
    'triglycerides': ('newtg','mg/dL'), 'hdl': ('newhdl','mg/dL'),
    'crp': ('newcrp','mg/L / 10 = mg/dL'), 'wbc': ('qc1_vb002','10^9/L'),
    'hgb': ('qc1_vb004','g/dL'), 'mcv': ('qc1_vb006','fL'),
    'platelets': ('qc1_vb009','10^9/L'),
}
EXAM = {'bmi': ('r1mbmi','kg/m2'), 'waist': ('r1mwaist','cm'),
        'sbp': ('r1systo1/2/3','mmHg'), 'dbp': ('r1diasto1/2/3','mmHg'),
        'pulse': ('r1pulse1/2/3','beats/min')}
FILES = []


def read_stata(path, columns):
    FILES.append(path)
    data = pd.read_stata(path, columns=columns, convert_categoricals=False)
    assert data.ID.notna().all() and data.ID.is_unique
    return data.set_index('ID')


def month_id(year, month):
    year = pd.to_numeric(year, errors='coerce')
    month = pd.to_numeric(month, errors='coerce')
    return (12 * year + month).where(year.between(2000, 2020) & month.between(1, 12))


def main():
    blood = read_stata(BLOOD, ['ID'] + [x[0] for x in LABS.values()])
    vcols = [f'r1{v}{i}' for v in ['systo','diasto','pulse'] for i in [1,2,3]]
    harm = read_stata(HARM,
        ['ID','ID_w1','r1agey','ragender','r1iwstat','r1iwy','r1iwm','r1mbmi','r1mwaist',
         'r2iwstat','r2iwy','r2iwm','radyear','radmonth'] + vcols)
    baseline_ids = harm[harm.ID_w1.ne('')].reset_index().set_index('ID_w1',verify_integrity=True)
    assert blood.index.isin(baseline_ids.index).all()
    joined = blood.join(baseline_ids, how='left', validate='one_to_one').set_index('ID',verify_integrity=True)
    assert joined.r1iwstat.eq(1).all()
    # Restrict to actually interviewed wave-1 blood donors, with an adult age and a dated interview.
    base = month_id(joined.r1iwy, joined.r1iwm)
    complete = joined.r1agey.ge(20) & base.notna() & joined[[x[0] for x in LABS.values()]].notna().any(axis=1)
    data = joined.loc[complete].copy()
    base = base.loc[complete].astype(int)
    data['age'] = data.r1agey
    data['sex'] = data.ragender.map({1:1,2:0})
    for feature, (src, _) in LABS.items():
        data[feature] = data[src].astype(float)
    data['crp'] /= 10.0
    qc = {}
    for feature, src, lo, hi in [('waist','r1mwaist',35,200),('bmi','r1mbmi',8,100)]:
        ok = data[src].between(lo,hi)
        qc[feature] = {'valid_range':[lo,hi],'outside_set_missing':int((data[src].notna() & ~ok).sum())}
        data[feature] = data[src].where(ok)
    for feature, stub in [('sbp','systo'),('dbp','diasto'),('pulse','pulse')]:
        readings = data[[f'r1{stub}{i}' for i in [1,2,3]]]
        data[feature] = readings.where(readings>0).mean(axis=1)
    wave2_date = month_id(harm.r2iwy, harm.r2iwm)
    # Respondent-level exit interview month is not released in wave 2. For
    # an undated wave-2 death use the latest observed wave-2 survey month as upper bound.
    wave2_end = int(wave2_date.max())
    records = {2: pd.DataFrame({
        'alive':wave2_date.where(harm.r2iwstat.eq(1)),
        'dead':pd.Series(float(wave2_end),index=harm.index).where(harm.r2iwstat.eq(5)),
        'exact':month_id(harm.radyear,harm.radmonth).where(harm.r2iwstat.eq(5)),
        'death_year_only':harm.radyear.where(harm.r2iwstat.eq(5) & harm.radmonth.isna())
    }).reindex(data.index)}
    for wave, year in [(3,2015),(4,2018),(5,2020)]:
        sample = read_stata(RAW / f'{year}charls' / 'Sample_Infor.dta', ['ID','died','iyear','imonth'])
        interviewed = read_stata(RAW / f'{year}charls' / 'Demographic_Background.dta', ['ID']).index
        assert sample.index[sample.died.eq(1)].intersection(interviewed).empty
        date = month_id(sample.iyear,sample.imonth)
        assert date.notna().all()
        rec = pd.DataFrame({'alive':date.where(sample.died.eq(0) & sample.index.isin(interviewed)),
                            'dead':date.where(sample.died.eq(1)), 'exact':np.nan,'death_year_only':np.nan})
        if wave == 5:
            e = read_stata(EXIT20, ['ID','exb001_1','exb001_2','exb002'])
            # exb002=1 is Gregorian. Lunar dates remain interval-censored.
            rec['exact'] = month_id(e.exb001_1,e.exb001_2).where(e.exb002.eq(1)).reindex(rec.index)
            assert rec.loc[rec.exact.notna(),'dead'].notna().all()
        records[wave] = rec.reindex(data.index)

    counts = {}
    def construct(last_wave):
        follow = list(range(2,last_wave+1))
        statuses = {'no_postbaseline_contact':0,'chronology_conflict':0,'no_positive_followup':0}
        results = []
        # Bounds are in calendar months; all comparisons are blind to model scores.
        alive = np.column_stack([base, *[records[w].alive for w in follow]])
        dead = np.column_stack([records[w].dead for w in follow])
        exact = np.column_stack([records[w].exact for w in follow])
        death_year_only = records[2].death_year_only.to_numpy()
        for i, ident in enumerate(data.index):
            start = alive[i,0]
            death = np.flatnonzero(np.isfinite(dead[i]))
            if len(death):
                j = death[0]
                lower, upper = np.nanmax(alive[i,:j+1]), dead[i,j]
                if np.isfinite(alive[i,j+1:]).any() or upper <= start or lower > upper:
                    statuses['chronology_conflict'] += 1
                    continue
                date = exact[i,j]
                source = 'alive_to_first_death_report_midpoint'
                if np.isfinite(date):
                    if date < lower or date > upper:
                        statuses['chronology_conflict'] += 1
                        continue
                    lower = upper = date
                    source = 'reported_death_month'
                elif j == 0 and np.isfinite(death_year_only[i]):
                    lower = max(lower, death_year_only[i]*12+1)
                    upper = min(upper, death_year_only[i]*12+12)
                    if upper < lower:
                        statuses['chronology_conflict'] += 1
                        continue
                    source = 'reported_death_year_bounded_midpoint'
                end = (lower+upper)/2.0
                event = 1
            else:
                if np.isnan(alive[i,1:]).all():
                    statuses['no_postbaseline_contact'] += 1
                    continue
                end = lower = upper = np.nanmax(alive[i,1:])
                source = 'last_dated_living_interview'
                event = 0
            if end <= start:
                statuses['no_positive_followup'] += 1
                continue
            results.append([ident,(end-start)/12,event,(lower-start)/12,(upper-start)/12,source])
        endpoints = pd.DataFrame(results,columns=['ID','time_yr','event','time_lower_yr','time_upper_yr','time_source']).set_index('ID')
        analysis = data[['age','sex',*LABS,*EXAM]].join(endpoints,how='inner',validate='one_to_one')
        assert np.isfinite(analysis.time_yr).all() and analysis.time_yr.gt(0).all()
        assert (analysis.time_lower_yr<=analysis.time_yr).all() and (analysis.time_yr<=analysis.time_upper_yr).all()
        assert len(data)==len(analysis)+sum(statuses.values())
        obs = analysis[[*LABS,*EXAM]].notna().sum(axis=1)
        counts[str(last_wave)] = {'n':len(analysis),'deaths':int(analysis.event.sum()),'exclusions':statuses,
            'sex_missing':int(analysis.sex.isna().sum()),'median_observed_markers':float(obs.median()),
            'feature_count_range':[int(obs.min()),int(obs.max())], 'no_liver_markers':True,
            'time_source_counts':{str(k):int(v) for k,v in analysis.time_source.value_counts().items()},
            'n_by_feature':{k:int(analysis[k].notna().sum()) for k in [*LABS,*EXAM]}}
        return analysis

    for w, directory, name in [(5,'inputs','charls'),(2,'sensitivity_inputs','charls_short')]:
        result = construct(w)
        out = OUT / directory / (name+'.csv.gz')
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp')
        result.to_csv(tmp,compression={'method':'gzip','mtime':0},index_label='ID')
        tmp.replace(out)
        print(name,counts[str(w)],flush=True)
    metadata = {
        'cohort':'CHARLS','baseline':'2011-2012 national baseline blood donors; wave-1 measured features only',
        'n_source_blood':len(blood),'n_id_linked':len(joined),
        'baseline_n_after_age_date_blood':len(data),'excluded_baseline':int((~complete).sum()),
        'n_missing_age':int(joined.r1agey.isna().sum()), 'n_under20':int(joined.r1agey.lt(20).sum()),
        'n_missing_baseline_date':int(month_id(joined.r1iwy,joined.r1iwm).isna().sum()),
        'n_no_mapped_blood_value':int((~joined[[x[0] for x in LABS.values()]].notna().any(axis=1)).sum()),
        'id_linkage':'Exact match old 11-character blood ID to official Harmonized CHARLS D ID_w1; 12-character ID for all follow-up joins.',
        'sex':'ragender 1 male, 2 female mapped to 1/0',
        'source_inputs':{str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(FILES))},
        'features':{k:{'source':src,'unit':unit} for k,(src,unit) in {**LABS,**EXAM}.items()},
        'quality_masks':qc,'endpoint_counts_by_last_wave':counts,
        'event':'All-cause death. Wave 2 r2iwstat=5, later Sample_Infor died=1.',
        'censoring':'Last personal or proxy survey of participant alive using a dated wave 2 interview, or dated living Sample_Infor intersected with Demographic_Background thereafter. A missing wave alone is never death.',
        'time_origin':'Respondent-specific wave-1 interview month as proxy for unreported blood collection date; blood was collected after county interviews, so dates are approximate and early follow-up is subject to immortal time.',
        'timing':'For 2013 use harmonized radyear/radmonth (lunar conversion done by source). For 2020 use reported Gregorian month when available. Otherwise use midpoint from last living interview to first death report; bounds retained. For wave 2 lacking month use wave-end upper bound and, if available, intersect death year. Contradictory survival histories and nonpositive follow-up excluded.',
        'wave2_end_month_id':wave2_end,'sensitivity':'Uniform follow-up through wave 2 only. Coincident deaths/censors at monthly resolution may tie.',
        'limits':['Baseline plasma chemistry, predominantly fasting, differs from NHANES sampling; CRP unit converted with no platform bridging. No liver panel.','Blood donors with ascertainable longitudinal time are a selected group. Intermittent or permanent loss to follow-up is possible.','No sampling weights used in cohort assembly.'],
        'documentation':['https://charls.pku.edu.cn/en/doc/Blood2011.pdf','https://charls.charlsdata.com/Public/ashelf/public/uploads/document/2011-charls-wave1/application/CHARLS_nationalbaseline_users_guide.pdf',str((HARM.parent/'Harmonized_CHARLS_D.pdf').relative_to(WORKSPACE))]
    }
    dest = OUT/'audit'/'charls.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
