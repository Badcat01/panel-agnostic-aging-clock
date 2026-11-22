"""Read the NHANES pilot table and apply source-verified corrections in memory."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


CDC = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public'
SOURCES = {
    'creatinine_1999': f'{CDC}/1999/DataFiles/LAB18.htm',
    'early_field_aliases': f'{CDC}/2001/DataFiles/L40_B.htm',
    'creatinine_2005': f'{CDC}/2005/DataFiles/BIOPRO_D.htm',
    'conventional_crp': f'{CDC}/2009/DataFiles/CRP_F.htm',
    'hscrp_2015': f'{CDC}/2015/DataFiles/HSCRP_I.htm',
    'hscrp_bridge': f'{CDC}/2017/DataFiles/HSCRP_J.htm',
    'dnam': 'https://wwwn.cdc.gov/nchs/nhanes/dnam/',
}

# For example, 5.397605346934028e-79 in pandas.read_sas(XPT) is a SAS
# numeric zero. This was cross-checked against ReadStat on BPX, CBC_I, BIOPRO_I.
XPT_ZERO = 16.0 ** -65


def xpt(raw, stem, participant_ids):
    records = pd.read_sas(raw / f'{stem}.xpt', format='xport', encoding='latin1')
    records = records.set_index('SEQN', verify_integrity=True)
    return records.reindex(participant_ids)


def require_match(values, source, name):
    if not np.allclose(values, source, atol=1e-10, rtol=1e-10, equal_nan=True):
        raise ValueError(f'Processed values or missingness disagree with raw XPT: {name}')


def prepare_table(project: Path) -> tuple[pd.DataFrame, set[int], dict]:
    project = Path(project)
    raw = project / 'data' / 'raw' / 'nhanes'
    processed = project / 'data' / 'processed'
    table = pd.read_csv(processed / 'nhanes_9918.csv.gz')
    if table.SEQN.isna().any() or table.SEQN.duplicated().any():
        raise ValueError('SEQN must uniquely identify NHANES participants')
    meta = {'sources': SOURCES.copy(), 'analytic_rows_before_panel_filter': len(table)}

    # Some 2001–02 variables have a different prefix because the published
    # values were already aligned by CDC. They retain the conventional units.
    early_checks = {}
    aliases = {
        'creatinine': ('LBXSCR', 'LBDSCR', 'mg/dL'),
        'bilirubin': ('LBXSTB', 'LBDSTB', 'mg/dL'),
        'phosphorus': ('LBXSPH', 'LBDSPH', 'mg/dL'),
        'alp': ('LBXSAPSI', 'LBDSAPSI', 'U/L'),
        'ldh': ('LBXSLDSI', 'LBDSLDSI', 'U/L'),
        'albumin': ('LBXSAL', 'LBXSAL', 'g/dL'),
        'glucose': ('LBXSGL', 'LBXSGL', 'mg/dL'),
    }
    for year, stem in [(1999, 'LAB18'), (2001, 'L40_B'), (2003, 'L40_C'), (2005, 'BIOPRO_D')]:
        rows = table.cycle.eq(year)
        source = xpt(raw, stem, table.loc[rows, 'SEQN'])
        early_checks[str(year)] = {}
        for name, (regular, alternate, unit) in aliases.items():
            column = alternate if year == 2001 else regular
            require_match(table.loc[rows, name], source[column], f'{stem}/{column}')
            early_checks[str(year)][name] = {'column': column, 'unit': unit, 'observed': int(source[column].count())}
    meta['raw_units_and_missingness_checks'] = early_checks

    corrections = {}
    for year, slope, offset in [(1999, 1.013, 0.147), (2005, 0.978, -0.016)]:
        rows = table.cycle.eq(year) & table.creatinine.notna()
        table.loc[rows, 'creatinine'] = slope * table.loc[rows, 'creatinine'] + offset
        corrections[str(year)] = {'unit': 'mg/dL', 'slope': slope, 'offset': offset, 'n': int(rows.sum())}
    meta['creatinine_calibration'] = corrections

    crp_stems = {1999: 'LAB11', 2001: 'L11_B', 2003: 'L11_C', 2005: 'CRP_D',
                 2007: 'CRP_E', 2009: 'CRP_F', 2015: 'HSCRP_I', 2017: 'HSCRP_J'}
    for year, stem in crp_stems.items():
        rows = table.cycle.eq(year)
        source = xpt(raw, stem, table.loc[rows, 'SEQN'])
        original = source.LBXHSCRP / 10.0 if year >= 2015 else source.LBXCRP
        require_match(table.loc[rows, 'crp'], original, stem)
        if year == 2015:
            eligible = source.LBXHSCRP.le(23)
            bridged = original.copy()
            bridged.loc[eligible] = (0.8695 * source.loc[eligible, 'LBXHSCRP'] + 0.2954) / 10.0
            table.loc[rows, 'crp'] = bridged.to_numpy()
            meta['crp_calibration'] = {
                'final_unit': 'mg/dL', 'reference_for_2015_and_2017': '2017-2018 Roche Cobas 6000',
                'formula_2015_hscrp_in_mg_L': '(0.8695*x + 0.2954) for x <= 23',
                'n_bridged': int(eligible.sum()), 'n_above_range_retained': int(source.LBXHSCRP.gt(23).sum()),
                'remaining_limit': 'No validated 1999-2010 to 2015-2018 assay bridge identified in the cited CDC codebooks. Older CRP is retained in mg/dL; complete cross-era equivalence is unverified.'}
    if table.loc[table.cycle.isin([2011, 2013]), 'crp'].notna().any():
        raise ValueError('Unexpected CRP measurements in 2011–2014')

    zero_repairs = {}
    for name in ['eos_pct', 'bilirubin']:
        encoded_zero = np.isclose(table[name], XPT_ZERO, rtol=1e-12, atol=0)
        table.loc[encoded_zero, name] = 0.0
        zero_repairs[name] = int(encoded_zero.sum())
    # Recompute from individual readings: the pilot explicitly excluded zero
    # diastolic readings, but its pandas XPT decoder left them nonzero.
    dbp_changed = 0
    dbp_to_missing = 0
    for year, suffix in zip(range(1999, 2019, 2), ['', '_B', '_C', '_D', '_E', '_F', '_G', '_H', '_I', '_J']):
        rows = table.cycle.eq(year)
        source = xpt(raw, f'BPX{suffix}', table.loc[rows, 'SEQN'])
        readings = source[[f'BPXDI{i}' for i in range(1, 5) if f'BPXDI{i}' in source]]
        nonzero = readings.mask(np.isclose(readings, XPT_ZERO, atol=0, rtol=1e-12) | readings.eq(0))
        corrected = nonzero.mean(axis=1).to_numpy()
        before = table.loc[rows, 'dbp'].to_numpy()
        dbp_changed += int((~np.isclose(before, corrected, equal_nan=True)).sum())
        dbp_to_missing += int((np.isfinite(before) & np.isnan(corrected)).sum())
        table.loc[rows, 'dbp'] = corrected
    meta['sas_zero_decoding'] = {**zero_repairs, 'dbp_mean_changed': dbp_changed,
        'dbp_all_zero_to_missing': dbp_to_missing, 'dbp_summary': 'Mean of all available nonzero readings per the project preprocessing script.'}

    dnam = pd.read_sas(raw / 'dnmepi.sas7bdat', encoding='latin1').set_index('SEQN', verify_integrity=True)
    has_marker = dnam.drop(columns='WTDN4YR').notna().any(axis=1)
    ids = set(dnam.index[has_marker].astype(int))
    exported = pd.read_csv(processed / 'dnmepi.csv.gz', usecols=['SEQN'])
    if ids != set(exported.SEQN) or exported.SEQN.duplicated().any() or not has_marker.equals(dnam.WTDN4YR.gt(0)):
        raise ValueError('DNAm holdout IDs differ across measured markers, survey weights, or processed IDs')
    meta['dnam'] = {'raw_participant_rows': len(dnam), 'any_biomarker_measured': len(ids),
        'no_measurement_zero_weight': int((~has_marker).sum()), 'verified_against_processed': True,
        'sorted_id_sha256': hashlib.sha256('\n'.join(map(str, sorted(ids))).encode()).hexdigest()}
    return table, ids, meta
