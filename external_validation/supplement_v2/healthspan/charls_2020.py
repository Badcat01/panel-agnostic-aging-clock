"""Extend CHARLS disability ascertainment through the 2020 interview, preserving the existing roster."""
import json
import pandas as pd
from prepare_healthspan import OUT,DATA,META,read,date
def build():
    base=pd.read_csv(OUT/'charls_baseline.csv.gz',dtype={'ID':str}).set_index('ID')
    history=pd.read_csv(OUT/'charls_visits.csv.gz',dtype={'ID':str},usecols=['ID','wave','time','adl','iadl'])
    variables=dict(adl=['db001','db003','db005','db007','db009'],iadl=['db022','db020','db016','db014'])
    file=next(DATA.rglob('2020charls/Health_Status_and_Functioning.dta'))
    raw=read(file,['ID',*variables['adl'],*variables['iadl']]).set_index('ID',verify_integrity=True)
    tracker=read(next(DATA.rglob('2020charls/Sample_Infor.dta')),['ID','died','iyear','imonth']).set_index('ID',verify_integrity=True)
    harm=read(next(DATA.rglob('H_CHARLS_D_Data.dta')),['ID','r1iwy','r1iwm']).set_index('ID',verify_integrity=True)
    dates=date(pd.to_numeric(tracker.iyear),pd.to_numeric(tracker.imonth)).where(tracker.died.eq(0))
    base_date=date(harm.r1iwy,harm.r1iwm)
    d=pd.DataFrame(index=base.index)
    d['wave']=5
    d['time']=(dates-base_date)/pd.Timedelta(days=365.25)
    for k,cols in variables.items():
        items=raw[cols].where(raw[cols].isin([1,2,3,4]))
        val=items.gt(1).any(axis=1).astype(float).where(items.gt(1).any(axis=1)|items.notna().all(axis=1))
        d[k]=val.reindex(d.index).where(d.time.notna())
    assert not ((d.adl.notna()|d.iadl.notna()) & d.time.le(0)).any()
    pd.concat([history,d.reset_index()]).to_csv(OUT/'charls_w5_visits.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    base['horizon']=(dates.max()+pd.offsets.MonthEnd(0)-base_date.reindex(base.index))/pd.Timedelta(days=365.25)
    base.to_csv(OUT/'charls_w5_baseline.csv.gz',compression={'method':'gzip','mtime':0})
    (OUT/'charls2020_sources.json').write_text(json.dumps(META,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':build()

