"""Compare HRS with NCHS sex- and exact-age-specific period mortality."""
import hashlib
import re
import numpy as np
import pandas as pd
import requests
from lifelines import KaplanMeierFitter
from shared import HERE,formal

def main():
    src=HERE/"nchs";src.mkdir(exist_ok=True)
    data={}; manifest=[]
    for year,folder in [(2016,"68_04"),(2017,"68_07"),(2018,"69-12"),(2019,"70-19")]:
        for male,table in [(1,"02"),(0,"03")]:
            url=f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/{folder}/Table{table}.xlsx"
            path=src/f"{year}_{table}.xlsx"
            if not path.exists():
                reply=requests.get(url,timeout=40);reply.raise_for_status()
                assert reply.content[:2]==b"PK"
                path.write_bytes(reply.content)
            f=pd.read_excel(path,header=None)
            title=f.iloc[:4,0].astype(str).str.cat(sep=" ").lower()
            assert ("males" if male else "females") in title and str(year) in title
            q={}
            for a,v in zip(f[0],f[1]):
                age=re.match(r"^(\d+)\s*[-–—]",str(a).strip())
                if age: q[int(age[1])]=float(v)
            assert all(k in q for k in range(100)) and all(0 < x < 1 for x in q.values())
            data[year,male]=q
            manifest.append(dict(year=year,sex=male,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(manifest).to_csv(src/"manifest.csv",index=False)
    raw,_=formal("hrs");raw=raw.dropna(subset=["sex","time_yr","event"])
    # Two overlapping three-calendar-year schedules bracket the 2016 VBS fieldwork.
    # Truncate to ages with finite 1-year q for all three attained ages.
    eligible=raw[raw.age.le(97)].copy()
    rows=[]
    for start in [2016,2017]:
        q=1-np.prod([[1-data[start+i,int(s)][int(a)+i] for a,s in zip(eligible.age,eligible.sex)] for i in range(3)],axis=0)
        for band,m in [("all",eligible.age.notna()),("<70",eligible.age<70),("70–79",eligible.age.between(70,79)),
                       ("80–84",eligible.age.between(80,84)),("85–97",eligible.age>=85)]:
            d=eligible[m]; obs=1-KaplanMeierFitter().fit(d.time_yr,d.event).predict(3)
            rows.append(dict(start=start,band=band,n=len(d),excluded_over_97=len(raw)-len(eligible),expected=q[m].mean(),observed_KM=obs))
    pd.DataFrame(rows).to_csv(HERE/"hrs_nchs_lifetable.csv",index=False)

if __name__=="__main__": main()
