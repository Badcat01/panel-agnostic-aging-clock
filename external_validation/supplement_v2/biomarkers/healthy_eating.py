"""Two-day HEI-2015 from NCHS and USDA. Read the data archive without executing the distributor."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zlib

import numpy as np
import pandas as pd
import requests

from download_references import SOURCE, download

HERE=SOURCE.parent


def fped(year,day):
    period=f"{year%100:02d}{(year+1)%100:02d}"
    name=f"FPED_DR{day}TOT_{period}"
    sas=SOURCE/(name.lower()+".sas7bdat")
    url=f"https://www.ars.usda.gov/ARSUserFiles/80400530/apps/{name}_sas.exe"
    r=requests.get(url,timeout=60);r.raise_for_status()
    compressed=r.content
    magic=b"7z\xbc\xaf\x27\x1c"
    off=compressed.find(magic)
    while off>=0 and zlib.crc32(compressed[off+12:off+32])!=int.from_bytes(compressed[off+8:off+12],"little"):
        off=compressed.find(magic,off+1)
    if off<0: raise ValueError(f"Unrecognized USDA archive format {name}")
    tar=shutil.which("bsdtar") or shutil.which("tar")
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/"payload.7z"
        archive.write_bytes(compressed[off:])
        members=subprocess.run([tar,"-tf",str(archive)],check=True,capture_output=True).stdout.decode().splitlines()
        member,=[part for part in members if Path(part).name.casefold()==sas.name.casefold()]
        extracted=subprocess.run([tar,"-xOf",str(archive),member],check=True,capture_output=True).stdout
    if sas.exists() and sas.read_bytes()!=extracted: raise ValueError(f"Different USDA source {sas}")
    sas.write_bytes(extracted)
    return dict(file=sas.name,url=url,sha256=hashlib.sha256(extracted).hexdigest(),
                wrapper_sha256=hashlib.sha256(compressed).hexdigest())


def day_totals(year,day,code):
    suffix=f"{year%100:02d}{(year+1)%100:02d}"
    stem=f"DR{day}TOT_{code}"; prefix=f"DR{day}T"
    data=pd.read_sas(SOURCE/(stem+".xpt"),format="xport").set_index("SEQN",verify_integrity=True)
    data=data[data[f"DR{day}DRSTZ"].eq(1)&data.WTDR2D.gt(1e-12)]
    cols=["KCAL","SFAT","MFAT","PFAT","SODI"]
    data=data[[prefix+k for k in cols]+["WTDR2D"]].rename(columns={prefix+k:k for k in cols})
    fp=pd.read_sas(SOURCE/f"fped_dr{day}tot_{suffix}.sas7bdat")
    fp.columns=fp.columns.str.upper()
    fp=fp.set_index("SEQN",verify_integrity=True)
    fp.columns=[c.replace(f"DR{day}T_","") for c in fp.columns]
    fcols=["F_CITMLB","F_OTHER","F_TOTAL","V_DRKGR","V_LEGUMES","V_TOTAL","G_WHOLE","G_REFINED","D_TOTAL",
           "PF_TOTAL","PF_LEGUMES","PF_SEAFD_HI","PF_SEAFD_LOW","PF_NUTSDS","PF_SOY","ADD_SUGARS"]
    both=data.join(fp[fcols],how="inner").dropna()
    return both


def make_hei(t):
    energy=t.KCAL
    per1000=1000/energy
    def scaled(x,minval,maxval,points):
        return points*((x-minval)/(maxval-minval)).clip(0,1)
    comp=[scaled(t.F_TOTAL*per1000,0,.8,5),scaled((t.F_CITMLB+t.F_OTHER)*per1000,0,.4,5),
          scaled((t.V_TOTAL+t.V_LEGUMES)*per1000,0,1.1,5),scaled((t.V_DRKGR+t.V_LEGUMES)*per1000,0,.2,5),
          scaled(t.G_WHOLE*per1000,0,1.5,10),scaled(t.D_TOTAL*per1000,0,1.3,10),
          scaled((t.PF_TOTAL+t.PF_LEGUMES)*per1000,0,2.5,5),
          scaled((t.PF_SEAFD_HI+t.PF_SEAFD_LOW+t.PF_NUTSDS+t.PF_SOY+t.PF_LEGUMES)*per1000,0,.8,5),
          scaled((t.MFAT+t.PFAT)/t.SFAT,1.2,2.5,10),scaled(t.G_REFINED*per1000,4.3,1.8,10),
          scaled(t.SODI/energy,2,1.1,10),scaled(100*t.ADD_SUGARS*16/energy,26,6.5,10),
          scaled(100*t.SFAT*9/energy,16,8,10)]
    return pd.concat(comp,axis=1).sum(axis=1,min_count=13)


if __name__=="__main__":
    jobs=[(yr,d,s) for yr,s in zip(range(2005,2019,2),"DEFGHIJ") for d in (1,2)]
    with ThreadPoolExecutor(6) as pool:
        cdc=list(pool.map(lambda job:download((job[0],f"DR{job[1]}TOT_{job[2]}")),jobs))
        usda=list(pool.map(lambda job:fped(job[0],job[1]),jobs))
    (SOURCE/"food_manifest.json").write_text(json.dumps(cdc+usda,indent=2),encoding="utf8")
    out=[]
    for yr,s in zip(range(2005,2019,2),"DEFGHIJ"):
        a,b=day_totals(yr,1,s),day_totals(yr,2,s)
        idx=a.index.intersection(b.index)
        a,b=a.loc[idx],b.loc[idx]
        np.testing.assert_allclose(a.WTDR2D,b.WTDR2D)
        h=make_hei(a+b).rename("HEI2015")
        assert h.dropna().between(0,100).all()
        out.append(pd.concat([h,a.WTDR2D],axis=1))
    df=pd.concat(out,verify_integrity=True)
    df.to_csv(HERE/"hei2015_two_days.csv.gz",compression="gzip")
    print(df.HEI2015.describe())

