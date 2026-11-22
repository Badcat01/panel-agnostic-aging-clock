"""Link the measured fields in each NHANES cycle to their official component and method documents."""
import ast
import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
from pathlib import Path
import re
from urllib.parse import urljoin
import pandas as pd
import requests

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
CACHE=HERE/"assay_docs"/"nhanes"
REGEX={"albumin":r"\balbumin\b","alp":"alkaline phosphatase","ast":r"\bast\b|aspartate amino",
       "alt":r"\balt\b|alanine amino","ggt":r"\bggt\b|glutamyl transfer","ldh":"lactate dehydrogenase",
       "bun":r"\bbun\b|urea nitrogen","phosphorus":r"phosphor|phosphate","triglycerides":"triglyceride",
       "total_protein":"total protein","calcium":"calcium","chloride":"chloride","potassium":"potassium",
       "bicarbonate":"bicarbonate","creatinine":"creatinine","uric_acid":"uric acid","cholesterol":"cholesterol",
       "glucose":"glucose","bilirubin":"bilirubin","sodium":"sodium","iron":r"\biron\b"}
MODELS=r"\b(?:(?:Beckman(?:[ -]Coulter)?\s+)?(?:DxC\s*\d+(?:i)?|Synchron\s+(?:LX\s*20|CX\d+)|UniCel\s+DxC\s*\d+)|(?:Roche\s+)?(?:Cobas\s+[a-z]?\s*\d{3,4}|Modular\s+P(?:\s*\d+)?|Hitachi\s+\d+)|(?:Coulter\s+)?(?:MAXM|HMX|DxH\s*\d+)|(?:Tosoh\s+)?(?:G7|G8)\s+(?:HPLC|glycohemoglobin))\b"

def constants():
    tree=ast.parse((PROJECT/"scripts/02_build_dataset.py").read_text(encoding="utf8"))
    return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ["CYC","STEMS","FEATS"]}

def inspect(job):
    yr,group,stem=job
    url=f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{yr}/DataFiles/{stem}.htm"
    file=CACHE/f"{stem}.html"
    if not file.exists():
        for attempt in range(3):
            try:
                r=requests.get(url,timeout=20);r.raise_for_status()
                if stem.lower() not in r.text.lower() or "sequence number" not in r.text.lower():raise ValueError("not the specified codebook")
                file.write_bytes(r.content)
                break
            except (requests.RequestException,ValueError):
                if attempt==2:return (url,"",[],[],"official HTML not recovered")
    data=file.read_bytes();s=data.decode("utf8",errors="replace")
    simple=lambda x:" ".join(unescape(re.sub("<[^>]*>"," ",x)).split()).strip()
    links=[]
    for _,_,href,title in re.findall(r"<a\b([^>]*?)href=(['\x22])(.*?)\2[^>]*>(.*?)</a>",s,re.I|re.S):
        if ".pdf" in href.lower() and ("labmethod" in href.lower() or "method" in title.lower()):
            entry=(simple(title),urljoin(url,unescape(href)))
            if entry not in links:links.append(entry)
    matches=list(dict.fromkeys(re.sub(r"\s+"," ",m.group()) for m in re.finditer(MODELS,simple(s),flags=re.I)))
    return url,sha256(data).hexdigest(),links,matches,"verified official component page"

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    x=constants()
    feature=dict(x["FEATS"])
    feature.update(crp=("CRP",["LBXHSCRP","LBXCRP"]),uacr=("ALB_CR",["URXUMA","URXUCR"]),
                   sbp=("BPX",[f"BPXSY{i}" for i in range(1,5)]),
                   dbp=("BPX",[f"BPXDI{i}" for i in range(1,5)]),pulse=("BPX",["BPXPLS"]))
    names=json.loads((PROJECT/"runs/nhanes_gpu_20261121/bundle/clock_bundle.json").read_text())["features"]
    stem_for=lambda group,year:x["STEMS"].get(group,{}).get(year,group+x["CYC"][year])
    jobs=sorted({(year,feature[f][0],stem_for(feature[f][0],year)) for f in names for year in x["CYC"]
                 if stem_for(feature[f][0],year)})
    with ThreadPoolExecutor(max_workers=4) as pool:
        meta={j[2]:v for j,v in zip(jobs,pool.map(inspect,jobs))}
    cols={}
    for _,_,s in jobs:
        p=PROJECT/"data/raw/nhanes"/(s+".xpt")
        if p.exists():
            with pd.read_sas(p,format="xport",encoding="latin1",iterator=True) as raw:
                cols[s]=list(raw.read(1))
    rows=[]
    for year in x["CYC"]:
        for f in names:
            group,candidates=feature[f];stem=stem_for(group,year)
            row=dict(cycle_start=year,feature=f,component=group,release_stem=stem)
            present=[c for c in candidates if c in cols.get(stem,[])]
            row["source_fields"]=" + ".join(present if f in ("sbp","dbp","uacr") else present[:1])
            if stem:
                url,digest,links,models,status=meta[stem]
                if group=="BIOPRO": exact=[l for l in links if f in REGEX and re.search(REGEX[f],l[0],re.I)]
                else:exact=links
                row.update(codebook=url,codebook_sha256=digest,codebook_status=status,
                           documented_methods=" | ".join(k+" "+u for k,u in exact),
                           instrument_names_in_component_document=" | ".join(models))
            rows.append(row)
    a=pd.DataFrame(rows)
    assert len(a)==430
    a.to_csv(HERE/"nhanes_release_methods.csv",index=False)
    print("cycle rows",len(a),"HTML verified",sum(bool(v[1]) for v in meta.values()),"/",len(meta),"rows with fields",a.source_fields.ne("").sum())

if __name__=="__main__": main()
