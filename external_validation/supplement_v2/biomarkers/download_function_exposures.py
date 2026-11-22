"""Download US survey function and exposure components using per-cycle NCHS catalogues."""
import concurrent.futures
import html
import json
import re
from pathlib import Path

import requests
from download_references import download, SOURCE

YEARS=range(1999,2019,2)
FAMILIES={
    "cotinine":r"^Cotinine - Serum$|^Cotinine( & Hydroxycotinine| and Hydroxycotinine| and trans-3.-Hydroxycotinine).*Serum|^Cotinine - Serum &",
    "metals": r"^Lead, Cadmium.*Blood|^Cadmium, Lead.*Blood|^Blood Lead.*Cadmium|^Cadmium, Lead, Mercury",
    "PFAS": r"^Perfluoroalkyl|^Perfluorinated Chemicals|^Polyfluoroalkyl",
    "phthalates": r"^Phthalates.*Urine|^Phthalate.*Metabolites|^Phthalates - Urine",
    "VOC": r"^Volatile Organic Compounds.*Blood|^Volatile Organic Compounds \(VOCs\)$",
}
QUEST=["PAQ","SLQ","SMQ"]
HEAD={"User-Agent":"Mozilla/5.0 NHANES research analysis"}
def discover(year):
    url=f"https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Component=Laboratory&CycleBeginYear={year}"
    response=requests.get(url,headers=HEAD,timeout=40);response.raise_for_status()
    picked={}
    for row in re.findall(r"<tr\b.*?</tr>",response.text,flags=re.I|re.S):
        doc=re.search(r"/DataFiles/([a-zA-Z0-9_]+)\.htm",row,re.I)
        if not doc: continue
        cells=[html.unescape(re.sub(r"<[^>]+>"," ",cell)).strip() for cell in re.findall(r"<td\b[^>]*>(.*?)</td>",row,re.I|re.S)]
        if len(cells)<3: continue
        label=" ".join(cells[1 if re.search(r'19\d\d|20\d\d',cells[0]) else 0].split())
        for fam,pat in FAMILIES.items():
            if re.search(pat,label,re.I) and not re.search(r"special|pooled|pools|isomers|method|second exam|volatile organic compounds and metabolites - blood|expanded|formerly.*pooled",label,re.I):
                picked.setdefault(fam,[]).append([doc.group(1),label])
    return year,picked

if __name__=="__main__":
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        catalog=dict(pool.map(discover,YEARS))
    print(json.dumps(catalog,ensure_ascii=True,indent=1),flush=True)
    (SOURCE/"exposure_catalog.json").write_text(json.dumps(catalog,indent=2),encoding="utf8")
    data=[]
    for year,suf in zip(YEARS,["","_B","_C","_D","_E","_F","_G","_H","_I","_J"]):
        for fam,choices in catalog[year].items():
            for name,label in choices:
                data.append((year,name))
        queries=(["PFQ"] if year>=2009 else [])+(["CFQ","MGX"] if year in [2011,2013] else [])+["SMQ"]+(["PAQ"] if year>=2007 else [])+(["SLQ"] if year>=2005 else [])
        data.extend((year,q+suf) for q in queries)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        rows=list(pool.map(download,sorted(set(data))))
    (SOURCE/"extra_manifest.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
    print(len(rows),"files verified",flush=True)

