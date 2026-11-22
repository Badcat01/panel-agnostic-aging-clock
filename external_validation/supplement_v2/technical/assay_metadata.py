"""Attach traceable field mappings and explicitly documented assay procedures to the numerical audit."""
import ast
import json
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
EXT = HERE.parents[1]
sys.path.insert(0, str(EXT/"mimic_iv"))
from mimic_iv_validate import MAP

CHARLS_RANGES = {"crp":"0.1–20 mg/L","hba1c":"0–40%","cholesterol":"3–800 mg/dL",
                 "hdl":"3–120 mg/dL","triglycerides":"4–1000 mg/dL","bun":"5–100 mg/dL",
                 "creatinine":"0.1–25 mg/dL","glucose":"2–450 mg/dL","uric_acid":"up to 20 mg/dL"}
EXAM={"bmi","pulse","waist","sbp","dbp"}
CBC={"wbc","lymph_pct","mono_pct","neut_pct","eos_pct","rbc","hgb","mcv","mch","rdw","platelets","mpv"}
HRS_METHODS = {"albumin":"bromcresol purple","alp":"colorimetric assay","ast":"enzymatic assay",
               "alt":"enzymatic assay","bilirubin":"diazo colorimetry","bicarbonate":"CO2-L reagent",
               "crp":"latex-enhanced immunoturbidimetry","sodium":"ion-selective electrode",
               "chloride":"ion-selective electrode","potassium":"ion-selective electrode","glucose":"hexokinase"}

def enrich():
    src=HERE/"assay_audit.csv"
    d=pd.read_csv(src)
    script=(EXT.parent/"scripts/02_build_dataset.py").read_text(encoding="utf8")
    assign=next(n for n in ast.parse(script).body if isinstance(n,ast.Assign)
                and any(getattr(t,"id",None)=="FEATS" for t in n.targets))
    fmap=ast.literal_eval(assign.value)
    fmap.update(crp="CRP: LBXCRP mg/dL or LBXHSCRP mg/L / 10",sbp="BPXSY1–4 mean",
                dbp="BPXDI1–4 positive mean",pulse="BPXPLS",uacr="URXUMA / URXUCR × 100 mg/g")
    icu={k[0]:f"labevents {n} {k[1]}, raw {k[2]}" for n,k in MAP.items()}
    icu.update(sbp="chartevents 220179 mmHg",dbp="chartevents 220180 mmHg",pulse="chartevents 220045 bpm",
               bmi="kg 226512 / height metres squared (226707 inches or 226730 cm)")
    elsa=json.loads((EXT/"audit/elsa.json").read_text(encoding="utf8"))["feature_columns_units"]
    records=[]
    for row in d.itertuples():
        co,f=row.cohort,row.feature
        meta=EXT/"audit"/(co+".json")
        origin=json.loads(meta.read_text(encoding="utf8")) if meta.exists() else {}
        links=origin.get("official_sources",origin.get("documentation",origin.get("references",[])))
        if not isinstance(links,list):links=[links]
        details=dict(primary_document=" | ".join(str(x) for x in links if str(x).startswith("http")),
                     documented_assay_range="",conversion_note=origin.get("conversions",{}).get(f,"")
                     if isinstance(origin.get("conversions"),dict) else "")
        if not row.present:
            records.append(details);continue
        if co in ["development","temporal"]:
            details.update(raw_mapping=str(fmap[f]),primary_document="https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory",
                           fasting="general exam serum panel; not restricted to fasting participants",
                           platform="release and analyte specific; check cycle component protocol")
        elif co=="mimic":
            details.update(raw_mapping=icu[f],primary_document="https://physionet.org/content/mimiciv/3.1/",
                           platform="ICU chart" if f in EXAM else "hospital laboratory; instrument not linked per result",
                           fasting="not determined at the hospital 24-hour landmark",
                           LOD_handling="numeric values with qualifying source unit and release timestamps; instrument-specific LOD unavailable")
            if f=="crp":details["conversion_note"]="mg/L / 10 to mg/dL"
        elif co=="elsa":
            obj=elsa.get("source_variable_label_and_multiplier",{}).get(f,elsa.get(f))
            details.update(raw_mapping=json.dumps(obj,ensure_ascii=False),
                           fasting="W2 FASTELI; glucose fasting samples only" if f=="glucose" else "mixed fasting status, W2 FASTELI",
                           source_metadata="ELSA W2 nurse file; nurse visits user guide pp.16–18")
        elif co=="hrs":
            details.update(primary_document="https://hrsdata.isr.umich.edu/sites/default/files/documentation/data-descriptions/HRS2016VBSDD.pdf",
                           fasting="fasting requested but optional; 6280 yes / 3316 no / 15 unknown in included cohort")
            if f=="globulin":details["raw_mapping"]="PTP minus PALB"
            if f in HRS_METHODS:details["platform"]="Roche Cobas 6000; "+HRS_METHODS[f]
            if f in CBC:details["platform"]="EDTA whole-blood CBC; HRS protocol lists Sysmex XS1000i; delayed central processing"
        elif co=="charls":
            details["documented_assay_range"]=CHARLS_RANGES.get(f,"")
            if f in CBC: details["platform"]="local quality-assured clinical CBC laboratory; instrument varies"
        elif co=="lasi":
            details["fasting"]="preferably fasting; not universally required (LASI-DAD venous blood protocol)"
            if f in {"rbc","wbc","platelets","hgb"}: details["platform"]="Beckman Coulter LH780; Metropolis Laboratory"
            if f=="hba1c": details["platform"]="Bio-Rad D-10, ion-exchange HPLC"
            if f in {"glucose","cholesterol","hdl","triglycerides","bilirubin","total_protein",
                     "albumin","alp","ast","alt","bun","creatinine","uric_acid"}:
                details["platform"]="Abbott Architect ci8200; Metropolis Laboratory Table 4"
        if f in EXAM:
            details["LOD_handling"]="physical or questionnaire measurement; laboratory detection limit inapplicable"
            details["fasting"]="physical or questionnaire measurement"
        records.append(details)
    for i,extra in enumerate(records):
        for k,v in extra.items(): d.at[i,k]=v
    assert not d.loc[d.present,"raw_mapping"].isna().any()
    assert not d.loc[d.present,"raw_mapping"].isin(["null","None"]).any()
    d.to_csv(HERE/"assay_audit_documented.csv",index=False)

if __name__=="__main__":
    enrich()
