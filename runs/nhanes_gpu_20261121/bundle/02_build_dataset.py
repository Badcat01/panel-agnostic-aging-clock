"""Merge NHANES 1999-2018 into one analysis table with harmonised units + linked mortality.
Output: data/processed/nhanes_9918.parquet (csv fallback)
"""
import os, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
CYC = {1999: "", 2001: "_B", 2003: "_C", 2005: "_D", 2007: "_E",
       2009: "_F", 2011: "_G", 2013: "_H", 2015: "_I", 2017: "_J"}
STEMS = {
    "BIOPRO": {1999: "LAB18", 2001: "L40_B", 2003: "L40_C"},
    "CBC": {1999: "LAB25", 2001: "L25_B", 2003: "L25_C"},
    "GHB": {1999: "LAB10", 2001: "L10_B", 2003: "L10_C"},
    "CRP": {1999: "LAB11", 2001: "L11_B", 2003: "L11_C", 2011: None, 2013: None, 2015: "HSCRP_I", 2017: "HSCRP_J"},
    "ALB_CR": {1999: "LAB16", 2001: "L16_B", 2003: "L16_C"},
    "HDL": {1999: "LAB13", 2001: "L13_B", 2003: "L13_C"},
}


def stem(comp, yr):
    if comp in STEMS and yr in STEMS[comp]:
        return STEMS[comp][yr]
    return comp + CYC[yr]


def rd(comp, yr):
    s = stem(comp, yr)
    if s is None:
        return None
    p = os.path.join(RAW, "nhanes", s + ".xpt")
    if not os.path.exists(p):
        print("  missing", s)
        return None
    df = pd.read_sas(p, format="xport", encoding="latin1")
    df["SEQN"] = df["SEQN"].astype(int)
    return df.set_index("SEQN")


def pick(df, names):
    """first existing column among candidate names"""
    if df is None:
        return None
    for n in names:
        if n in df.columns:
            return df[n]
    return None


FEATS = {  # output name: (component, candidate columns)  -- conventional units
    "albumin": ("BIOPRO", ["LBXSAL"]),          # g/dL
    "alp": ("BIOPRO", ["LBXSAPSI", "LBDSAPSI"]),            # U/L
    "ast": ("BIOPRO", ["LBXSASSI"]),
    "alt": ("BIOPRO", ["LBXSATSI"]),
    "bun": ("BIOPRO", ["LBXSBU"]),              # mg/dL
    "calcium": ("BIOPRO", ["LBXSCA"]),
    "cholesterol": ("BIOPRO", ["LBXSCH"]),
    "creatinine": ("BIOPRO", ["LBXSCR", "LBDSCR"]),       # mg/dL
    "globulin": ("BIOPRO", ["LBXSGB"]),
    "glucose": ("BIOPRO", ["LBXSGL"]),          # mg/dL (serum, non-fasting)
    "ggt": ("BIOPRO", ["LBXSGTSI"]),
    "iron": ("BIOPRO", ["LBXSIR"]),
    "potassium": ("BIOPRO", ["LBXSKSI"]),
    "ldh": ("BIOPRO", ["LBXSLDSI", "LBDSLDSI"]),
    "sodium": ("BIOPRO", ["LBXSNASI"]),
    "phosphorus": ("BIOPRO", ["LBXSPH", "LBDSPH"]),
    "bilirubin": ("BIOPRO", ["LBXSTB", "LBDSTB"]),
    "total_protein": ("BIOPRO", ["LBXSTP"]),
    "triglycerides": ("BIOPRO", ["LBXSTR"]),
    "uric_acid": ("BIOPRO", ["LBXSUA"]),
    "bicarbonate": ("BIOPRO", ["LBXSC3SI"]),
    "chloride": ("BIOPRO", ["LBXSCLSI"]),
    "wbc": ("CBC", ["LBXWBCSI"]),
    "lymph_pct": ("CBC", ["LBXLYPCT"]),
    "mono_pct": ("CBC", ["LBXMOPCT"]),
    "neut_pct": ("CBC", ["LBXNEPCT"]),
    "eos_pct": ("CBC", ["LBXEOPCT"]),
    "rbc": ("CBC", ["LBXRBCSI"]),
    "hgb": ("CBC", ["LBXHGB"]),
    "mcv": ("CBC", ["LBXMCVSI"]),
    "mch": ("CBC", ["LBXMCHSI"]),
    "rdw": ("CBC", ["LBXRDW"]),
    "platelets": ("CBC", ["LBXPLTSI"]),
    "mpv": ("CBC", ["LBXMPSI"]),
    "hba1c": ("GHB", ["LBXGH"]),
    "hdl": ("HDL", ["LBDHDD", "LBDHDL", "LBXHDD"]),
    "urine_alb": ("ALB_CR", ["URXUMA"]),
    "urine_cr": ("ALB_CR", ["URXUCR"]),
    "bmi": ("BMX", ["BMXBMI"]),
    "waist": ("BMX", ["BMXWAIST"]),
    "height": ("BMX", ["BMXHT"]),
    "weight": ("BMX", ["BMXWT"]),
}


def read_mort(yr):
    p = os.path.join(RAW, "mort", f"NHANES_{yr}_{yr+1}_MORT_2019_PUBLIC.dat")
    rows = []
    with open(p) as f:
        for line in f:
            line = line.rstrip("\n").ljust(48)
            g = lambda a, b: line[a - 1:b].strip()
            rows.append(dict(SEQN=int(g(1, 6)), eligstat=g(15, 15), mortstat=g(16, 16), ucod=g(17, 19),
                             permth_exm=g(46, 48)))
    m = pd.DataFrame(rows)
    for c in ["eligstat", "mortstat", "ucod", "permth_exm"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m.set_index("SEQN")


frames = []
for yr in CYC:
    print("cycle", yr)
    demo = rd("DEMO", yr)
    comps = {c: rd(c, yr) for c in ["BIOPRO", "CBC", "GHB", "CRP", "ALB_CR", "HDL", "BMX", "BPX"]}
    d = pd.DataFrame(index=demo.index)
    d["cycle"] = yr
    d["age"] = demo["RIDAGEYR"]
    d["male"] = (demo["RIAGENDR"] == 1).astype(float)
    d["race"] = demo["RIDRETH1"]
    d["pregnant"] = demo["RIDEXPRG"] if "RIDEXPRG" in demo else np.nan
    d["wt_mec"] = demo["WTMEC2YR"]
    d["psu"] = demo["SDMVPSU"]; d["strata"] = demo["SDMVSTRA"]
    for name, (comp, cands) in FEATS.items():
        s = pick(comps[comp], cands)
        d[name] = s.reindex(d.index) if s is not None else np.nan
    # CRP harmonised to mg/dL (hsCRP 2015-18 is mg/L)
    crp = comps["CRP"]
    if crp is not None:
        if "LBXHSCRP" in crp:
            d["crp"] = (crp["LBXHSCRP"] / 10.0).reindex(d.index)
        else:
            d["crp"] = crp["LBXCRP"].reindex(d.index)
    else:
        d["crp"] = np.nan
    bp = comps["BPX"]
    if bp is not None:
        sy = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in bp]
        di = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in bp]
        d["sbp"] = bp[sy].mean(axis=1).reindex(d.index)
        dia = bp[di].replace(0, np.nan).mean(axis=1)
        d["dbp"] = dia.reindex(d.index)
        d["pulse"] = bp["BPXPLS"].reindex(d.index) if "BPXPLS" in bp else np.nan
    m = read_mort(yr)
    d = d.join(m, how="left")
    frames.append(d)

df = pd.concat(frames)
df["uacr"] = df["urine_alb"] / df["urine_cr"] * 100  # mg/g
df = df.drop(columns=["urine_alb", "urine_cr"])
# cohort: adults >=20, mortality-eligible, not pregnant, follow-up known
n0 = len(df)
df = df[(df.age >= 20) & (df.eligstat == 1) & (df.pregnant != 1) & df.permth_exm.notna()]
df["time_yr"] = df["permth_exm"] / 12.0
df["event"] = df["mortstat"].astype(int)
df = df[df.time_yr > 0]
print(f"rows {n0} -> analytic adults {len(df)}; deaths {df.event.sum()}")
os.makedirs(OUT, exist_ok=True)
df.reset_index().to_csv(os.path.join(OUT, "nhanes_9918.csv.gz"), index=False)

# coverage table
cov = df.drop(columns=["eligstat", "mortstat", "ucod", "permth_exm"]).groupby("cycle").apply(lambda x: x.notna().mean()).T
cov.round(2).to_csv(os.path.join(ROOT, "results", "feature_coverage_by_cycle.csv"))
print(cov.round(2).to_string())

# epigenetic clocks subset
p = os.path.join(RAW, "nhanes", "dnmepi.sas7bdat")
if os.path.exists(p):
    e = pd.read_sas(p, encoding="latin1")
    e = e[e.HorvathAge.notna()]
    e["SEQN"] = e["SEQN"].astype(int)
    e.to_csv(os.path.join(OUT, "dnmepi.csv.gz"), index=False)
    print("DNMEPI", e.shape, list(e.columns))
