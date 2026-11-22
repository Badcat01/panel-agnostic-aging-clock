"""Build the two main tables and the twenty supplementary tables from the study results."""
import os
from pathlib import Path

import pandas as pd

import data_src as DS

OUT = os.environ["AC_OUT"]


T1H = ["Cohort", "Setting", "Country", "Baseline", "n", "Deaths", "Follow-up, years", "Age ≥65 years, %",
       "Women, %", "Inputs (of 43)"]
T2H = ["Setting", "n / deaths", "Clinical clock", "Comparator", "Value", "Δ (95% CI)"]


def table1_rows():
    t1 = DS.derived("table1_nhanes.csv")
    sg = DS.subgroups(); cs = DS.cohorts()
    def ext(code, name, setting, country, years, fu, inputs, deaths=None):
        a = sg[(sg.cohort == code) & (sg.group == "all")].iloc[0]
        f = sg[(sg.cohort == code) & (sg.group == "female")].iloc[0]
        o = sg[(sg.cohort == code) & (sg.group.isin(["age65_79", "age80plus"]))].n.sum()
        n = int(a.n)
        return [name, setting, country, years, f"{n:,}", f"{int(deaths if deaths else a.all_events):,}", fu,
                f"{100 * o / n:.1f}", f"{100 * f.n / n:.1f}", inputs]
    dev, hold = t1.iloc[0], t1.iloc[1]
    rows = [["NHANES 1999–2008", "Development", "USA", "1999–2008", f"{int(dev.n):,}", f"{int(dev.deaths):,}",
             f"{dev.fu_median:.1f}", f"{dev.age65_pct:.1f}", f"{dev.female_pct:.1f}", f"{int(dev.meas_median)}"],
            ["NHANES 2009–2018", "Temporal holdout", "USA", "2009–2018", f"{int(hold.n):,}", f"{int(hold.deaths):,}",
             f"{hold.fu_median:.1f}", f"{hold.age65_pct:.1f}", f"{hold.female_pct:.1f}", f"{int(hold.meas_median)}"]]
    fu = {r.cohort: r.followup_median_years for r in cs.itertuples()}
    rows.append(ext("nhanes3", "NHANES III", "Community", "USA", "1988–1994", f"{fu['NHANES III']:.1f}", "40"))
    rows.append(ext("hrs", "HRS", "Community", "USA", "2016", f"{fu['HRS 2016']:.1f}", "31"))
    rows.append(ext("charls", "CHARLS", "Community", "China", "2011", f"{fu['CHARLS']:.1f}", "18"))
    rows.append(ext("elsa", "ELSA", "Community", "UK", "2004–2005", f"{fu['ELSA']:.1f}", "12"))
    rows.append(ext("mimic", "MIMIC-IV v3.1", "Hospital", "USA", "2008–2022", "1.0", "19"))
    return rows


def table2_rows():
    s = DS.fig1b(); tm = DS.temporal_metrics(); re_ = DS.random_effects().set_index("model")
    d = DS.dnam_nhanes(); h = DS.hrs_disc(); m = DS.mimic_main(); icu = DS.icu_inc(); raw = DS.icu_raw()
    f = lambda x: f"{x:.3f}"
    sg = lambda v: f"{v:.4f}".replace("-", "−")
    ci = lambda e, lo, hi: f"{e:+.4f}".replace("-", "−") + f" ({sg(lo)} to {sg(hi)})"
    rows = []
    def coh(code, label):
        a = s[(s.cohort == code) & (s.model == "full_lp")].iloc[0]; r = s[(s.cohort == code) & (s.model == "ridge_lp")].iloc[0]
        rows.append([label, f"{int(a.n):,} / {int(a.events):,}", f(a.C), "Ridge Cox, same inputs", f(r.C),
                     ci(r.clock_minus_model, r.delta_low, r.delta_high)])
    coh("temporal", "NHANES temporal holdout")
    pc = tm["phenoage_complete_cases"]
    rows.append(["Holdout, PhenoAge-complete", f"{pc['n']:,} / {pc['deaths']:,}", f(pc["C_DL"]), "Clinical PhenoAge",
                 f(pc["C_PhenoAge"]), f"{pc['C_DL'] - pc['C_PhenoAge']:+.4f}ᶜ"])
    coh("nhanes3", "NHANES III"); coh("hrs", "HRS"); coh("charls", "CHARLS"); coh("elsa", "ELSA")
    x = re_.loc["M3_XGB_raw"]
    rows.append(["Four community cohorts, pooled", "43,104 / 10,499", "—", "XGBoost-Cox", "—", ci(x.delta, x.low, x.high)])
    g = d[(d.analysis == "primary") & (d.model == "GrimAge2Mort")].iloc[0]; c = d[(d.analysis == "primary") & (d.model == "clinical")].iloc[0]
    rows.append(["NHANES DNAm subset", "2,525 / 1,359", f(c.C), "GrimAge2", f(g.C), ci(g.clinical_minus_C, g.delta_lower, g.delta_upper)])
    hp = h[(h.analysis == "primary")].iloc[0]
    rows.append(["HRS DNAm sample", "3,967 / 732", f(hp.C_clinical), "GrimAge2", f(hp.C_dnam),
                 ci(hp.delta, hp.lower, hp.upper) + "ᵃ"])
    for o, lab in [("In-hospital", "MIMIC-IV, in-hospital death"), ("365-day", "MIMIC-IV, 365-day death")]:
        a = m[(m.panel == "a") & (m.outcome == o) & (m.model == "Clinical clock")].iloc[0]
        r = m[(m.panel == "a") & (m.outcome == o) & (m.model == "Same-input ridge Cox")].iloc[0]
        b = m[(m.panel == "b") & (m.outcome == o)].iloc[0]
        rows.append([lab, f"130,242 / {int(a.deaths):,}", f(a.value), "Ridge Cox, same inputs", f(r.value), ci(b.value, b.low, b.high)])
    rr = raw[(raw.population == "all_27005") & (raw.outcome == "event")]
    cl = rr[rr.score == "hospital24"].iloc[0]; sp = rr[rr.score == "sapsii"].iloc[0]
    ii = icu[(icu.population == "all_27005") & (icu.outcome == "event") & (icu.base_severity == "sapsii") & (icu.clock == "hospital24")].iloc[0]
    lo, hi = [float(v) for v in ii.incremental_auc_oof_ci95.strip("[]").split(",")]
    rows.append(["MIMIC-IV ICU, 1-year death", f"27,005 / {int(cl.events):,}", f(cl.AUC), "SAPS-II", f(sp.AUC),
                 ci(ii.incremental_auc_oof, lo, hi) + "ᵇ"])
    return rows


def supp_tables():
    import json
    bundle = json.load(open(f"{DS.S}/runs/nhanes_gpu_20261121/bundle/clock_bundle.json", encoding="utf-8"))
    dictionary = pd.read_csv(Path(__file__).resolve().parents[1] / "docs" / "input_dictionary.csv")
    if dictionary["variable"].tolist() != bundle["features"]:
        raise ValueError("Input dictionary order differs from the frozen model")
    st1 = dictionary.assign(
        log_transformed=dictionary["variable"].isin(bundle["log_features"]),
        dev_mean_transformed=[round(v, 4) for v in bundle["mean"]],
        dev_sd_transformed=[round(v, 4) for v in bundle["std"]],
    )
    cs = DS.cohort_summary()
    region = {"中国": "China", "美国": "USA", "英国": "UK", "墨西哥": "Mexico", "欧洲与以色列": "Europe and Israel",
              "韩国": "Korea", "印度": "India"}
    role = {"预定": "Primary", "补充": "Supplementary", "探索": "Exploratory", "仅评分": "Scoring only"}
    cs = cs.assign(region=cs.region.map(region), role=cs.role.map(role))
    sheets = [
        ("ST1 Input dictionary", "Supplementary Table 1 | Model input dictionary, units and development standardization parameters", st1),
        ("ST2 Cohort discrimination", "Supplementary Table 2 | Cohort characteristics and discrimination in all external cohorts", cs),
        ("ST3 Fixed horizons", "Supplementary Table 3 | Fixed-horizon AUC, Brier score and calibration", DS.horizons()),
        ("ST4 Age-sex subgroups", "Supplementary Table 4 | Discrimination and calibration by sex and age group", DS.subgroups()),
        ("ST5 Panel experiments", "Supplementary Table 5 | Simulated, actual and random-panel comparisons with alternative models",
         DS.csv(f"{DS.SV}/models/comparisons.csv").drop(columns=["hash"], errors="ignore")),
        ("ST6 Random effects", "Supplementary Table 6 | Random-effects pooled comparisons across four community cohorts", DS.random_effects()),
        ("ST7 Leave-one-cohort-out", "Supplementary Table 7 | Frozen clock versus leave-one-cohort-out retrained networks", DS.loco()),
        ("ST8 NHANES DNAm", "Supplementary Table 8 | NHANES DNAm head-to-head comparisons", DS.dnam_nhanes()),
        ("ST9 NHANES DNAm HR LRT", "Supplementary Table 9 | NHANES DNAm hazard ratios and likelihood-ratio tests",
         pd.concat([DS.dnam_hr().assign(table="HR"), DS.dnam_lrt().assign(table="LRT")])),
        ("ST10 HRS discrimination", "Supplementary Table 10 | HRS DNAm head-to-head discrimination (primary and sensitivity)", DS.hrs_disc()),
        ("ST11 HRS HR LRT", "Supplementary Table 11 | HRS hazard ratios and likelihood-ratio tests",
         pd.concat([DS.hrs_hr().assign(table="HR"), DS.hrs_lrt().assign(table="LRT")])),
        ("ST12 HRS healthspan", "Supplementary Table 12 | HRS incident disability and cognitive impairment (mutually adjusted)",
         DS.hrs_health()[DS.hrs_health().model != "new chronic count, Poisson rate"]),
        ("ST13 MIMIC panels", "Supplementary Table 13 | MIMIC-IV whole-body and system-score performance", DS.mimic_panels()),
        ("ST14 ICU comparisons", "Supplementary Table 14 | ICU severity score comparisons and incremental value",
         pd.concat([DS.icu_raw().assign(table="standalone AUC"), DS.icu_inc().assign(table="incremental AUC")])),
        ("ST15 MIMIC 7-day events", "Supplementary Table 15 | MIMIC-IV seven-day organ-related events", DS.mimic_incident()),
        ("ST16 NHANES III organ", "Supplementary Table 16 | NHANES III organ validation and cause-specific mortality matrix",
         pd.concat([DS.n3_validation().assign(table="matched validation"), DS.cause_matrix().assign(table="cause x score matrix")])),
        ("ST17 HRS organ references", "Supplementary Table 17 | HRS measured references and DNAm surrogates by system score",
         pd.concat([DS.hrs_vbs_ref()[DS.hrs_vbs_ref().group == "clock_GrimAge2"].assign(table="measured reference"),
                    DS.hrs_proxy().assign(table="DNAm surrogate")])),
        ("ST18 Community disease", "Supplementary Table 18 | Matched incident disease in community cohorts", DS.disease_diag()),
        ("ST19 Exposome", "Supplementary Table 19 | Exposome-wide associations across three NHANES stages", DS.xwas()),
        ("ST20 Sex-age profiles", "Supplementary Table 20 | Whole-body and system scores by sex and age (temporal holdout)",
         DS.derived("organ_sex_age_profiles.csv")),
    ]
    os.makedirs(f"{OUT}/07_Supplementary_Information", exist_ok=True)
    path = f"{OUT}/07_Supplementary_Information/Supplementary_Tables_1-20.xlsx"
    from openpyxl.styles import Font, Alignment
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"Sheet": [s[0] for s in sheets], "Title": [s[1] for s in sheets]}).to_excel(xw, sheet_name="Contents", index=False)
        for sh, title, df in sheets:
            df = df.copy()
            df.to_excel(xw, sheet_name=sh[:31], index=False, startrow=2)
            ws = xw.sheets[sh[:31]]
            ws["A1"] = title; ws["A1"].font = Font(bold=True, size=11)
            for c in ws[3]:
                c.font = Font(bold=True)
        xw.sheets["Contents"].column_dimensions["A"].width = 28
        xw.sheets["Contents"].column_dimensions["B"].width = 110
    return sheets

if __name__ == "__main__":
    dest = Path(OUT) / "05_Main_Tables"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table1_rows(), columns=T1H).to_excel(dest / "Table1.xlsx", index=False)
    pd.DataFrame(table2_rows(), columns=T2H).to_excel(dest / "Table2.xlsx", index=False)
    supp_tables()
