"""Write the HRS paired-clock report from named aggregate tables only."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


TABLES = (
    "discrimination", "HR", "LRT", "healthspan", "availability", "organ_regression",
    "VBS_reference_matrix", "VBS_reference_unavailable", "dnam_proxy_organ", "dnam_proxy_unavailable",
)
CONTROLS = ("paired.csv.gz.json", "tracker.csv.gz.json", "health_verification.json", "release_versions.json")
PRIMARY = "clock_GrimAge2"
LABELS = {
    "primary": "主要比较", "other_clock": "各时钟配对", "all_clocks_complete": "所有时钟齐全",
    "one_year_landmark": "满一年地标", "age_cap_80": "临床输入年龄上限 80 岁",
    "age_cap_85": "临床输入年龄上限 85 岁", "time_lower_yr": "死亡时间区间下界",
    "time_upper_yr": "死亡时间区间上界", "cell_composition": "调整细胞组成",
    "survey_design": "专项权重与分层抽样设计",
    "clinical":"仅临床分数", "dnam":"仅甲基化时钟", "joint":"两者同时纳入",
    "a":"临床分数", "b":"甲基化时钟", "adl":"新发 ADL 失能", "lw":"新发认知障碍",
    "new_conditions":"新发慢病数", "cause_specific":"原因别风险比",
    "fine_gray":"Fine–Gray 亚分布风险比", "new chronic count, Poisson rate":"发生率比",
    "liver":"肝", "kidney":"肾", "immune_heme":"免疫/血液", "cardiovascular":"心血管",
    "metabolic":"代谢", "cystatin":"胱抑素 C", "ntprobnp":"NT-proBNP",
    "il6":"IL-6", "stnfr1":"sTNFR1", "cmv":"CMV 抗体",
    "proxy_cystatin":"DNAm 胱抑素 C", "proxy_b2m":"DNAm β₂-微球蛋白", "proxy_leptin":"DNAm 瘦素",
    "proxy_loga1c":"DNAm log A1c", "proxy_logcrp":"DNAm log CRP", "proxy_gdf15":"DNAm GDF15",
    "proxy_pai1":"DNAm PAI-1", "proxy_adm":"DNAm 肾上腺髓质素",
}


def named(value):
    s = str(value).removeprefix("clock_").removeprefix("organ_")
    return LABELS.get(s, s)


def num(x, digits=4):
    return "未估计" if pd.isna(x) else f"{x:.{digits}f}"


def integer(x):
    return "未报告" if pd.isna(x) else f"{int(x):,}"


def prob(x):
    if pd.isna(x):
        return "未估计"
    return "<0.0001" if x < .0001 else f"{x:.4f}"


def ci(lo, hi):
    return "未估计" if pd.isna(lo) or pd.isna(hi) else f"[{num(lo)}, {num(hi)}]"


def md_table(headers, rows):
    def row(values):
        return "| " + " | ".join(str(v).replace("|", "／").replace("\n", " ") for v in values) + " |"
    content = list(rows)
    if not content:
        return "该聚合表中无可报告的估计。"
    return "\n".join([row(headers), row(["---"] * len(headers)), *[row(r) for r in content]])


def matching(frame, **tests):
    if frame.empty:
        return frame
    keep = pd.Series(True, index=frame.index)
    for column, value in tests.items():
        keep &= frame[column].eq(value)
    return frame.loc[keep]


def primary_report(disc, fits, tests):
    pf = matching(disc, analysis="primary", clock=PRIMARY)
    tests = matching(tests, analysis="primary", clock=PRIMARY)
    if len(pf)>1 or len(tests)>1:
        raise ValueError("Repeated primary comparison")
    if pf.empty:
        return ["主要比较缺少可评价的 GrimAge2 配对结果，H1 和 H2 均待评估。"]
    a = pf.iloc[0]
    if not np.isclose(a.delta, a.C_clinical - a.C_dnam, atol=1e-8):
        raise ValueError("Inconsistent paired clinical-minus-GrimAge2 difference")
    lines = [
        f"临床分数与 GrimAge2 配对者共 **{integer(a.n)} 人、{integer(a.families)} 户**，随访期间死亡 **{integer(a.deaths)} 人**，抽血后三年内死亡 {integer(a.deaths_3y)} 人。",
        f"加入临床分数后，家庭五折留出预测的 Harrell C 为 **{num(a.C_clinical)}**，加入 GrimAge2 后为 **{num(a.C_dnam)}**。"
        f"两模型均调整年龄与性别。临床减 GrimAge2 的差值为 **{num(a.delta)}**，95% 区间为 **{ci(a.get('lower'),a.get('upper'))}**。",
    ]
    complete = a.get("requested") == 2000 and a.get("successful") == 2000
    if pd.notna(a.get("lower")) and pd.notna(a.get("upper")) and complete:
        if a.lower > a.upper:
            raise ValueError("Inverted primary interval")
        h1 = a.lower > -.010
        if a.get("H1") != ("supported" if h1 else "not_supported"):
            raise ValueError("H1 in aggregate disagrees with the pre-specified margin")
        if h1:
            verdict = "支持临床分数非劣于 GrimAge2"
        elif a.upper < -.010:
            verdict = "不支持非劣效，差值区间整体位于约定界值以下"
        else:
            verdict = "现有精度不足以按约定界值确认临床分数非劣于 GrimAge2"
        lines.append(f"**H1**　{verdict}。约定界值为 −0.010，按家庭配对重抽并重新拟合成功 2,000/2,000 次，以差值双侧 95% 区间下界作判定。")
    else:
        lines.append(f"**H1**　有效重抽 {integer(a.get('successful'))}/{integer(a.get('requested'))} 次，完整的配对区间不可用，非劣效待评估。")
    if not tests.empty:
        lrt = tests.iloc[0]
        p = [lrt.p_add_dnam, lrt.p_add_clinical]
        if any(pd.isna(x) for x in p):
            lines.append("**H2**　两个方向的检验未全部取得有效 p 值，独立增量待评估。")
        else:
            supported = max(p) < .01
            if str(lrt.H2).lower() != str(supported).lower():
                raise ValueError("H2 in aggregate disagrees with the two prespecified tests")
            verdict = "两种分数各自均提供额外死亡风险信息" if supported else "两种分数各自均提供额外信息这一联合假设未达到预定标准"
            lines.append(f"**H2**　{verdict}。已有临床分数时加入 GrimAge2 的似然比检验 p 为 {prob(p[0])}，已有 GrimAge2 时加入临床分数的 p 为 {prob(p[1])}，两个方向均要求 p<0.01。联合模型折外 C 为 {num(a.C_joint)}。")
    else:
        lines.append("**H2**　缺少主要比较的双向似然比检验，独立增量待评估。")
    lines.append(f"三年 Uno C 分别为临床 {num(a.Uno3_clinical)}、GrimAge2 {num(a.Uno3_dnam)}、联合 {num(a.Uno3_joint)}。")
    primary_hr = matching(fits,analysis="primary",clock=PRIMARY)
    lines.append("下表报告每升高 1 个标准差的死亡风险比和家庭聚类稳健区间。")
    lines.append(md_table(["年龄、性别调整模型","变量","HR","95% 区间","p"],
                          ((named(r.model),named(r.term),num(r.HR),ci(r.lower,r.upper),prob(r.p)) for r in primary_hr.itertuples())))
    return lines


def discrimination_table(table):
    return md_table(["分析","甲基化时钟","人数","家庭","死亡","临床 C","甲基化 C","联合 C","ΔC","ΔC 95% 区间"],
                    ((named(r.analysis),named(r.clock),integer(r.n),integer(r.families),integer(r.deaths),
                      num(r.C_clinical),num(r.C_dnam),num(r.C_joint),num(r.delta),
                      ci(getattr(r,"lower",None),getattr(r,"upper",None))) for r in table.itertuples()))


def flow_report(controls, status):
    paired, tracker, health, release = (controls.get(x,{}) for x in CONTROLS)
    if paired and (paired.get("prepared_sha256") != status["input_hash"] or paired.get("n_clinical") != status["source_rows"]):
        raise ValueError("Flow metadata is not from the current paired run")
    cutoffs = {x for x in [paired.get("tracker_cutoff"),tracker.get("cutoff"),
                           health.get("tracker_administrative_cutoff"),release.get("tracker_cutoff")] if x}
    if len(cutoffs)>1:
        raise ValueError("Aggregate files disagree on the mortality administrative cutoff")
    if paired and release and paired.get("mapping_sha256") != release.get("mapping_sha256"):
        raise ValueError("Mismatched mapping hashes in the release and paired summaries")
    text = ["## 数据来源、随访和筛选"]
    if cutoffs:
        text.append(f"死亡追踪采用 {paired.get('tracker_version') or tracker.get('tracker_version') or '项目记录的 Tracker 版本'}。"
                    f"项目选定的行政随访上界为 **{next(iter(cutoffs))}**，存活者按最后确认存活时间删失。")
    if release.get("official_osf_registration") is False:
        text.append("项目有本地冻结的分析方案，无公开 OSF 注册号。")
    if tracker:
        text.append(f"2016 年 VBS 抽血名册共 {integer(tracker.get('n_VBS'))} 人，其中死亡或删失时间可判定者为 {integer(tracker.get('valid_time'))} 人，"
                    f"纳入死亡事件 {integer(tracker.get('included_deaths'))} 例。报告死亡且缺少日期者 {integer(tracker.get('death_missing_date'))} 人。")
    if paired:
        text.append(f"临床分数可用者 {integer(paired.get('n_clinical'))} 人，其中与经质控 DNAm 产品匹配 {integer(paired.get('n_DNAm_QC_matched'))} 人，"
                    f"随后应用入组规则，合计 {integer(status['eligible'])} 人。各源表的匹配人数有重叠。")
        text.append(md_table(["数据源","文件记录数","匹配临床人数"],
                             ((s["name"],integer(s.get("n")),integer(s.get("clinical_linked"))) for s in paired.get("source",[]))))
    if health:
        text.append(f"健康寿命访视的行政截止为 **{health.get('visit_administrative_cutoff','未提供')}**。下表按完整 2016 年 VBS 名册统计各波可观察人数。认知列包括本人和代理访谈。"
                    "各项时钟分析另要求配对和完整协变量，实际分母见健康寿命模型表。")
        text.append(md_table(["访视波次","有日期访视","ADL","八项慢病完整","正式认知分类","其中代理认知访谈"],
                             ((year,integer(s.get("observed")),integer(s.get("ADL")),integer(s.get("fully_observed_chronic")),
                               integer(s.get("official_LW")),integer(s.get("official_LW_proxy"))) for year,s in sorted(health.get("waves",{}).items()))))
        for key, label in [("adl","ADL"),("lw","认知")]:
            if key in health:
                ep = health[key]
                text.append(f"全 VBS 名册中，基线无{label}困难者 {integer(ep.get('baseline_at_risk'))} 人，有有效结局观察时间者 {integer(ep.get('positive_time'))} 人，"
                            f"新发 {integer(ep.get('incident'))} 人、竞争死亡 {integer(ep.get('competing_death'))} 人、末次访视删失 {integer(ep.get('censored_at_last_observation'))} 人，"
                            f"基线风险人群中无正随访时间者 {integer(ep.get('at_risk_without_time'))} 人。")
    missing = [x for x in CONTROLS if x not in controls]
    if missing:
        text.append("缺少下列流程汇总来源，相应筛选、版本或访视明细暂缺。 " + "、".join(f"`{x}`" for x in missing))
    return text


def dnam_proxy_report(tables):
    proxy = tables["dnam_proxy_organ"]
    skipped = tables["dnam_proxy_unavailable"]
    text = ["## 器官分数与甲基化替代指标",
            "这些指标由 DNA 甲基化推算。偏相关调整年龄、性别和全身临床分数。对预定的对应器官分别给出关联 q 和匹配优势 q。"
            "匹配优势以对应相关减去各非对应器官相关绝对值后的最小差衡量，单侧家庭聚类检验经 BH 校正。"
            "判断匹配优势时看差值方向及匹配优势 q。单独的关联 q 只反映该指标与该器官成分的关联。"]
    if not proxy.empty:
        matched = proxy[proxy.matched.astype(str).str.lower().eq("true")]
        text.append(md_table(["甲基化替代指标","预定器官","人数","家庭","偏相关","95% 区间","关联 BH q","最小匹配优势","匹配优势 BH q"],
                             ((named(r.marker),named(r.panel),integer(r.n),integer(r.households),num(r.partial_r),
                               ci(r.ci95_low,r.ci95_high),prob(r.association_q_BH),num(r.min_matched_advantage),prob(r.dominance_q_BH))
                              for r in matched.itertuples())))
    else:
        text.append("甲基化替代指标比较无可用聚合效应表。")
    if not skipped.empty:
        reasons = {"matched organ score unavailable":"对应器官无评分", "no nonmatched organ available":"缺少可比较的非对应器官",
                   "surrogate or all organ scores unavailable":"替代指标或器官评分不可用",
                   "fewer than 40 paired participants or two households":"不足 40 名配对者或不足两个家庭",
                   "rank-deficient age, sex and whole-body adjustment":"调整模型协变量不满秩",
                   "constant adjusted surrogate or organ; common comparison unavailable":"调整后数值恒定，共同样本比较不可估计"}
        text.append(md_table(["缺项指标","器官","原因"],((named(r.marker),named(r.panel) if pd.notna(r.panel) else "—",
                                                      reasons.get(r.reason,r.reason)) for r in skipped.itertuples())))
        if (skipped.panel.eq("cardiovascular") & skipped.reason.eq("matched organ score unavailable")).any():
            text.append("HRS 同次抽血面板缺少心血管组评分，因此无法用 DNAm PAI-1 或 DNAm 肾上腺髓质素验证该组的匹配优势。")
    return text


def generate(folder, synthetic=False, output_dir=None):
    status = json.loads((folder/"run_status.json").read_text(encoding="utf-8"))
    expected = "synthetic" if synthetic else "hrs"
    if status["mode"] != expected:
        raise ValueError(f"Expected mode {expected}; observed {status['mode']}. For synthetic data request --synthetic.")
    if (status["n_folds"] != 5 or status["household_split_violations"] != 0
            or not re.fullmatch(r"[0-9a-f]{64}",status["input_hash"])
            or status["bootstrap_refits"] < 2
            or (not synthetic and status["bootstrap_refits"] != 2000)):
        raise ValueError("Unverified fold/replicate metadata or invalid input hash")
    tables = {}
    for name in TABLES:
        path = folder/(name + ".csv")
        tables[name] = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    controls = {name: json.loads((folder/name).read_text(encoding="utf-8-sig")) for name in CONTROLS if (folder/name).is_file()}
    disc = tables["discrimination"]
    if disc.empty:
        raise ValueError("Missing the aggregate discrimination results")
    primary_rows = matching(disc, analysis="primary", clock=PRIMARY)
    primary_tests = matching(tables["LRT"], analysis="primary", clock=PRIMARY)
    can_judge = False
    if len(primary_rows) == 1 and len(primary_tests) == 1:
        row = primary_rows.iloc[0]
        bounds = [row.get("lower", np.nan), row.get("upper", np.nan)]
        can_judge = (row.get("successful") == row.get("requested") == status["bootstrap_refits"]
                     and np.isfinite(bounds).all()
                     and pd.notna(primary_tests.iloc[0].get("H2")))
    if can_judge != bool(status["primary_complete"]):
        raise ValueError("Primary result is inconsistent with the run status")
    counts = disc[["n","families","deaths","deaths_3y"]]
    if (counts.isna().any().any() or (counts < 0).any().any() or (counts % 1 != 0).any().any()
            or (counts.n > status["eligible"]).any() or (counts.families > counts.n).any()
            or (counts.deaths > counts.n).any() or (counts.deaths_3y > counts.deaths).any()):
        raise ValueError("Inconsistent comparison denominators")
    title = "HRS甲基化同人验证结果（合成数据演练）" if synthetic else "HRS甲基化同人验证结果"
    report = ["# " + title]
    if synthetic:
        report.append("**本文件的所有数字均来自合成数据，仅用于验证报告生成程序。**")
    report += [
        "## 同一批人的全因死亡比较",
        f"临床评分输入表中共有 {integer(status['source_rows'])} 行，应用 DNA 甲基化质控、抽血时年龄 ≥50 岁、临床分数、性别和有效随访等入组条件后，保留 {integer(status['eligible'])} 人、{integer(status['families'])} 户。每项比较再限制为相应时钟可用的同一批人。",
        *primary_report(disc,tables["HR"],tables["LRT"]),
        *flow_report(controls,status),
        "## 其他甲基化时钟及敏感性分析",
        "各时钟配对保留该时钟有值者，所有时钟齐全分析使用共同完整样本。下表的差值均按临床减对应甲基化时钟计算。仅主要比较和抽样设计敏感性安排了配对区间估计。",
        discrimination_table(disc[disc.analysis.ne("primary")]),
        "## 健康寿命",
        "模型同时纳入临床分数和相应甲基化时钟，并调整年龄、性别和教育。失能及认知障碍分析排除基线已有该结局者，以死亡作为竞争事件。慢病模型按观察时长估计新发诊断数的发生率比。下表给出主要对手 GrimAge2 的每标准差效应、家庭数和实际分母。慢病栏的事件数累计诊断次数，其他栏累计发生结局的人数。",
    ]
    secondary = matching(tables["healthspan"],clock=PRIMARY)
    report.append(md_table(["结局","模型","变量","人数","家庭","事件","效应比","95% 区间","p","BH q"],
                           ((named(r.endpoint),named(r.model),named(r.term),integer(r.n),integer(r.households),integer(r.events),
                             num(r.ratio),ci(r.lower,r.upper),prob(r.p),prob(r.q_BH)) for r in secondary.itertuples())))
    report += ["## 器官风险成分",
               "在 GrimAge2 同人组中，年龄、性别和所有可计算器官组共同进入线性回归。系数表示临床器官成分每增加一个原始单位所对应的 GrimAge2 变化量。"]
    organs = matching(tables["organ_regression"],clock=PRIMARY)
    report.append(md_table(["器官成分","人数","系数","家庭聚类 95% 区间","BH q"],
                           ((named(r.organ),integer(r.n),num(r.estimate),ci(r.lower,r.upper),prob(r.q_BH)) for r in organs.itertuples())))
    ref = matching(tables["VBS_reference_matrix"],group=PRIMARY)
    if not ref.empty:
        ref = ref[ref.panel.eq(ref.expected)]
    report.append("器官组与独立实测参照的偏相关调整年龄、年龄平方、性别及全身临床分数。匹配优势检验考察对应组是否强于所有其他可比较组。该分析与既有 VBS 人群重叠。")
    report.append(md_table(["实测参照","匹配器官","人数","家庭","偏相关","95% 区间","关联 BH q","匹配优势 BH q"],
                           ((named(r.target),named(r.expected),integer(r.n),integer(r.households),num(r.partial_r),
                             ci(r.ci95_low,r.ci95_high),prob(r.association_q_BH),prob(r.dominance_q_BH)) for r in ref.itertuples())))
    skipped = matching(tables["VBS_reference_unavailable"],clock=PRIMARY)
    for skip in skipped.itertuples():
        reason = {"matched organ has no score":"缺少对应器官组评分","reference analyte unavailable":"缺少该项独立实测参照"}.get(skip.reason,skip.reason)
        report.append(f"{named(skip.marker)} 的对应器官比较不可评估，原因是{reason}。")
    report.extend(dnam_proxy_report(tables))
    report.append("## 可评价范围与核查")
    availability = tables["availability"]
    if not availability.empty:
        report.append(md_table(["分析或时钟","可用人数","缺失或限制"],
                               ((named(r.analysis),integer(r.n),re.sub(r"^missing (\d+)$",r"缺失 \1 人",str(r.reason))) for r in availability.itertuples())))
    report += ["本报告依据本地既有分析方案。主要结论针对 HRS 2016 甲基化与临床配对的 ≥50 岁样本。Harrell C 和 Uno C 衡量风险排序，绝对死亡风险校准需另行评价。次要分析按各输出表指定的检验族作 BH 校正。认知障碍含 CIND 和痴呆的首次访视记录。单次临床指标、甲基化时钟之间的关联不能确定因果方向。",
               f"分析输入 SHA-256 为 `{status['input_hash']}`，家庭跨折次数为 {status['household_split_violations']}。"]
    target = output_dir or folder
    target.mkdir(parents=True,exist_ok=True)
    if target.resolve() == folder.resolve():
        links = ["[运行汇总](run_status.json)"] + [f"[{name}]({name}.csv)" for name in TABLES if not tables[name].empty]
        links += [f"[{name}]({name})" for name in CONTROLS if name in controls]
        report.append("全部时钟的逐项比较、健康寿命和参照结果可在聚合源表核对。 " + " · ".join(links))
    basename = ("合成演练_" if synthetic else "") + "HRS甲基化同人验证结果.md"
    path = target/basename
    path.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    p.add_argument("--synthetic",action="store_true")
    p.add_argument("--output-dir",type=Path)
    args = p.parse_args()
    print(generate(args.folder,args.synthetic,args.output_dir))
