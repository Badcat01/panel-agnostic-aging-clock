"""Produce a concise Chinese readout directly from the comparative output tables."""
import pandas as pd
from design import OUT


def make_readout(d,meta):
    primary=d[d.setting=="primary"]
    if primary.empty:return
    labels={"full":"完整 43","nhanes3":"NHANES III 40","hrs":"HRS 31","mimic":"MIMIC 19",
            "charls":"CHARLS 18","elsa":"ELSA 12","share":"SHARE 6"}
    models=["M1_nomask","M2_panel","M3_XGB_raw","M3_XGB_aug","M4_RSF","M5_MLP","M6_MICE","ridge_panel"]
    rows=["# 掩码消融与强基线对照","",
          "NHANES 时间留出集有 25,614 人、1,977 例死亡。六种外部面板都在这批患者上模拟。下表为各模型的 Harrell C，越高表示死亡风险排序越好。",
          "","| 模拟面板 | 冻结 M0 | 无遮盖 M1 | 专训 M2 | XGB | 遮盖 XGB | RSF | 缺失指示 MLP | MICE 岭 Cox | 专训岭 Cox |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for panel,label in labels.items():
        s=primary[primary.panel==panel].set_index("model")
        if s.empty:continue
        same=s[s.n==25614]
        numbers=[f"{same.M0_C.iloc[0]:.4f}"]
        numbers+= [f"{s.loc[m,'comparator_C']:.4f}" if m in same.index else "—" for m in models]
        rows.append("| "+" | ".join([label,*numbers])+" |")
    rows.extend(["","M2 和 RSF 按六个目标面板训练，因此完整 43 项一行的这两格留空。",
                 "","## 冻结模型与面板专训模型的差值","",
                 "下表按 M0 减 M2 计算。单侧 95% 下界大于 −0.003 时达到该口径的非劣效；等效还要求上界低于 +0.003。",
                 "","| 面板 | ΔC | 双侧 95% 区间 | 单侧 95% 下界 | 90% 中央区间（双单侧各 0.05） |",
                 "|---|---:|---:|---:|---:|"])
    for panel,label in labels.items():
        s=primary[(primary.panel==panel)&(primary.model=="M2_panel")].dropna(subset=["ci_low"])
        if len(s):
            r=s.iloc[0]
            low90,high90=r.get("ci90_low"),r.get("ci90_high")
            rows.append(f"| {label} | {r.delta_M0_minus_comparator:+.4f} | {r.ci_low:+.4f} 至 {r.ci_high:+.4f} | {low90:+.4f} | {low90:+.4f} 至 {high90:+.4f} |")
    if primary[primary.model.eq("M2_panel")].panel.nunique()==6:
        rows.extend(["","六组专训模型的 C 均高于冻结模型。以单侧 95% 下界判断，两个面板（NHANES III、HRS）的差距落在可接受范围，四个面板未证明非劣。用单侧 97.5% 下界时仅 NHANES III 达标。与无掩码 M1 对比时，四个 ≤20 项的面板中，MIMIC 和 SHARE 两组的配对区间高于零。"])
    rows.extend(["","## 真实队列中冻结模型与专训模型的差值","",
                 "各人在该队列固定目标项目内仍保留原有缺失。MIMIC 的固定 19 项分析含 129,727 人，直接应用原全部可测指标的正式验证含 130,242 人。",
                 "","| 队列 | 同人样本 / 死亡 | M0 | M2 | ΔC（双侧 95% 区间） |","|---|---:|---:|---:|---:|"])
    for panel,label in labels.items():
        s=d[(d.setting=="external")&(d.panel==panel)&(d.model=="M2_panel")].dropna(subset=["ci_low"])
        if len(s):
            r=s.iloc[0]
            rows.append(f"| {label} | {r.n:,} / {r.deaths:,} | {r.M0_C:.4f} | {r.comparator_C:.4f} | {r.delta_M0_minus_comparator:+.4f}（{r.ci_low:+.4f} 至 {r.ci_high:+.4f}） |")
    rows.extend(["","## 传统衰老指标在完整面板中的同人结果","",
                 "| 对手 | 同人样本 / 死亡 | 冻结 M0 | 对手 | ΔC（条件于计算所需项目齐备） |",
                 "|---|---:|---:|---:|---:|"])
    for m in ["M7_KDM","M7_PhenoAge","M7_HD","M7_age_HD"]:
        s=primary[(primary.panel=="full")&(primary.model==m)]
        if len(s):
            r=s.iloc[0]
            rows.append(f"| {m} | {r.n:,} / {r.deaths:,} | {r.M0_C:.4f} | {r.comparator_C:.4f} | {r.delta_M0_minus_comparator:+.4f} |")
    rows.extend(["","## 社区外部队列随机效应汇总","",
                 "| 对手 | 可纳入队列数 | M0 − 对手 ΔC（95% CI） | I² |",
                 "|---|---:|---:|---:|"])
    for r in meta.itertuples():
        rows.append(f"| {r.model} | {r.k} | {r.delta:+.4f}（{r.low:+.4f} 至 {r.high:+.4f}） | {r.I2:.1f}% |")
    rows.extend(["","单队列估计和区间基于患者配对的 2,000 次 bootstrap。纳入独立社区队列时保留各自病例和随访情况，跨队列平均的置信区间同时考虑队列间异质性。",
                 "","[完整指标与区间](comparisons.csv) · [留出集配对热图](temporal_paired_heatmap.png) · [真实队列配对热图](external_paired_heatmap.png) · [随机效应森林图](external_random_effects.png) · [随机面板丰富度](random_panel_richness.png) · [方法与复算](复算说明.md) · [各方法评分覆盖](scoring_coverage.csv)"])
    file=OUT/"模型消融结果.md"
    tmp=file.with_suffix(".tmp")
    tmp.write_text("\n".join(rows)+"\n",encoding="utf8")
    tmp.replace(file)
