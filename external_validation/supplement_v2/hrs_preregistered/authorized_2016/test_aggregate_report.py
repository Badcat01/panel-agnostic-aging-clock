"""Run the aggregate-only Markdown report against simulated summaries with/without ancillary tables."""
import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import pandas as pd

from report_aggregate import TABLES, generate


def verify(folder, fixture):
    for name in ["run_status.json", *[t + ".csv" for t in TABLES]]:
        if (fixture/name).is_file():
            copyfile(fixture/name, folder/name)
    try:
        generate(folder)
    except ValueError as e:
        assert "--synthetic" in str(e)
    else:
        raise AssertionError("Accidentally labeled a simulated run as real")
    sparse = generate(folder, synthetic=True).read_text(encoding="utf8")
    assert "合成数据演练" in sparse.splitlines()[0] and "缺少下列流程汇总来源" in sparse
    run = json.loads((folder/"run_status.json").read_text())
    controls = {
        "paired.csv.gz.json":{"prepared_sha256":run["input_hash"],"n_clinical":320,"n_DNAm_QC_matched":290,
                              "tracker_cutoff":"2023-12-31","mapping_sha256":"fixture","source":[{"name":"simulated_clock.dta","n":290,"clinical_linked":288}]},
        "tracker.csv.gz.json":{"cutoff":"2023-12-31","n_VBS":360,"valid_time":330,"included_deaths":185,
                               "death_missing_date":3,"tracker_version":"合成版 Tracker"},
        "health_verification.json":{"tracker_administrative_cutoff":"2023-12-31","visit_administrative_cutoff":"2023-04-30",
                                    "waves":{str(y):{"observed":300,"ADL":285,"fully_observed_chronic":280,"official_LW":282,"official_LW_proxy":32}
                                             for y in (2016,2018,2020,2022)}},
        "release_versions.json":{"official_osf_registration":False,"tracker_cutoff":"2023-12-31","mapping_sha256":"fixture"}}
    for name, content in controls.items():
        (folder/name).write_text(json.dumps(content,ensure_ascii=False),encoding="utf8")
    pd.DataFrame([{"marker":"proxy_cystatin","panel":"kidney","matched":True,"n":240,"households":127,"partial_r":.3,
                   "ci95_low":.1,"ci95_high":.5,"association_q_BH":.02,"min_matched_advantage":.12,"dominance_q_BH":.15}]).to_csv(
                       folder/"dnam_proxy_organ.csv",index=False)
    pd.DataFrame([{"marker":"proxy_pai1","panel":"cardiovascular","reason":"matched organ score unavailable"}]).to_csv(
        folder/"dnam_proxy_unavailable.csv",index=False)
    filled = generate(folder,synthetic=True).read_text(encoding="utf8")
    assert "缺少下列流程汇总来源" not in filled
    for token in ("| 2022 | 300 | 285 | 280 | 282 | 32 |","simulated_clock.dta","2023-12-31","DNAm 胱抑素 C",
                  "心血管组评分","最小匹配优势","0.1500","H1","H2","公开 OSF"):
        assert token in filled, token
    controls["paired.csv.gz.json"]["prepared_sha256"]="wrong"
    (folder/"paired.csv.gz.json").write_text(json.dumps(controls["paired.csv.gz.json"]),encoding="utf8")
    try:
        generate(folder,synthetic=True)
    except ValueError as e:
        assert "current paired run" in str(e)
    else:
        raise AssertionError("Cross-run manifest was accepted")
    return {"type":"synthetic aggregate results only","simulated_mode_enforced":True,
            "denominators_and_H1_H2":True,"missing_controls_reported":True,"wave_and_proxy_counts":True,
            "exclusions_and_proxy_dominance":True,"mixed_run_metadata_rejected":True}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=Path(__file__).resolve().parents[1]/"synthetic/tested")
    with TemporaryDirectory() as d:
        print(json.dumps(verify(Path(d), parser.parse_args().fixture_dir),indent=2))
