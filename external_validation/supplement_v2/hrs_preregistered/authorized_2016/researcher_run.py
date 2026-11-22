"""Local researcher's entry point for the authorized HRS 2016 paired analysis."""
import argparse
from datetime import datetime
from getpass import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pyzipper

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent
WORKSPACE = HERE.parents[3]
RELEASES = {
    "EPIGENETIC.zip": ("c7a4dad05e1ce63ae4d09f2e3fa442a6e861e8774eeb7de8d2bec84fadb30a06", "original/EPICLOCKA_R.dta"),
    "EPIGENETICsup.zip": ("7763e787a6deca3546d364eb16ee05e437e57eddc2694e652709554aeaa929fd", "supplement/epiclocka_r.dta"),
    "trk2022v2.zip": ("f1b1b0de96f779c12a010e0c556c0f48a0f16021cd87287b78803170e1b0c361", "tracker/trk2022tr_r.dta"),
    "Trk22_WeightsV1.zip": ("d328f7cc5379704afde2cb8c4aaed8a8d42bc3449943866068cbbdc868f5729c", "tracker/wgts2022tr_r.dta"),
    "LangaWeir2022v1.zip": ("9e875f6af991011c22725a6b8519e18f13bc23f5678627388752fd9d04fcdf5f", "langa/cogfinalimp_9522wide.dta"),
    "randhrs1992_2022v2_STATA.zip": ("b307a0fc0ce9491c81022116b85fce39e86d42facc938ee023393608e9419eeb", "rand/randhrs1992_2022v2.dta"),
}

# Variable names are from the Stata dictionaries of the two downloaded releases.
BASE = {"Horvath": "HORVATH_DNAMAGE", "Hannum": "HANNUM_DNAMAGE", "DNAmPhenoAge": "LEVINE_DNAMAGE",
        "SkinBlood": "HORVATHSKIN_DNAMAGE", "Lin": "LIN_DNAMAGE", "Weidner": "WEIDNER_DNAMAGE",
        "VidalBralo": "VIDALBRALO_DNAMAGE", "Yang": "YANG_DNAMAGE", "Zhang_2017": "ZHANG_DNAMAGE",
        "Bocklandt": "BOCKLANDT_DNAMAGE", "Garagnani": "GARAGNANI_DNAMAGE", "GrimAge": "DNAMGRIMAGE",
        "DunedinPoAm": "MPOA"}
SUPPLEMENT = {"clock_GrimAge2": "DNAmGrimAge2", "clock_DunedinPACE": "PACE",
              **{"cell_" + name: name for name in ("GR", "NK", "B", "CD4", "CD8", "MO")},
              "proxy_cystatin": "DNAmCystatinC", "proxy_b2m": "DNAmB2M", "proxy_leptin": "DNAmLeptin",
              "proxy_loga1c": "DNAmlogA1C", "proxy_logcrp": "DNAmlogCRP", "proxy_gdf15": "DNAmGDF15",
              "proxy_pai1": "DNAmPAI1", "proxy_adm": "DNAmADM"}


def sha(path):
    obj = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            obj.update(chunk)
    return obj.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def archives(downloads, storage, password=None):
    files = {}
    password_from_window = password is not None
    for name, (expected, dest) in RELEASES.items():
        archive = downloads / name
        if sha(archive) != expected:
            raise ValueError(f"发布包校验失败，请确认下载版本：{name}")
        target = storage / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = target.with_suffix(".sha256.json")
        if target.exists() and receipt.exists():
            cached = json.loads(receipt.read_text(encoding="utf8"))
            if cached.get("zip") == expected and cached.get("member") == sha(target):
                files[name] = target
                continue
        with pyzipper.AESZipFile(archive) as z:
            matches = [m for m in z.infolist() if Path(m.filename).name == target.name]
            if len(matches) != 1:
                raise ValueError(f"找不到唯一的官方 Stata 成员：{name}")
            member = matches[0]
            encrypted = bool(member.flag_bits & 1)
            cached_password = password
            prompt = f"请输入 HRS Special Access Downloads 中 {name} 对应的解压口令："
            if encrypted and password is None:
                password = getpass(prompt).encode()
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                pending = Path(tmp.name)
            try:
                try:
                    stream = z.open(member, pwd=password if encrypted else None)
                except RuntimeError as exc:
                    if not (encrypted and cached_password is not None and "password" in str(exc).lower()):
                        raise
                    if password_from_window:
                        raise ValueError(f"{name} 的本机解压口令未通过校验。") from None
                    password = getpass(prompt).encode()
                    stream = z.open(member, pwd=password)
                with stream as source, pending.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=2**20)
                if pending.stat().st_size != member.file_size:
                    raise IOError(f"解压文件长度不符：{name}")
                if target.exists():
                    if sha(pending) != sha(target):
                        raise FileExistsError(f"已有文件内容与此发布包不同，请另选受控目录：{target.name}")
                else:
                    # Publish only a complete file, without replacing a competing extraction.
                    if os.name == "nt":
                        pending.rename(target)  # Windows rename fails if target already exists.
                    else:
                        os.link(pending, target)
                write_json(receipt, {"zip": expected, "member": sha(target)})
            finally:
                pending.unlink(missing_ok=True)
        files[name] = target
    return files


def source(role, file, columns, **details):
    return dict(role=role, file=str(file.resolve()), sha256=sha(file), columns=columns, **details)


def find_unique(folder, name):
    found = list(folder.rglob(name))
    if len(found) != 1:
        raise ValueError(f"应只有一份 {name}，目前找到 {len(found)} 份。请用命令行指定输入文件。")
    return found[0]


def build_map(work, files, vbs, cytokines):
    spec = json.loads((work / "tracker/tracker_source_map.json").read_text(encoding="utf8"))
    qc = "Official HRS release from 4,018 samples passing DNAm quality control; retain observed clock values."
    main_refs = {"ref_cystatin": "PCYSC", "ref_ntprobnp": "PNTBNPE", "ref_cmv": "PCMVGE"}
    other_refs = {"ref_il6": "PIL6", "ref_stnfr1": "PTNFR1"}
    health = ("education", "ti_adl", "st_adl", "ti_lw", "st_lw", "new_conditions", "years_at_new_conditions")
    spec["sources"] += [
        source("dnam", files["EPIGENETIC.zip"], {"clock_" + k: v for k, v in BASE.items()}, official_qc_note=qc),
        source("dnam", files["EPIGENETICsup.zip"], SUPPLEMENT, official_qc_note=qc),
        source("reference", vbs, main_refs, positive_log=list(main_refs)),
        source("reference", cytokines, other_refs, positive_log=list(other_refs)),
        source("health", work / "health/health.csv.gz", {k: k for k in health}),
    ]
    desc = work / "registered_fields.json"
    write_json(desc, spec)
    return desc


def python(script, *args):
    cmd = [sys.executable, "-X", "utf8", script, *args]
    subprocess.run(cmd, check=True)


def publish_aggregate(out, published):
    names = ("discrimination.csv","HR.csv","LRT.csv","availability.csv","partial_correlation.csv","organ_regression.csv",
             "healthspan.csv","healthspan_unavailable.csv","VBS_reference_matrix.csv","VBS_reference_unavailable.csv",
             "dnam_proxy_organ.csv","dnam_proxy_unavailable.csv","run_status.json","release_versions.json",
             "tracker.csv.gz.json","health_verification.json","paired.csv.gz.json","HRS甲基化同人验证结果.md")
    published.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=published.parent) as tmp:
        staging = Path(tmp) / "aggregate"
        staging.mkdir()
        for name in names:
            if (out / name).exists():
                shutil.copyfile(out / name, staging / name)
        if not (staging / "HRS甲基化同人验证结果.md").is_file():
            raise FileNotFoundError("汇总报告未生成。")
        staging.rename(published)


def run(args, password=None):
    print("HRS 2016 甲基化同人比较。正式分析由获准研究者在本机启动。", flush=True)
    vbs = args.vbs.resolve(strict=True)
    cytokines = args.cytokines.resolve(strict=True)
    rscript = shutil.which(args.rscript)
    if rscript is None:
        raise FileNotFoundError(f"Rscript executable not found: {args.rscript}")
    subprocess.run([rscript, "--vanilla", "-e",
                    'stopifnot(all(sapply(c("survival","data.table","jsonlite","digest","R.utils"),requireNamespace,quietly=TRUE)))'],
                   check=True)
    private = args.private.resolve()
    if private == WORKSPACE or WORKSPACE in private.parents:
        raise ValueError("请选择项目工作区以外的受控目录用于逐人中间文件。")
    files = archives(args.downloads, private / "source", password)
    del password
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    work = private / stamp
    work.mkdir(parents=True)
    print(f"本地运行目录 {work}", flush=True)
    tracker_dir = files["trk2022v2.zip"].parent
    python(HERE / "stage_tracker_2022.py", "--vbs", vbs, "--release-dir", tracker_dir,
           "--out", work / "tracker", "--cutoff", args.cutoff)
    python(HERE / "build_health.py", "--vbs", vbs, "--raw-tracker", files["trk2022v2.zip"],
           "--tracker", work / "tracker/tracker.csv.gz", "--rand", files["randhrs1992_2022v2_STATA.zip"],
           "--langa", files["LangaWeir2022v1.zip"], "--out", work / "health")
    spec = build_map(work, files, vbs, cytokines)
    paired = work / "paired.csv.gz"
    python(PROTOCOL / "prepare.py", spec, paired)
    print("开始家庭五折 Cox、2,000 次家庭和 2,000 次抽样设计重抽。", flush=True)
    out = work / "aggregate"
    subprocess.run([rscript, "--vanilla", PROTOCOL / "run.R", paired, out, "hrs", "2000"], check=True)
    subprocess.run([rscript, "--vanilla", PROTOCOL / "secondary.R", paired, out], check=True)
    python(PROTOCOL / "references.py", paired, out)
    python(HERE / "proxy_organ.py", paired, out)
    for src in [work / "tracker/tracker.csv.gz.json", work / "health/health_verification.json", Path(str(paired)+".json")]:
        shutil.copyfile(src, out / src.name)
    write_json(out / "release_versions.json", {"release_sha256": {k:v[0] for k,v in RELEASES.items()},
                 "tracker_cutoff": args.cutoff, "official_osf_registration": False,
                 "mapping_sha256": sha(spec), "clock_fields": dict(**{"clock_"+k:v for k,v in BASE.items()}, **SUPPLEMENT)})
    python(HERE / "report_aggregate.py", out)
    # Export only named aggregate products. The input directory and person-level predictions stay private.
    published = HERE / "汇总结果" / stamp
    publish_aggregate(out, published)
    print(f"汇总报告：{published / 'HRS甲基化同人验证结果.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True, help="Directory containing the six official HRS ZIP releases")
    parser.add_argument("--private", type=Path, required=True, help="Controlled working directory outside this repository")
    parser.add_argument("--vbs", type=Path, required=True, help="Path to hrs2016vbs.sav")
    parser.add_argument("--cytokines", type=Path, required=True, help="Path to vbs16aa.dta")
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable name or path")
    parser.add_argument("--cutoff", default="2023-07-31", help="项目死亡随访截止日，另可给 2022-12-31")
    parser.add_argument("--zip-key-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.zip_key_stdin:
        key = sys.stdin.buffer.readline(4096).rstrip(b"\r\n")
        if not key:
            parser.error("本机窗口未传入解压口令")
        run(args, key)
    else:
        run(args)
