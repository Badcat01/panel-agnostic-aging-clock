"""Integration smoke test for encrypted releases, harmonized mappings, and aggregate-only publication."""
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyzipper

import researcher_run as runner

sys.path.insert(0, str(runner.PROTOCOL))
from prepare import source as load_source


@contextmanager
def must_fail(exception):
    try:
        yield
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def sample(fields):
    d = pd.DataFrame({"HHID":["000001","000001","000002","000003"], "PN":["001","002","001","001"]})
    for j, field in enumerate(fields):
        d[field] = np.arange(len(d),dtype=float)+j+1.
    return d


def write_source(path, fields):
    frame = sample(fields)
    if path.suffix == ".dta":
        frame.to_stata(path, write_index=False, version=118)
    else:
        frame.to_csv(path,index=False)
    return frame


def encrypted_releases(downloads):
    releases = {}
    for i, (name, (_, dest)) in enumerate(runner.RELEASES.items()):
        columns = list(runner.BASE.values()) if i==0 else list(runner.SUPPLEMENT.values()) if i==1 else []
        raw = BytesIO()
        sample(columns).to_stata(raw, write_index=False, version=118)
        archive = downloads/name
        with pyzipper.AESZipFile(archive,"w",compression=pyzipper.ZIP_DEFLATED) as z:
            if i < 2:
                z.setencryption(pyzipper.WZ_AES)
                z.setpassword([b"synthetic-first", b"synthetic-second"][i])
            z.writestr(Path(dest).name, raw.getvalue())
        releases[name] = runner.sha(archive),dest
    return releases


def check_archives(downloads, root, releases):
    with patch.object(runner,"getpass",return_value="incorrect"), must_fail(RuntimeError):
        runner.archives(downloads,root/"wrong")
    assert not list((root/"wrong").rglob("*.dta")) and not any(p.is_file() for p in (root/"wrong").rglob("*"))
    first = next(iter(releases))
    with patch.dict(runner.RELEASES,{first:("0"*64,releases[first][1])}), must_fail(ValueError):
        runner.archives(downloads,root/"bad_hash")
    assert not (root/"bad_hash").exists()
    with patch.object(runner,"getpass",side_effect=["synthetic-first", "synthetic-second"]) as password:
        files = runner.archives(downloads,root/"source")
        assert password.call_count == 2
    one,two = files["EPIGENETIC.zip"],files["EPIGENETICsup.zip"]
    assert one.name.casefold()==two.name.casefold() and one.parent!=two.parent and runner.sha(one)!=runner.sha(two)
    with patch.object(runner,"getpass",side_effect=AssertionError("Unexpected cached password prompt")):
        assert runner.archives(downloads,root/"source") == files
    existing = root/"foreign"/releases[first][1]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing fixture")
    with patch.object(runner,"getpass",return_value="synthetic-first"),must_fail(FileExistsError):
        runner.archives(downloads,root/"foreign")
    assert existing.read_bytes()==b"existing fixture" and not existing.with_suffix(".sha256.json").exists()
    return files


def check_mapping(root,files):
    work = root/"private"
    (work/"tracker").mkdir(parents=True)
    (work/"health").mkdir()
    tr_fields = ["age","sex","time_yr","event","time_lower_yr","time_upper_yr"]
    des_fields = ["weight","stratum","psu"]
    tr, des = work/"tracker/tracker.csv.gz",work/"tracker/design.csv.gz"
    write_source(tr,tr_fields)
    write_source(des,des_fields)
    controls = {"sources":[runner.source(role,path,{k:k for k in fields}) for role,path,fields in
                           [("tracker",tr,tr_fields),("design",des,des_fields)]]}
    runner.write_json(work/"tracker/tracker_source_map.json",controls)
    vbs, cytokines=work/"vbs.dta",work/"cytokines.dta"
    write_source(vbs,["PCYSC","PNTBNPE","PCMVGE"])
    write_source(cytokines,["PIL6","PTNFR1"])
    write_source(work/"health/health.csv.gz",
                 ["education","ti_adl","st_adl","ti_lw","st_lw","new_conditions","years_at_new_conditions"])
    mapping = json.loads(runner.build_map(work,files,vbs,cytokines).read_text(encoding="utf8"))
    columns = set()
    count_clocks = []
    for spec in mapping["sources"]:
        tab, passed = load_source(spec,work)
        assert tab.index.tolist()==["000001-001","000001-002","000002-001","000003-001"]
        assert not (columns & set(tab))
        columns.update(tab)
        if spec["role"]=="dnam":
            assert passed.all()
            count_clocks.append(sum(k.startswith("clock_") for k in tab))
            assert tab.iloc[0,0]==1.
        if spec["role"]=="reference":
            assert np.isclose(tab.iloc[1,0],np.log(2.))
    assert count_clocks==[13,2]
    assert {c for c in columns if c.startswith("cell_")}=={"cell_GR","cell_NK","cell_B","cell_CD4","cell_CD8","cell_MO"}
    assert {c for c in columns if c.startswith("proxy_")}=={
        "proxy_cystatin","proxy_b2m","proxy_leptin","proxy_loga1c","proxy_logcrp","proxy_gdf15","proxy_pai1","proxy_adm"}
    assert {"clock_GrimAge2","clock_DunedinPACE","st_lw","st_adl","new_conditions",*des_fields,*tr_fields} <= columns
    with must_fail(ValueError):
        load_source(dict(mapping["sources"][0],sha256="0"*64),work)


def check_publication(root):
    out = root/"outputs"
    out.mkdir()
    report = "HRS甲基化同人验证结果.md"
    for name in (report,"discrimination.csv","paired.csv.gz.json","paired.csv.gz","participant_ids.csv","registered_fields.json"):
        (out/name).write_text("fixture",encoding="utf8")
    publication = root/"published"/"run1"
    runner.publish_aggregate(out,publication)
    assert {f.name for f in publication.iterdir()}=={report,"discrimination.csv","paired.csv.gz.json"}
    (out/report).unlink()
    with must_fail(FileNotFoundError):
        runner.publish_aggregate(out,root/"published"/"run2")
    assert not (root/"published"/"run2").exists()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        downloads=root/"downloads"
        downloads.mkdir()
        releases=encrypted_releases(downloads)
        with patch.object(runner,"RELEASES",releases):
            files=check_archives(downloads,root,releases)
            check_mapping(root,files)
        check_publication(root)
    print("Synthetic release integration passed: 6 archives, 15 clocks, 6 cells, 8 proxies, design, health and aggregate-only export.")
