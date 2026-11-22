"""Fetch NHANES public XPT inputs directly from NCHS. Verify headers, IDs and record hashes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sources"
FILES = {1999: ["SSCYST_A", "SSBNP_A", "SSTROP_A"], 2001: ["SSCYST_B"],
         2009: ["GLU_F"], 2011: ["GLU_G"], 2013: ["INS_H", "GLU_H"],
         2015: ["INS_I", "GLU_I"], 2017: ["INS_J", "GLU_J", "LUX_J"]}


def download(pair):
    year, stem = pair
    path = SOURCE / (stem + ".xpt")
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}.xpt"
    if not path.exists():
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=(10, 40))
                r.raise_for_status()
                if not r.content.startswith(b"HEADER RECORD"):
                    raise ValueError(f"Response is not SAS transport data: {stem}")
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(path)
                break
            except (ValueError, requests.RequestException):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    data = pd.read_sas(path, format="xport")
    if data.SEQN.isna().any() or not data.SEQN.is_unique:
        raise ValueError(f"Invalid identifiers in {stem}")
    return dict(file=path.name, url=url, documentation=url[:-3]+"htm", bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), n=len(data), columns=list(data))


if __name__ == "__main__":
    SOURCE.mkdir(exist_ok=True)
    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(download, [(year,stem) for year,files in FILES.items() for stem in files]))
    old = SOURCE / "manifest.json"
    if old.exists():
        assert results == json.loads(old.read_text(encoding="utf8")), "Stored manifest differs from source"
    old.write_text(json.dumps(results, indent=2), encoding="utf8")
    for row in results:
        print(row["file"], row["n"], row["columns"], flush=True)

