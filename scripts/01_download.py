"""Download the public NHANES files used to build the frozen cohort."""

import argparse
import hashlib
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs/nhanes_gpu_20261121/input_hashes.json"
NHANES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
DNAM = "https://wwwn.cdc.gov/nchs/data/nhanes/dnam/dnmepi.sas7bdat"
MORTALITY = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"


def source_url(path):
    if path.parent.name == "mort":
        return f"{MORTALITY}/{path.name}"
    if path.name == "dnmepi.sas7bdat":
        return DNAM
    suffix = path.stem.rsplit("_", 1)[-1]
    cycle = 1999 + 2 * (ord(suffix) - ord("A")) if len(suffix) == 1 else 1999
    return f"{NHANES}/{cycle}/DataFiles/{path.name}"


def matches(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == metadata["sha256"]


def download(path, url, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "NHANES-FT-Mask-reproduction/1.0"})
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(request, timeout=120) as response:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if not matches(temporary, metadata):
                raise ValueError("The response differs from the recorded NHANES file (size or SHA-256)")
            temporary.replace(path)
            return
        except (OSError, ValueError) as exc:
            if attempt == 2:
                raise
            print(f"Retrying {url} after {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without a network request")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project directory containing data/raw")
    args = parser.parse_args()
    files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    raw = {Path(name): spec for name, spec in files.items() if name.startswith("data/raw/")}
    for relative, spec in raw.items():
        path = args.root / relative
        if matches(path, spec):
            continue
        if args.verify_only or path.exists():
            parser.exit(1, f"Missing or mismatching NHANES file: {path}\n")
        print(f"Downloading {relative.as_posix()}", flush=True)
        download(path, source_url(relative), spec)
    print(f"Verified {len(raw)} public NHANES source files")


if __name__ == "__main__":
    main()
