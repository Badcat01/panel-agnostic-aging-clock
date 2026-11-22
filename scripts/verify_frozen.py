"""Check the released FT-Mask bundle and reference files against their SHA-256 digests."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "nhanes_gpu_20261121"
BUNDLE = RUN / "bundle"
SCALE = ROOT / "external_validation" / "nhanes3_organ" / "organ_score_reference.json"


def verify():
    records = (RUN / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    if not records:
        raise ValueError("The SHA-256 manifest is empty")
    for record in records:
        digest, relative = record.split("  ", 1)
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen artifact failed SHA-256 verification: {relative}")
    return len(records)


if __name__ == "__main__":
    print(f"Verified {verify()} frozen files")
