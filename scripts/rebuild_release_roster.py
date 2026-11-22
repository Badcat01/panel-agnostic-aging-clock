"""Rebuild and verify the original study's local NHANES development roster."""

import hashlib
import json
import tempfile
from pathlib import Path

from train_clock_gpu import get_cohorts, sha
from verify_frozen import ROOT, RUN, SCALE, verify


def main():
    verify()
    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    for filename, digest in protocol["data"].items():
        if sha(ROOT / "data" / "processed" / filename) != digest:
            raise ValueError(f"The reconstructed NHANES file differs from the original: {filename}")
    development, _, _ = get_cohorts()
    ids = development["SEQN"].to_numpy(dtype="<i8").tobytes()
    if hashlib.sha256(ids).hexdigest() != protocol["train_ids_sha256"]:
        raise ValueError("The reconstructed development roster differs from the original")
    expected = json.loads(SCALE.read_text(encoding="utf-8"))["development_roster_sha256"]
    roster = RUN / "development_ids.csv.gz"
    if roster.exists():
        if sha(roster) != expected:
            raise ValueError(f"An existing roster differs from the original: {roster}")
    else:
        with tempfile.TemporaryDirectory(dir=RUN) as directory:
            temporary = Path(directory) / roster.name
            development[["SEQN", "cycle"]].to_csv(
                temporary, index=False, lineterminator="\r\n",
                compression={"method": "gzip", "mtime": 1795223363},
            )
            if sha(temporary) != expected:
                raise ValueError("The roster's gzip encoding differs from the original")
            temporary.replace(roster)
    print(f"Verified NHANES development roster ({len(development)} participants): {roster}")


if __name__ == "__main__":
    main()
