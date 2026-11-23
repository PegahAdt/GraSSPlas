"""Recover selected processing and experiment files from an authorized PLASMIDS checkout."""
import argparse
import csv
import hashlib
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def recover(root, dest):
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / "sources"
        with (ROOT / "docs/local_sources.csv").open() as stream:
            for row in csv.DictReader(stream):
                rel = Path(row["relative_to_PLASMIDS"])
                content = (root / rel).read_bytes()
                if hashlib.sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError("Different historical revision: " + str(rel))
                target = staged / rel
                target.parent.mkdir(exist_ok=True, parents=True)
                target.write_bytes(content)
        staged.rename(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    recover(args.lab_root, args.destination)
