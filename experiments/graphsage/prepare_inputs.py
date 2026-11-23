"""Prepare the GraphSAGE manifest from pinned public inputs or a local data root."""
import argparse
import csv
from pathlib import Path
import subprocess
import tempfile

EXPERIMENT = Path(__file__).resolve().parent
LABELS = "0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6"
GOLD = "eb264d9a45ac278fae37a678e4440cf23145801a"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def public_input(name):
    if name.startswith("Ecoli_GroundTruth/"):
        location = GOLD + ":Data/" + name
    else:
        location = LABELS + ":data/" + name
    return subprocess.check_output(["git", "-C", str(EXPERIMENT), "show", location])


def prepare(destination, existing=None):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    samples = read_rows(EXPERIMENT / "samples.csv")
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staged = Path(tmp) / "prepared"
        staged.mkdir()
        root = existing.resolve() if existing else destination / "data"
        for row in samples:
            for key in list(row)[1:]:
                name = row[key]
                path = root / name
                if any(char in str(path) for char in ",\r\n"):
                    raise ValueError("The historical parser requires comma-free, single-line paths")
                if existing:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                else:
                    target = staged / "data" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(public_input(name))
                row[key] = str(path)
        with (staged / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        staged.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--data-root", type=Path, help="Use existing inputs in place; use the pinned public inputs for exact data")
    args = parser.parse_args()
    prepare(args.destination, args.data_root)
