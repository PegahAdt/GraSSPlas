"""Preprocess one sample using explicit input and output paths."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile


def combine(pfam_path, mob_path, output):
    """Join chromosome calls from Pfam to MOB plasmid calls on contig ID."""
    def calls(path, column):
        with path.open(newline="") as stream:
            result = {}
            for row in csv.DictReader(stream, delimiter="\t"):
                key = row["contig"]
                if key in result:
                    raise ValueError(f"Duplicate contig {key} in {path}")
                result[key] = row[column]
            return result
    chrom = calls(pfam_path, "chromosome")
    plasmid = calls(mob_path, "plasmid")
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("contig", "chromosome", "plasmid"))
        writer.writerows((key, value, plasmid[key]) for key, value in chrom.items() if key in plasmid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    ops = parser.add_subparsers(dest="operation", required=True)
    join = ops.add_parser("combine")
    join.add_argument("pfam", type=Path)
    join.add_argument("mob", type=Path)
    filt = ops.add_parser("filter")
    filt.add_argument("script", type=Path, help="Locally recovered historical filter_GFA.py")
    filt.add_argument("input_gfa", type=Path)
    filt.add_argument("--minimum", type=int, default=100)
    for op in (filt, join):
        op.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.operation == "combine":
        combine(args.pfam, args.mob, args.output)
    else:
        if args.output.exists():
            parser.error("Output already exists")
        with tempfile.TemporaryDirectory(dir=args.output.parent) as d:
            staged = Path(d) / "filtered.gfa"
            subprocess.run([sys.executable, str(args.script), str(args.input_gfa), str(args.minimum),
                            str(staged)], check=True)
            staged.rename(args.output)
