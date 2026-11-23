"""Describe plasmid F1 by recorded or nominal seed, separately within each intervention."""
import argparse
import csv
from collections import defaultdict
from statistics import mean, stdev
import sys

from validate_results import ROOT, rows


def summarize(group_by, threshold, stream):
    fits = rows(ROOT / "experiments/fall/fits.csv")
    seeds = {(f["family"], f["folder_seed"], f["run"], f["method"]): f[group_by] for f in fits}
    groups = defaultdict(list)
    for r in rows(ROOT / "results/fall_metrics.csv"):
        if r["threshold"] == str(threshold) and r["mol_type"] == "plasmid":
            method = r["method"].split(":")[1]
            seed = seeds[r["family"], r["seed"], r["run"], method]
            groups[r["family"], method, seed].append(float(r["f1"]))
    writer = csv.writer(stream)
    writer.writerow(["family", "method", group_by, "fits", "mean_plasmid_f1", "sample_sd"])
    for key, values in sorted(groups.items()):
        writer.writerow([*key, len(values), mean(values), stdev(values) if len(values) > 1 else ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-by", choices=["saved_seed", "folder_seed"], default="saved_seed")
    parser.add_argument("--threshold", type=int, choices=[100, 1000], default=1000)
    args = parser.parse_args()
    summarize(args.group_by, args.threshold, sys.stdout)
