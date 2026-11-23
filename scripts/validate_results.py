"""Check published aggregates, optionally recount the five original August output sets."""
import argparse
import csv
from collections import Counter, defaultdict
from itertools import zip_longest
from math import isclose
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("plasforest_pls", "plasgraph2_pls", "platon_pls", "rfplasmid_pls", "mlplasmid_pls")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratios(tp, fp, fn):
    def divide(a, b):
        return a / b if b else 0.0
    return divide(tp, tp + fp), divide(tp, tp + fn), divide(2 * tp, 2 * tp + fp + fn)


def recount(directory):
    counts = Counter()
    features, gold, predictions = [rows(directory / name) for name in
                                    ("features.csv", "ground_truth.csv", "y_final.csv")]
    for feature, label, pred in zip_longest(features, gold, predictions):
        require(None not in (feature, label, pred), "Unequal table lengths")
        require(feature["contig"] == label["contig"], "Contig order differs")
        predicted = {name: float(feature[name]) > 0.5 for name in BASELINES}
        predicted["GrassPlas"] = int(pred["y_final"])
        require(predicted["GrassPlas"] in (0, 1), "Nonbinary prediction")
        intersection, union = all(predicted.values()), any(predicted.values())
        predicted.update(intersection=intersection, union=union)
        actual = int(label["Plasmid"])
        require(actual in (0, 1), "Nonbinary gold")
        for cutoff in (0, 100, 1000):
            if float(feature["length"]) >= cutoff:
                for name, call in predicted.items():
                    counts[cutoff, name, ("TN", "FP", "FN", "TP")[2 * actual + int(call)]] += 1
    return counts


def validate(august_root=None):
    august = rows(ROOT / "results/august_metrics.csv")
    expected = {(str(i), str(c), m) for i in range(1, 6) for c in (0, 100, 1000)
                for m in (*BASELINES, "GrassPlas", "intersection", "union")}
    require(len(august) == 120 and {(r["run"], r["threshold"], r["tool"]) for r in august} == expected,
            "Incomplete August metrics")
    observed = {}
    if august_root:
        observed = {str(i): recount(august_root / f"results_{i}") for i in range(1, 6)}
    for r in august:
        tp, fp, fn, tn = (int(r[k]) for k in ("TP", "FP", "FN", "TN"))
        require(tp + fp + fn + tn == (4440 if r["threshold"] == "1000" else 12176), str(r))
        for key, value in zip(("precision", "recall", "f1"), ratios(tp, fp, fn)):
            require(isclose(float(r[key]), value, abs_tol=1e-12), str(r))
        if observed:
            for key in ("TP", "FP", "FN", "TN"):
                require(observed[r["run"]][int(r["threshold"]), r["tool"], key] == int(r[key]), str(r))
    fits = rows(ROOT / "experiments/fall/fits.csv")
    keys = {(r["family"], r["folder_seed"], r["run"], r["method"]) for r in fits}
    require(len(fits) == len(keys) == 210, "Fall fit inventory")
    require(sum(r["saved_seed"] != r["folder_seed"] for r in fits) == 33, "Seed discrepancies")
    metrics = rows(ROOT / "results/fall_metrics.csv")
    cells = {(r["family"], r["seed"], r["run"], r["method"].split(":")[1], r["threshold"], r["mol_type"]) for r in metrics}
    require(len(metrics) == 840 and cells == {k + (t, m) for k in keys for t in ("100", "1000")
                                              for m in ("plasmid", "chrom")}, "Fall metric inventory")
    grouped = defaultdict(list)
    def group(r):
        return tuple(r[k] for k in ("family", "threshold", "mol_type", "method"))
    for r in metrics:
        grouped[group(r)].append(r)
    summary = rows(ROOT / "results/fall_summary.csv")
    require(len(summary) == len(grouped) and {group(r) for r in summary} == set(grouped), "Fall groups")
    for r in summary:
        members = grouped[group(r)]
        require(len(members) == int(r["n"]), str(r))
        for metric in ("f1", "precision", "recall", "accuracy", "auc"):
            values = [float(x[metric]) for x in members]
            for suffix, operation in (("mean", mean), ("sd", stdev), ("min", min), ("max", max)):
                require(isclose(float(r[metric + "_" + suffix]), operation(values), abs_tol=1e-12), str(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--august-root", type=Path)
    validate(parser.parse_args().august_root)
    print("Validated 120 custom metric rows, 210 Fall fits and 840 Fall metric rows.")
