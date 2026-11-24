"""Validate, summarize, or score predictions for the three experiments."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import groupby, zip_longest
from math import isclose
from pathlib import Path
import sys
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
BASELINES = ("plasforest_pls", "plasgraph2_pls", "platon_pls", "rfplasmid_pls", "mlplasmid_pls")
GRASSPLAS = "GraSSPlas"


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


def recount(directory, predictions_path=None):
    counts = Counter()
    features = rows(directory / "features.csv")
    gold = rows(directory / "ground_truth.csv")
    predictions = rows(predictions_path or directory / "y_final.csv")
    for feature, label, pred in zip_longest(features, gold, predictions):
        require(None not in (feature, label, pred), "Unequal table lengths")
        require(feature["contig"] == label["contig"], "Contig order differs")
        predicted = {name: float(feature[name]) > 0.5 for name in BASELINES}
        predicted[GRASSPLAS] = int(pred["y_final"])
        require(predicted[GRASSPLAS] in (0, 1), "Nonbinary prediction")
        intersection, union = all(predicted.values()), any(predicted.values())
        predicted.update(intersection=intersection, union=union)
        actual = int(label["Plasmid"])
        require(actual in (0, 1), "Nonbinary gold")
        for cutoff in (0, 100, 1000):
            if float(feature["length"]) >= cutoff:
                for name, call in predicted.items():
                    counts[cutoff, name, ("TN", "FP", "FN", "TP")[2 * actual + int(call)]] += 1
    return counts


def experiment_rows(name):
    return rows(EXPERIMENTS / name / "results.csv")


def validate(grassplas_exports=None, grassplas_evaluation=None):
    grassplas = experiment_rows("grassplas")
    expected = {(str(i), str(c), m) for i in range(1, 6) for c in (0, 100, 1000)
                for m in (*BASELINES, GRASSPLAS, "intersection", "union")}
    require(len(grassplas) == len(expected) and
            {(r["run"], r["threshold"], r["tool"]) for r in grassplas} == expected,
            "Incomplete GraSSPlas metrics")
    observed = {}
    if grassplas_evaluation is not None:
        predictions = EXPERIMENTS / "grassplas/results"
        observed = {str(i): recount(grassplas_evaluation, predictions / f"run_{i}.csv")
                    for i in range(1, 6)}
    elif grassplas_exports is not None:
        observed = {str(i): recount(grassplas_exports / f"results_{i}") for i in range(1, 6)}
    for r in grassplas:
        tp, fp, fn, tn = (int(r[k]) for k in ("TP", "FP", "FN", "TN"))
        require(tp + fp + fn + tn == (4440 if r["threshold"] == "1000" else 12176), str(r))
        for key, value in zip(("precision", "recall", "f1"), ratios(tp, fp, fn)):
            require(isclose(float(r[key]), value, abs_tol=1e-12), str(r))
        if observed:
            for key in ("TP", "FP", "FN", "TN"):
                require(observed[r["run"]][int(r["threshold"]), r["tool"], key] == int(r[key]), str(r))

    weak = experiment_rows("weak-supervision")
    diagnostic = experiment_rows("robustness")
    with (EXPERIMENTS / "weak-supervision/conditions.json").open() as stream:
        conditions = json.load(stream)
    with (EXPERIMENTS / "robustness/protocols.json").open() as stream:
        protocols = json.load(stream)
    repeated = [r for r in weak if r["protocol"] == "repeated_fits"]
    fits = repeated + diagnostic
    fields = ("protocol", "condition", "nominal_seed", "repeat", "minimum_bp", "molecule")
    expected = {(protocol, condition, str(seed), str(repeat), length, molecule)
                for protocol, spec in protocols.items()
                for condition in conditions
                for seed in spec["nominal_seeds"]
                for repeat in spec["repeat_ids"]
                for length in ("100", "1000") for molecule in ("plasmid", "chrom")}
    actual = {tuple(r[k] for k in fields) for r in fits}
    require(len(repeated) == 600 and len(diagnostic) == 240 and
            len(fits) == len(expected) and actual == expected, "Incomplete plASgraph2 fits")
    recorded = defaultdict(set)
    for r in fits:
        key = tuple(r[k] for k in fields[:4])
        recorded[key].add(r["configured_seed"])
    require(all(len(seeds) == 1 for seeds in recorded.values()), "Inconsistent fit seed")
    require(sum(seed not in seeds for (_, _, seed, _), seeds in recorded.items()) == 33,
            "Unexpected count of seed discrepancies")
    require(Counter(r["protocol"] for r in weak) ==
            {"repeated_fits": 600, "single_fit_comparison": 36, "external_baselines": 10},
            "Incomplete weak-supervision comparisons")
    for r in weak + diagnostic:
        real, predicted, n = (float(r[k]) for k in ("real", "predicted", "contigs"))
        tp = round(real * float(r["recall"]))
        fp, fn = predicted - tp, real - tp
        require(0 <= tp <= real and 0 <= fp <= n - real, str(r))
        precision = tp / predicted if predicted else 1
        recall = tp / real if real else 1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        for key, value in (("precision", precision), ("recall", recall), ("f1", f1)):
            require(isclose(float(r[key]), value, abs_tol=1e-12), str(r))
        if r["accuracy"]:
            require(isclose(float(r["accuracy"]), (n - fp - fn) / n, abs_tol=1e-12), str(r))


def f1_groups(records, fields):
    groups = defaultdict(list)
    for r in records:
        if r["molecule"] == "plasmid" and r["labels"] == "gold":
            groups[tuple(r[k] for k in fields)].append(float(r["f1"]))
    return groups


def summarize(experiment, stream, seed=None):
    weak = [r for r in experiment_rows("weak-supervision") if r["protocol"] == "repeated_fits"]
    if experiment == "robustness":
        # Match nominal and configured seed for the reference to both intervention sets.
        records = weak + experiment_rows("robustness")
        if seed is None:
            records = [r for r in records if r["nominal_seed"] == r["configured_seed"] == "123"]
    else:
        records = weak
    fields = ("protocol", "condition", "minimum_bp")
    if seed:
        fields += (seed,)
    groups = f1_groups(records, fields)
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow([*fields, "fits", "f1_mean", "f1_sample_sd"])
    for key, values in sorted(groups.items()):
        writer.writerow([*key, len(values), mean(values), stdev(values) if len(values) > 1 else ""])


def roc_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    rank = 1
    positive_ranks = 0.0
    for _, tied in groupby(sorted(zip(scores, labels)), key=lambda pair: pair[0]):
        group = list(tied)
        positive_ranks += (rank + (len(group) - 1) / 2) * sum(y for _, y in group)
        rank += len(group)
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def score_plasgraph(gold_rows, predictions):
    def keys(row):
        return row["sample"], row["contig"]

    scores = {keys(r): r for r in predictions}
    require(len(scores) == len(predictions), "Duplicate prediction contig IDs")
    for length in (100, 1000):
        cohort = [r for r in gold_rows if r["label"] != "unlabeled" and float(r["length"]) >= length]
        require(cohort and len({keys(r) for r in cohort}) == len(cohort), "Invalid gold contig IDs")
        for molecule, label, column in (("plasmid", "plasmid", "plasmid_score"),
                                        ("chrom", "chromosome", "chrom_score")):
            y = [int(r["label"] in (label, "ambiguous")) for r in cohort]
            p = [float(scores[keys(r)][column]) for r in cohort]
            require(all(0 <= value <= 1 for value in p), "Scores must be probabilities")
            calls = [round(value) for value in p]
            tp = sum(a * b for a, b in zip(y, calls))
            precision = tp / sum(calls) if sum(calls) else 1
            recall = tp / sum(y) if sum(y) else 1
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            yield dict(minimum_bp=length, molecule=molecule, contigs=len(y), real=sum(y),
                       predicted=sum(calls), expected=sum(p), auc=roc_auc(y, p), precision=precision,
                       recall=recall, f1=f1, accuracy=sum(a == b for a, b in zip(y, calls)) / len(y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    check = actions.add_parser("validate")
    inputs = check.add_mutually_exclusive_group()
    inputs.add_argument("--grassplas-exports", type=Path,
                        help="Recount the original complete exports under results_1 through results_5")
    inputs.add_argument("--grassplas-evaluation", type=Path,
                        help="Recount the included predictions using score and reference tables")
    # Keep the former option names for existing command lines.
    inputs.add_argument("--graphsage-exports", dest="grassplas_exports", type=Path,
                        help=argparse.SUPPRESS)
    inputs.add_argument("--graphsage-evaluation", dest="grassplas_evaluation", type=Path,
                        help=argparse.SUPPRESS)
    summary = actions.add_parser("summary")
    summary.add_argument("experiment", choices=["weak-supervision", "robustness"])
    summary.add_argument("--by-seed", choices=["nominal_seed", "configured_seed"])
    score = actions.add_parser("score-plasgraph")
    score.add_argument("gold", type=Path)
    score.add_argument("predictions", type=Path)
    args = parser.parse_args()
    if args.action == "validate":
        validate(args.grassplas_exports, args.grassplas_evaluation)
        print("Validated 120 GraSSPlas experiment rows, 840 plASgraph2 fit rows and 46 comparison rows.")
        if args.grassplas_exports or args.grassplas_evaluation:
            print("Recomputed confusion counts for all 120 GraSSPlas experiment rows from prediction inputs.")
    elif args.action == "summary":
        summarize(args.experiment, sys.stdout, args.by_seed)
    else:
        results = list(score_plasgraph(rows(args.gold), rows(args.predictions)))
        writer = csv.DictWriter(sys.stdout, list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
