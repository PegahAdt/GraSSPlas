"""Run one condition with an authorized plASgraph2 adaptation and prepared tables."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

EXPERIMENT = Path(__file__).resolve().parent


def run(args, prefix):
    source = args.adaptation / "src"
    # The adapter concatenates this prefix with the relative GFA path.
    data_prefix = os.path.join((args.adaptation / "data").resolve(), "")
    cohort = args.adaptation / "sets/plaseval-nolabels.csv"
    features = args.inputs / (prefix + "-extfeatures.csv")
    targets = args.inputs / (prefix + "-pseudolabels.csv")
    gold = args.inputs / "mob-ground-truth.csv"
    train, classify, evaluate = (source / name for name in
                                 ("plASgraph2_train.py", "plASgraph2_classify.py", "eval.py"))
    for file in (cohort, features, targets, gold, args.config, train, classify, evaluate):
        if not file.is_file():
            raise FileNotFoundError(file)
    model = args.output / "model"
    predictions = args.output / "predictions.csv"
    args.output.mkdir(parents=True)
    with (args.output / "train.log").open("w") as log:
        subprocess.run([sys.executable, str(train), "-f", str(features), "-G", str(targets),
                        str(args.config), str(cohort), data_prefix, str(model),
                        "-l", str(args.output / "loss")], stdout=log, check=True)
    subprocess.run([sys.executable, str(classify), "set", "-f", str(features), str(cohort),
                    data_prefix, str(model), str(predictions)], check=True)
    for length in (100, 1000):
        with (args.output / f"stats_{length}.csv").open("w") as stream:
            subprocess.run([sys.executable, str(evaluate), "-m", str(length), "-o", "-n",
                            args.condition, str(gold), str(predictions)], stdout=stream, check=True)


if __name__ == "__main__":
    with (EXPERIMENT / "conditions.json").open() as stream:
        conditions = json.load(stream)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adaptation", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("condition", choices=conditions)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "model.yaml")
    args = parser.parse_args()
    run(args, conditions[args.condition]["input_prefix"])
