#!/usr/bin/env bash
# Portable wrapper around the lab implementation. Run under its separate environment.
set -eu
if [ "$#" -ne 5 ]; then
    echo "Usage: bash scripts/run_fall.sh PLASGRAPH2_ROOT EXP_INPUT_DIR CONDITION CONFIG NEW_OUTPUT" >&2
    exit 2
fi
repo=$(realpath "$1")
inputs=$(realpath "$2")
condition=$3
config=$(realpath "$4")
case "$condition" in mob|mob2|pfam|dropout|random|true) ;; *) exit 2 ;; esac
out=$5
mkdir -p "$(dirname "$out")"
mkdir "$out"
setfile="$repo/sets/plaseval-nolabels.csv"
features="$inputs/$condition-extfeatures.csv"
python3 "$repo/src/plASgraph2_train.py" -f "$features" -G "$inputs/$condition-pseudolabels.csv" "$config" "$setfile" "$repo/data/" "$out/model" -l "$out/loss" > "$out/train.log"
python3 "$repo/src/plASgraph2_classify.py" set -f "$features" "$setfile" "$repo/data/" "$out/model" "$out/predictions.csv"
for threshold in 100 1000; do
    python3 "$repo/src/eval.py" -m "$threshold" -o -n "$condition" "$inputs/mob-ground-truth.csv" "$out/predictions.csv" > "$out/stats_$threshold.csv"
done
