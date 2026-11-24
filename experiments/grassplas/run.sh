#!/usr/bin/env bash
set -eu
if [ "$#" -lt 2 ]; then
    echo "Usage: bash experiments/grassplas/run.sh MANIFEST NEW_OUTPUT_DIRECTORY [MODEL_FLAGS...]" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")" && pwd)
manifest=$(realpath "$1")
output=$2
shift 2
mkdir -p "$(dirname "$output")"
mkdir "$output"
exec "${PYTHON:-python3}" "$root/src/mainPlasDetection.py" "$@" --name "$manifest" --output_dir "$output"
