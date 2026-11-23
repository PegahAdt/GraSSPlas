#!/usr/bin/env bash
set -eu
if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/run_august.sh MANIFEST NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/.." && pwd)
manifest=$(realpath "$1")
mkdir -p "$(dirname "$2")"
mkdir "$2"
exec "${PYTHON:-python3}" "$root/src/august-2024/mainPlasDetection.py" --name "$manifest" --output_dir "$2"
