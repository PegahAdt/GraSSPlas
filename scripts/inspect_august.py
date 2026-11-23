"""Inspect historical graph loading and check the order used for positional labels."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/august-2024"))
from toolsDetection import get_yinitial, process_files


def inspect(manifest, saved):
    edges, features, labels, _, gold = process_files(str(manifest), 100)
    if not features.contig.is_unique or not features.contig.equals(labels.contig):
        raise ValueError("Feature and label IDs differ or their row orders disagree")
    if saved:
        for table, name in [(features, "features"), (labels, "labels"), (gold, "ground_truth")]:
            historical = pd.read_csv(saved / (name + ".csv"))
            pd.testing.assert_frame_equal(table.reset_index(drop=True), historical,
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
    pairs = set(edges)
    return dict(contigs=len(features), directed_entries=len(edges),
                self_loops=sum(u == v for u, v in pairs),
                reciprocal_nonself_entries=sum(u != v and (v, u) in pairs for u, v in pairs),
                masked_nodes=int(get_yinitial(labels)[1].sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--saved-run", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.manifest, args.saved_run), indent=2))
