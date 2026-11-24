"""Exercise the GraSSPlas driver's graph, loss and export paths with synthetic inputs on CPU."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    (d / "graph.gfa").write_text("".join(f"S\t{i}\t" + "A"*1000 + "\n" for i in range(4))
                                 + "L\t0\t+\t1\t-\t0M\nL\t2\t+\t3\t+\t0M\n")
    scores = [k + "_" + c for k in ("plasforest", "plasgraph2", "platon", "rfplasmid", "mlplasmid")
              for c in ("pls", "chr")]
    with (d / "features.tsv").open("w") as stream:
        w = csv.writer(stream, delimiter="\t")
        w.writerow(["contig", *scores, "length", "read_depth", "gc_content"])
        for i in range(4):
            w.writerow([i, *([i % 2, 1-i % 2]*5), 1000, 1, 0.5])
    (d / "labels.tsv").write_text("contig\tplasmid\tchromosome\n0\t0\t1\n1\t1\t0\n2\t0\t0\n3\t1\t1\n")
    (d / "gold.tsv").write_text("contig\n1\n3\n")
    manifest = d / "manifest.csv"
    with manifest.open("w") as stream:
        w = csv.writer(stream)
        w.writerow(["sample", "gfa_path", "features_path", "labels_path", "ground_truth"])
        w.writerow(["toy", *(d / x for x in ("graph.gfa", "features.tsv", "labels.tsv", "gold.tsv"))])
    output = d / "output"
    subprocess.run(["bash", str(ROOT / "experiments/grassplas/run.sh"), str(manifest), str(output),
                    "--N_iter", "1", "--noGNN", "1"],
                   env=dict(os.environ, PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                   check=True, cwd=d)
    with (output / "y_final.csv").open() as f:
        preds = list(csv.DictReader(f))
    assert len(preds) == 4 and all(r["y_final"] in {"0", "1"} for r in preds)
    assert all((output / f"metrics_threshold_{t}.txt").is_file() for t in (0,100,1000))
    print("Synthetic GraSSPlas driver passed.")
