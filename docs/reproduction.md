# Reproduction guide

## August environment and smoke test

On Fir, the synthetic CPU smoke test and real-input preflight ran in the recovered Python 3.8.10 environment with PyTorch 1.13.1, PyG 2.5.3, NumPy 1.21.0, pandas 1.2.4 and scikit-learn 0.24.2. For a compatible Linux x86-64 setup, install a platform-matched PyTorch 1.13.1 build and the versioned dependencies in a Python 3.8 virtual environment. The original Fir environment used Compute Canada wheels. The public-wheel installation below is an environment recipe. The verified run used the Fir environment. [Official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/) · [PyG 2.5.3 installation](https://pytorch-geometric.readthedocs.io/en/2.5.3/install/installation.html)

```sh
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==1.13.1+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-august.txt
python tests/smoke_august.py
```

The smoke test runs a single optimization step and checks exports on a four-node synthetic graph. It uses the historical driver's `--noGNN 1` shortcut to limit training to one epoch.

## Prepare and inspect the historical inputs

Fetch `data` and `pseudolabels` in a clone of the repository. The preparer reads the pinned blobs and verifies each against the reconstructed Fir manifest before writing a new input directory. With an authorized local copy, add `--data-root /path/to/data` to verify and use existing files in place.

```sh
git fetch origin data pseudolabels
python scripts/prepare_inputs.py work/august
python scripts/inspect_august.py work/august/manifest.csv
```

With access to saved outputs, `--saved-run /path/to/results_1` compares loaded feature, label and gold tables against the historical exports. Preparation does not require accelerator libraries. Inspection imports the historical PyG helper.

The exact historical Slurm launcher is preserved for provenance. The portable wrapper uses the active Python environment and caller-supplied paths. Each invocation follows the historical defaults of ten 2,001-epoch fits. The driver saves the last prediction vector and its metrics. This invocation was left unrun on the full data.

```sh
bash scripts/run_august.sh work/august/manifest.csv work/new-fit
python scripts/validate_results.py --august-root /path/to/saved-august-runs
```

The validator expects `results_1` through `results_5` under the supplied absolute directory. The saved predictions have no row IDs, so their ordering comes from the historical driver. The validator checks feature and gold IDs before pairing them with that vector.

## Optional local preprocessing

An authorized lab checkout can supply the original graph contraction script, its batch launcher and the Pfam/MOB merge code. The recovery command copies only files listed by checksum and writes to an ignored local directory. The filter wrapper runs the original algorithm for one sample. The independent combine routine expresses the documented inner join and rejects duplicate IDs.

```sh
python scripts/recover_local.py /path/to/PLASMIDS local-recovery
python scripts/preprocess.py filter local-recovery/DATA/Filter_GFA/filter_GFA.py input.gfa work/filtered.gfa --minimum 100
python scripts/preprocess.py combine pfam.tsv mob.tsv work/combined.tsv
```

The public filtered graphs and score tables already provide the verified model inputs. The preprocessing wrappers do not reconstruct the original five tools' databases or rebuild their score tables.

## Fall experiments

The shared adaptation used a separate Python environment. Its requirements list TensorFlow 2.8.0, Spektral 1.1.0, protobuf 3.19.4, PyYAML 6.0, NetworkX 2.8.3, NumPy 1.22.2, pandas 1.4.1, scikit-learn 1.0.2 and SciPy 1.8.0, plus Biopython and Matplotlib. The public plASgraph2 package does not establish a compatible checkout for the external feature and pseudolabel flags. Obtain an authorized copy of the lab adaptation, its cohort and the condition-specific CSVs. Review the config and output paths before any execution.

```sh
bash scripts/run_fall.sh /path/to/authorized/plasgraph2-working /path/to/inputs mob2 experiments/fall/example-seed123.yaml work/mob2-replay
```

The wrapper follows the documented training, classification and pooled gold evaluation interfaces. Exact historical replay requires a verified source revision and per-run input provenance. The Fall wrapper received shell syntax checks. No Fall training or end-to-end run was performed. The published result tables can be validated independently of TensorFlow.
