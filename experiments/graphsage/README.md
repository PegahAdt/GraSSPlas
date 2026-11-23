# GraphSAGE plasmid classification

Can assembly-graph context improve plasmid classification over the tools that supply the node features? This experiment trains a custom PyTorch Geometric model on a subset of pseudolabeled contigs and evaluates every contig in the same assembly collection.

## Data and model

The [ordered sample manifest](samples.csv) describes 52 PLASEVAL *E. coli* assemblies. Its four inputs per sample are a filtered GFA, a feature TSV, a pseudolabel TSV and the binary plasmid gold list. The input preparer reads the filtered graphs, features and `_060` labels from the public `pseudolabels` ref and the exact gold files from `data`. The pinned commits in the preparer are the source of the input versions. [PLASEVAL](https://doi.org/10.1186/s12859-024-05941-0) · [Public plasmid inputs and pseudolabel implementation](https://github.com/PegahAdt/GraSSPlas/tree/0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6) · [Binary gold inputs](https://github.com/PegahAdt/GraSSPlas/tree/eb264d9a45ac278fae37a678e4440cf23145801a/Data/Ecoli_GroundTruth)

The input graphs were built by contracting nodes shorter than 100 bp in an undirected GFA representation. The contraction connected pairs of neighboring extremities and wrote replacement links with `0M` overlap. The model loader prefixes contig IDs with the sample's one-based manifest row number and concatenates the assemblies as a disjoint graph. Node features are the chromosome and plasmid scores from each of PlasForest, plASgraph2, Platon, RFPlasmid and mlplasmids. Length, read depth and GC content are retained in input tables, and length is used for evaluation. The model uses only the ten classifier scores.

One `SAGEConv` maps the scores to 16 hidden channels. Batch normalization and ReLU precede a linear two-class head. Of the 12,176 contigs, 1,093 have exactly one plasmid or chromosome pseudolabel and contribute to the loss. Model defaults in the [driver](src/mainPlasDetection.py) and [model helper](src/toolsDetection.py) are Adam at 0.001, 2,001 epochs and ten fits per invocation. The five reported invocations use the GraphSAGE head. An optional random forest uses the masked labels to fit 100 trees over raw scores and embeddings.

## Evaluation and results

Baseline scores above 0.5 are called plasmid. Gold is binary membership in the reference plasmid list. Each precision, recall and F1 is pooled over contigs at or above the chosen length. The 1 kb cutoff filters the scored predictions. The 0 bp and 100 bp rows have the same input denominator because the graph was already filtered at 100 bp. Union and intersection in the CSV include GraphSAGE and all five baseline calls.

GraphSAGE ranges are the minimum and maximum of each metric across the five saved predictions. Baseline rows are constant across those runs.

| Classifier | Precision ≥100 bp | Recall ≥100 bp | F1 ≥100 bp | Precision ≥1 kb | Recall ≥1 kb | F1 ≥1 kb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE | 0.2040–0.3180 | 0.8067–0.9397 | 0.3353–0.4640 | 0.3999–0.4493 | 0.9247–0.9525 | 0.5624–0.6068 |
| PlasForest | 0.4801 | 0.8001 | 0.6001 | 0.6984 | 0.7316 | 0.7146 |
| plASgraph2 | 0.3830 | 0.5597 | 0.4548 | 0.4941 | 0.4812 | 0.4876 |
| Platon | 0.9705 | 0.2554 | 0.4043 | 0.9705 | 0.7005 | 0.8137 |
| RFPlasmid | 0.8241 | 0.3300 | 0.4712 | 0.8357 | 0.7741 | 0.8037 |
| mlplasmids | 0.7883 | 0.2243 | 0.3493 | 0.7883 | 0.6154 | 0.6912 |

At 1 kb the best-F1 GraphSAGE fit has 571 true positives, 700 false positives, 40 false negatives and 3,129 true negatives across 4,440 contigs. Platon has 428 true positives and 13 false positives on the same subset. GraphSAGE trades substantially lower precision for higher recall in that comparison. At 100 bp, GraphSAGE's best-F1 fit makes 3,084 false-positive calls compared with 1,452 for PlasForest. [Per-run metrics and confusion matrices](results.csv) · [F1 comparison](comparison.png)

## Reproduction

Set up the GraphSAGE environment from the [root installation guide](../../README.md#installation-and-reproducibility). Run from the repository root.

```sh
git fetch origin data pseudolabels
python experiments/graphsage/prepare_inputs.py work/graphsage-inputs
python experiments/graphsage/inspect_inputs.py work/graphsage-inputs/manifest.csv
bash experiments/graphsage/run.sh work/graphsage-inputs/manifest.csv work/graphsage/results_1
```

The preparer extracts the pinned graph, feature, pseudolabel and gold tables into a new directory. For a compatible local dataset tree, pass `--data-root /path/to/data`. The preflight checks contig identity and row order before training. The expected cohort has 12,176 contigs and 25,347 directed edge entries.

The launcher writes `features.csv`, `labels.csv`, `ground_truth.csv`, `y_final.csv` and three `metrics_threshold_*.txt` files. Use a separate output directory for each invocation. To validate all reported confusion counts with access to the five original `results_1` through `results_5` exports, run

```sh
python scripts/results.py validate --graphsage-exports /path/to/graphsage-exports
```

### Reproducibility limits

The original head applies softmax before `CrossEntropyLoss`, which expects logits. The canonical source preserves that loss computation. The loader uses positional feature and pseudolabel alignment. Reordering an input table can silently change training labels, so run the preflight when supplying inputs. The GFA loader discards overlap and orientation flags, retains serialized directions and accepts only links whose segments have already appeared. The experimental graph has 125 self-loops and no explicit reverse pairs for non-self links.

The full graph is available during training and evaluation, which makes this a transductive test. The published input scores and labels can be loaded again, but the exact run seeds, tool database versions and trained weights are unavailable. Each ten-fit invocation saved only its last prediction. A synthetic one-epoch run verifies the executable path, while full historical retraining and regeneration of the classifier score tables remain unverified.
