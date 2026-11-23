# A. Semi-supervised plasmid classification using GraphSAGE

## Research question

How well does the custom GraphSAGE pipeline classify plasmid contigs compared with the five classifiers that provide its input scores, when all are evaluated on the same assemblies and reference labels?

## Plain-English explanation

Only some contigs have training labels. The model learns from provisional labels on those contigs, while assembly links let it combine information from neighboring contigs. For example, a contig without a training label can still pass its five classifier pairs to a linked contig through the graph layer. It contributes no direct error term. The model learns through classification targets assigned by biological rules. It uses no separate self-supervised task such as predicting hidden input features. After training, the model makes a plasmid-or-chromosome prediction for every contig, regardless of whether it had a training label.

## Model and implementation

One PyTorch Geometric `SAGEConv` combines a contig's own features with the mean of its incoming neighbors' features. The resulting 16 values per contig are normalized, passed through ReLU, which sets negative values to zero, and mapped to two class scores. Links are used in the direction stored by the loader. [PyG GraphSAGE layer](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.SAGEConv.html) · [Message direction](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.MessagePassing.html)

| Step | Code |
| --- | --- |
| Load the graph and scores | `process_files()` calls `parse_gfa_file()` and `modify_contig_names()`. The driver uses `replace_contig_with_index()` to turn links into node indices. |
| Select training targets | `get_yinitial()` uses `compute_train_mask()` to select contigs with exactly one flag. The plasmid flag supplies the binary target for each selected contig. |
| Learn from the graph | `GCN.forward()` applies the layer and head to all nodes. In `embLearn()`, only the selected predictions and targets enter `CrossEntropyLoss`. |
| Classify | `embLearn()` runs the model over all nodes in evaluation mode and picks the higher-scoring class. The driver saves the predictions in `y_final.csv`. |
| Evaluate | `read_ground_truth()` and `create_ground_truth_indicator()` load the reference separately. The driver filters the scored contigs by length and calls `compute_and_save_metrics()`. |

The [driver](src/mainPlasDetection.py) wires these steps together. The [model and data helpers](src/toolsDetection.py) implement them. `--isSemiSupervised` and `--p` are printed, but do not change training or label generation. `--noGNN 1` still runs one graph-training epoch. It drops the graph embeddings from the optional random forest input. With the default forest bypass, predictions still come from the graph model.

## Dataset and training labels

The [ordered manifest](samples.csv) covers 52 PLASEVAL *E. coli* assemblies and 12,176 contigs. Short contigs were contracted during graph preparation, connecting neighboring ends before removing a short node. The model loader keeps contigs of at least 100 bp and prefixes IDs with the manifest row number. Assemblies remain disconnected from each other. Each node has ten input scores, two each from PlasForest, plASgraph2, Platon, RFPlasmid and mlPlasmids. Length is used for filtering and evaluation. Read depth and GC content are loaded but excluded from model features. [PLASEVAL](https://doi.org/10.1186/s12859-024-05941-0)

The experiment consumes precomputed `_060` pseudolabel tables. The public generator sets a plasmid flag for matches in the marker mapping files. It sets a chromosome flag when a contig meets a minimum length and its read depth is below the assembly mean plus twice the standard deviation. The actual loss uses **250 plasmid-only and 843 chromosome-only contigs**. The full graph includes **11,075 contigs with neither flag and eight with both flags**. The model's training targets come from these flags. The evaluation label is positive if the contig is in the separate reference plasmid list and negative otherwise. [Public label generator](https://github.com/PegahAdt/GraSSPlas/blob/0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6/code/generate_labels.py)

## Experimental design

The assembled graph, score inputs, target flags, model settings and reference labels remain fixed across the five reported invocations. Each invocation initializes ten models and runs 2,001 epochs per fit with Adam at learning rate 0.001. It saves the last fit's predictions. The reported calls come from the GraphSAGE head. The optional random forest was bypassed.

Each baseline calls plasmid for scores above 0.5. The six methods are scored against the same reference on the same contigs at each cutoff. A 1 kb cutoff selects a subset of already predicted contigs. The 0 bp and 100 bp CSV results are identical because the input graph already uses a 100 bp floor. Union and intersection results in the CSV combine all six methods.

## Results

GraphSAGE ranges show the minimum and maximum of each metric across the five saved predictions. The minima and maxima for different metrics need not come from the same fit. Baselines give the same results across invocations.

| Classifier | Precision ≥100 bp | Recall ≥100 bp | F1 ≥100 bp | Precision ≥1 kb | Recall ≥1 kb | F1 ≥1 kb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE | 0.2040–0.3180 | 0.8067–0.9397 | 0.3353–0.4640 | 0.3999–0.4493 | 0.9247–0.9525 | 0.5624–0.6068 |
| PlasForest | 0.4801 | 0.8001 | 0.6001 | 0.6984 | 0.7316 | 0.7146 |
| plASgraph2 | 0.3830 | 0.5597 | 0.4548 | 0.4941 | 0.4812 | 0.4876 |
| Platon | 0.9705 | 0.2554 | 0.4043 | 0.9705 | 0.7005 | 0.8137 |
| RFPlasmid | 0.8241 | 0.3300 | 0.4712 | 0.8357 | 0.7741 | 0.8037 |
| mlPlasmids | 0.7883 | 0.2243 | 0.3493 | 0.7883 | 0.6154 | 0.6912 |

At 1 kb, the best-F1 GraphSAGE fit detected 571 reference plasmid contigs, missed 40 and made 700 false-positive calls. It had 3,129 true negatives across the 4,440-contig subset. Platon detected 428 and made 13 false-positive calls on that same subset. [Per-run results](results.csv)

## Conclusions

GraphSAGE found more reference plasmid contigs than Platon at 1 kb, while its extra false-positive calls drove down precision and F1. PlasForest had higher F1 than every saved GraphSAGE fit at 100 bp. These comparisons measure the whole GraphSAGE pipeline relative to individual classifiers. They do not isolate what the graph adds, because a matched model trained and scored without links is unavailable.

## How to run or verify

Install the [GraphSAGE environment](../../README.md#installation-and-reproducibility). Run from the repository root.

```sh
git fetch origin
python experiments/graphsage/prepare_inputs.py work/graphsage-inputs
python experiments/graphsage/inspect_inputs.py work/graphsage-inputs/manifest.csv
bash experiments/graphsage/run.sh work/graphsage-inputs/manifest.csv work/graphsage/results_1
```

[`prepare_inputs.py`](prepare_inputs.py) builds a local manifest from pinned public graph, feature and pseudolabel files on `pseudolabels` and reference files on `data`. [`inspect_inputs.py`](inspect_inputs.py) checks the feature and label ID order and reports the graph and training-set sizes. For a local copy of the data, pass `--data-root /path/to/data` to the preparer. [`run.sh`](run.sh) launches the original driver. The launcher writes `features.csv`, `labels.csv`, `ground_truth.csv`, `y_final.csv` and three threshold reports. Use a separate output directory for each invocation.

```sh
python scripts/results.py validate
# With access to the five saved exports under results_1 through results_5
python scripts/results.py validate --graphsage-exports /path/to/graphsage-exports
```

The first command checks the included table. The second recomputes its GraphSAGE confusion counts from the supplied exports. The training command above runs the full configured fit sequence. The synthetic smoke test in the root README is the lightweight execution check.

## Limitations

The original head feeds probabilities computed by softmax to `CrossEntropyLoss`, which expects raw class scores. The published metrics describe the objective as implemented. Input rows are aligned by position, so run the preflight to detect mismatched contig order. The loader takes the serialized direction of each accepted link, ignores GFA orientation and overlap fields, and skips links whose endpoint records have yet to appear. The evaluated graph has 25,347 directed entries, including 125 self-loops and no explicit reciprocal non-self pairs.

Training and evaluation share the same graphs and include the selected training contigs in the final evaluation. These scores do not estimate accuracy on new isolates. The input tables can be loaded again, while the original tool database versions and all label-generation commands remain incomplete. Per-fit seeds and trained GraphSAGE weights are unavailable. Only the last prediction from each ten-fit invocation is available.
