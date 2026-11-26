# GraSSPlas — Graph-Based Semi-Supervised Learning for Plasmid Classification

## Project overview

**GraSSPlas** is a graph-based, semi-supervised learning model for identifying plasmid-derived DNA fragments in bacterial genome assemblies. It combines scores from five existing plasmid classifiers with connections between fragments. Its **GraphSAGE** architecture learns from automatically generated labels for a subset of those fragments.

This repository contains the GraSSPlas implementation, a prepared-input workflow, and evaluation on **52 *E. coli* assemblies**. Two related experiments use an adapted **plASgraph2** model to investigate how training labels and input features affect classification and why results vary across repeated training runs.

## Implementation

| Component | Source code | Purpose |
| --- | --- | --- |
| Model, graph loading, and training functions | [toolsDetection.py](experiments/grassplas/src/toolsDetection.py) | Defines GraSSPlas, loads assembly graphs and classifier scores, selects training targets, and trains the model. |
| Training and evaluation workflow | [mainPlasDetection.py](experiments/grassplas/src/mainPlasDetection.py) | Assembles model inputs, runs training, exports predictions, and evaluates GraSSPlas alongside existing classifiers. |
| Prepared-data setup | [prepare_inputs.py](experiments/grassplas/prepare_inputs.py) | Stages prepared graphs, classifier scores, training labels, and reference labels from pinned public revisions, or from a local copy. |
| Results verification | [results.py](scripts/results.py) | Checks GraSSPlas metrics against saved predictions, scores supplied plASgraph2 predictions, and summarizes repeated fits. |

The GraSSPlas workflow builds on [GraSSRep](https://github.com/aliaaz99/GraSSRep) and uses PyTorch Geometric's [GraphSAGE layer](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.SAGEConv.html). The [GraSSPlas experiment guide](experiments/grassplas/README.md) identifies the code paths, required inputs, and reproduction commands.

## How GraSSPlas works

**Plasmids** are DNA molecules separate from bacterial chromosomes. Some carry antibiotic-resistance genes, making plasmid identification useful for studying how resistance spreads. Genome assembly produces DNA fragments called **contigs**. The task is to determine whether each contig comes from a plasmid or a chromosome.

GraSSPlas approaches this problem in four stages:

1. **Prepare connected DNA fragments.** Each contig is represented as a node in an assembly graph, with edges indicating possible connections between fragments.
2. **Combine existing predictions.** Five classifiers—PlasForest, Platon, RFPlasmid, plASgraph2, and mlPlasmids—provide scores used as input features. Graph connections offer additional context when individual scores are ambiguous.
3. **Learn from provisional labels.** Automatically generated **pseudolabels** identify a subset of contigs as likely plasmid or chromosome. GraSSPlas uses GraphSAGE to incorporate information from neighboring contigs. All contigs remain in the graph, but only the **1,093 selected training contigs** directly contribute to the classification loss in the recovered **12,176-contig** experiment.
4. **Evaluate predictions.** GraSSPlas predicts a class for each contig. Its predictions and those of the five classifiers are compared against reference labels at minimum contig lengths of **100 bp** and **1 kb**. **Precision** measures how many predicted plasmid contigs are correct; **recall** measures how many actual plasmid contigs are identified; **F1** balances the two.

The [GraSSPlas experiment README](experiments/grassplas/README.md) describes the pseudolabel rules and experimental setup in greater detail.

## Experiments and results

The three experiments address related questions, but they do **not** all train the same model. Experiment A evaluates the proposed GraSSPlas model. Experiments B and C use an **adapted plASgraph2 implementation** on a different set of **51 isolates** and different evaluation labels. Their F1 scores should not be used as a direct ranking of the two architectures.

| Experiment | Research question | Main result and conclusion |
| --- | --- | --- |
| **[A. GraSSPlas classification](experiments/grassplas/README.md)** | Can semi-supervised graph learning combine classifier scores and assembly links to identify plasmid contigs? | At **1 kb**, the best GraSSPlas F1 was **0.6068**, versus **0.8137** for Platon. GraSSPlas found more of the actual plasmid contigs (higher recall), but additional false positives reduced its F1. |
| **[B. Pseudolabel and feature strategies](experiments/weak-supervision/README.md)** | How do training-label choices and input features affect classification? | **MOB2** achieved the highest mean F1 among the automatic-label strategies: **0.8597 ± 0.0146** at 1 kb across **25 fits**. The true-label reference performed better but used some evaluation labels during training. |
| **[C. Training repeatability](experiments/robustness/README.md)** | Do fixed validation data or disabling network dropout reduce run-to-run variability? | Neither intervention consistently reduced F1 variability. For MOB2 at 1 kb, the standard deviations across five matched fits per group were **0.0090** (reference), **0.0185** (fixed validation), and **0.0293** (fixed validation without dropout). |

These findings highlight the challenge of balancing plasmid detection and false positives, the sensitivity of classification to training-label choices, and the difficulty of obtaining consistent results across repeated fits. The experiment READMEs provide full result tables, condition definitions, and reproduction instructions.

## Installation and reproducibility

To validate the included numerical results without installing the model dependencies, run from the repository root using **Python 3.8 or later**:

```bash
python3 scripts/results.py validate
```

The [GraSSPlas reproduction guide](experiments/grassplas/README.md#how-to-run-or-verify) explains how to obtain the documented prepared inputs, train the model, and independently score the included prediction files. The workflow **does not require regenerating every intermediate input from raw sequencing data**.

For a Linux x86-64 CPU environment, the documented setup and synthetic smoke test are:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==1.13.1+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/grassplas/requirements.txt
python tests/smoke_grassplas.py
```

The adapted plASgraph2 studies include experiment configurations, launchers, saved results, and evaluation tools. **Their training requires an authorized external adaptation of plASgraph2.** See the [pseudolabel/feature experiment guide](experiments/weak-supervision/README.md#how-to-run-or-verify). The [repeatability guide](experiments/robustness/README.md#how-to-run-or-verify) reproduces analysis of saved results, but exact historical intervention retraining is not available.

## Limitations

Training and evaluation used contigs from the same collection of assemblies; the results do not establish performance on entirely new bacterial samples. A controlled experiment without assembly links would be necessary to isolate the value of graph connectivity. Missing historical training states and intervention-specific source revisions also limit exact reproduction of some original fits.

## References and acknowledgments

- [GraSSRep](https://github.com/aliaaz99/GraSSRep) provided the original graph-based repeat-detection workflow. GraphSAGE was introduced by [Hamilton, Ying, and Leskovec](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html).
- Sielemann and colleagues developed [plASgraph2](https://doi.org/10.3389/fmicb.2023.1267695). Mane and colleagues developed the [PLASEVAL](https://doi.org/10.1186/s12859-024-05941-0) evaluation resource.
- The research was conducted at Simon Fraser University with Cédric Chauve, Mahsa Feizrahnemoon, and collaborators. Existing classifier authors and data contributors retain their respective attribution and licensing terms.
