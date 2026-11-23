# GraSSPlas — Graph-Based Semi-Supervised Learning for Plasmid Classification

## Overview and research motivation

GraSSPlas studies whether assembly-graph context can improve plasmid contig classification. Short-read assemblies often fragment chromosomes and plasmids into many contigs. When an individual contig has weak or conflicting classifier scores, its assembly neighbors may contribute useful evidence. The project tests that idea with a custom GraphSAGE model, then compares weak-supervision strategies and training variability using a plASgraph2 adaptation.

## Methodology and model architecture

The custom model uses PyTorch Geometric. One GraphSAGE convolution aggregates contig features across assembly links, followed by batch normalization, ReLU and a two-class head. Pseudolabels supervise only nodes with an unambiguous training target. The loss is masked and inference covers the full graph.

The second model uses the existing TensorFlow/Spektral plASgraph2 architecture. The adaptation integrates external classifier scores and tests alternative training targets and feature inputs. Its independent plasmid and chromosome outputs support ambiguous labels.

## Dataset and features

The custom GraphSAGE experiment combines 52 PLASEVAL *E. coli* assemblies into a disjoint graph of 12,176 contigs. Short contigs were contracted during graph preparation. Each retained node has ten features from the plasmid and chromosome scores of PlasForest, plASgraph2, Platon, RFPlasmid and mlplasmids. A subset of 1,093 nodes contributes to the loss. The plASgraph2 study uses nine external scores and a separate 51-isolate cohort. Its gold can mark a contig as both plasmid and chromosome. [PLASEVAL](https://doi.org/10.1186/s12859-024-05941-0) · [GraphSAGE inputs](experiments/graphsage/README.md#data-and-model) · [plASgraph2 evaluation cohort](experiments/weak-supervision/README.md#evaluation-and-results)

## Experiments

1. **GraphSAGE plasmid classification.** Measures whether graph context improves on the five classifiers on the custom cohort at 100 bp and 1 kb. PlasForest and Platon led at the two cutoffs. [Experiment and comparison](experiments/graphsage/README.md)
2. **Weak-supervision strategies.** Compares MOB, MOB2, Pfam, feature masking, random labels and a partial true-label reference. MOB2 had the highest mean F1 among the weak-supervision strategies. [Conditions and repeated-fit results](experiments/weak-supervision/README.md)
3. **Randomness and robustness.** Tests repeat variability under different nominal seeds, fixed validation and fixed validation with network dropout disabled. Variability persisted and the evidence is inconclusive about its causes. [Protocols and matched-seed comparison](experiments/robustness/README.md)

## Key results and comparisons

| Experiment | Key result |
| --- | --- |
| GraphSAGE | F1 ranged from **0.3353–0.4640** at 100 bp and **0.5624–0.6068** at 1 kb across five saved fits. PlasForest reached **0.6001** at 100 bp and Platon reached **0.8137** at 1 kb. |
| Weak supervision | At 1 kb, MOB2 had mean plasmid F1 **0.8597 ± 0.0146** across 25 fits. The standard deviation measures variation across fits on the same cohort. |
| Robustness | In the matched seed-123 MOB2 comparison at 100 bp, sample SD was **0.0372** for the reference, **0.0507** with fixed validation and **0.1464** with fixed validation and network dropout disabled. |

![Pooled plasmid F1 for GraphSAGE and five classifiers at 100 bp and 1 kb. Dots on the GraphSAGE row are the saved fits.](experiments/graphsage/comparison.png)

The custom GraphSAGE model did not exceed the strongest baseline at either cutoff. The two model families use different cohorts and gold, so their scores do not isolate an architecture effect. The experiments are transductive and provide no estimate for independent held-out isolates.

## Installation and reproducibility

Python 3.8 or later can validate the published tables without ML dependencies.

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/results.py validate
```

For GraphSAGE, the tested version combination was Python 3.8.10, PyTorch 1.13.1 and PyG 2.5.3. This example targets Linux x86-64 with a compatible CPU wheel.

```sh
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==1.13.1+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/graphsage/requirements.txt
python tests/smoke_graphsage.py
```

The GraphSAGE guide provides exact public input preparation, training and evaluation commands. The plASgraph2 launcher requires an authorized adaptation and its input tables. Recompute the weak-supervision and robustness summaries directly from the included CSVs. With Matplotlib installed, `python scripts/plot_results.py` regenerates both figures. [GraphSAGE commands](experiments/graphsage/README.md#reproduction) · [plASgraph2 commands](experiments/weak-supervision/README.md#execution-and-verification) · [Robustness verification](experiments/robustness/README.md#reproduction)

## Repository structure

| Directory | Contents |
| --- | --- |
| [experiments/graphsage/](experiments/graphsage/) | Custom executable source, ordered input manifest, results and launcher |
| [experiments/weak-supervision/](experiments/weak-supervision/) | Shared plASgraph2 settings, condition definitions, launcher and results |
| [experiments/robustness/](experiments/robustness/) | Intervention protocols, diagnostic results and variability figure |
| [scripts/](scripts/) | Result validation, independent scoring, summaries and figures |
| [tests/](tests/) | Lightweight metric tests and a synthetic GraphSAGE CPU run |

## Limitations

The trained GraphSAGE weights and fit-level random states are unavailable. The loader, objective and graph direction preserve the semantics of the evaluated implementation. The robustness protocols depend on exact code and validation behavior that cannot be reproduced from saved configuration values alone. In the adapted plASgraph2 study, 33 of 210 fits have nominal and configured seeds that disagree. Each experiment README describes the scope of its executable checks.

## References and acknowledgments

- GraSSRep supplied the original graph-based repeat-detection workflow that informed this project. [GraSSRep](https://github.com/aliaaz99/GraSSRep)
- Hamilton, Ying and Leskovec introduced GraphSAGE. The custom model uses the PyTorch Geometric implementation. [GraphSAGE paper](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) · [PyG](https://pytorch-geometric.readthedocs.io/)
- Sielemann, Sielemann, Brejová, Vinař and Chauve developed plASgraph2. Mane and colleagues developed PLASEVAL. [plASgraph2](https://doi.org/10.3389/fmicb.2023.1267695) · [PLASEVAL](https://doi.org/10.1186/s12859-024-05941-0)
- The research was conducted at Simon Fraser University with Cédric Chauve, Mahsa Feizrahnemoon and collaborators. The baseline classifier packages, external datasets and upstream code retain their respective attribution and terms.
