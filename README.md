# GraSSPlas — Graph-Based Semi-Supervised Learning for Plasmid Classification

## Project overview

GraSSPlas studies whether a machine-learning model can identify plasmid DNA by combining existing classifier scores with assembly connections. The study also tests how training labels and repeated fits affect the results.

## Background on plasmid classification

A **plasmid** is a DNA molecule separate from a bacterium’s chromosome. Some plasmids carry genes for antibiotic resistance. Identifying plasmid DNA helps researchers study how resistance can spread between bacteria. [PLASEVAL study](https://doi.org/10.1186/s12859-024-05941-0)

Sequencers read DNA in short pieces. Assembly software joins them into longer stretches called **contigs**. An **assembly graph** records the contigs and their possible connections. The task is to classify each contig by whether it came from a plasmid.

## Motivation and research questions

Classifiers can disagree because they use different evidence or decision rules. A short fragment may offer few clues. Linked contigs may come from the same molecule, so evidence for one could help classify the other. The study tests whether a model can use these clues together.

The study starts with a custom classifier, then uses an established graph model to compare training inputs and check how much its scores vary between runs.

## Proposed semi-supervised learning approach

1. **Start with bacterial assemblies.** PLASEVAL provides *E. coli* samples with assembled contigs and their connection records.
2. **Represent the assemblies as graphs.** Contigs are nodes and assembly connections are edges. The custom model uses graphs prepared with a 100 base-pair minimum.
3. **Collect scores from existing classifiers.** The five tools are **PlasForest, Platon, RFPlasmid, plASgraph2, and mlPlasmids**. Each supplies a plasmid and a chromosome score, giving GraphSAGE ten input features per contig.
4. **Choose provisional training labels.** A **pseudolabel** is a provisional label assigned by a rule. Matches to known plasmid-associated sequences give plasmid flags. Chromosome flags require a minimum length and sequencing depth below a cutoff. Depth measures how many reads cover a fragment. The cutoff is the assembly mean plus twice a measure of spread called the standard deviation. The model trains on contigs with exactly one flag. A separate set of reference labels is used for evaluation. [Labeling rules and source code](experiments/graphsage/README.md#dataset-and-training-labels)
5. **Learn and predict with GraphSAGE.** The graph neural network combines a contig's scores with scores from its linked neighbors. It processes the full graph, but the training loss measures errors only on the **1,093 of 12,176 contigs** with selected pseudolabels. Contigs with neither or both flags remain available as neighbors. This use of a partly labeled graph is semi-supervised learning. The trained model predicts plasmid or chromosome for every contig.
6. **Compare predictions with the reference.** **Precision** is the fraction of plasmid calls that were correct. **Recall** is the fraction of reference plasmid contigs that were found. **F1** combines the two and is high when both are high. The study scores GraphSAGE and the five individual tools on the same contigs. Scoring at both 100 base pairs (**bp**) and 1,000 base pairs (**1 kb**) shows how excluding shorter fragments changes the comparison.

## Dataset and experimental setup

The custom **GraphSAGE** experiment uses 52 assemblies and treats a contig as plasmid-positive if it appears on the reference plasmid list. The other experiments use **adapted plASgraph2**, an existing graph model extended to accept additional classifier scores and alternative training labels. It uses 51 bacterial samples and gives separate plasmid and chromosome scores. Its reference allows both labels.

Each model family is evaluated on its own cohort. The 1 kb cutoff scores a subset of existing predictions.

## Experiments and results

### A. GraphSAGE classification

**Question.** How well can a model combine the five classifiers' scores, assembly links, and a small set of training labels to classify plasmid contigs?

At each cutoff, the methods use the same contigs and reference labels. The GraphSAGE ranges cover five saved predictions.

| Method | Plasmid F1 at ≥100 bp | Plasmid F1 at ≥1 kb |
| --- | ---: | ---: |
| GraphSAGE | 0.3353–0.4640 | 0.5624–0.6068 |
| PlasForest | **0.6001** | 0.7146 |
| Platon | 0.4043 | **0.8137** |
| RFPlasmid | 0.4712 | 0.8037 |
| plASgraph2 | 0.4548 | 0.4876 |
| mlPlasmids | 0.3493 | 0.6912 |

At 1 kb, the best-F1 GraphSAGE fit found 571 reference plasmid contigs and made 700 false-positive calls. Platon found 428 and made 13 false-positive calls. GraphSAGE found more plasmid contigs, but at a cost in precision and F1.

**Conclusion.** The strongest baseline achieved higher F1 at both cutoffs. The comparison shows where the custom pipeline could improve by making fewer false-positive calls. A matched run with the edges removed would be needed to measure what the connections contribute. [Full precision, recall, and F1 results](experiments/graphsage/README.md#results)

### B. Pseudolabel and feature-input strategies

**Question.** How sensitive is classification to the training labels and classifier scores supplied to the model?

The first experiment's precision–recall tradeoff motivated a closer look at those choices. This comparison uses the **adapted plASgraph2** model. The six conditions share the same assembly collection, model architecture, and evaluation reference.

| Condition | What changes | Plasmid F1 at ≥1 kb, mean ± SD |
| --- | --- | ---: |
| MOB | Uses MOB-recon matches for plasmid labels and strict Pfam calls for chromosome labels. Pfam provides evidence from protein families. | 0.7402 ± 0.0271 |
| MOB2 | Keeps the MOB plasmid rule and uses a lower Pfam chromosome cutoff, which labels more contigs as chromosome. | 0.8597 ± 0.0146 |
| Pfam | Uses Pfam for both plasmid and chromosome labels. | 0.5872 ± 0.0346 |
| Feature masking | Keeps the MOB labels and replaces scores with defaults for some labeled contigs. The modified scores are used in training and prediction. | 0.6499 ± 0.0306 |
| Random-label control | Assigns random classes to a sample of contigs to check performance when the training labels carry no biological information by design. | 0.1727 ± 0.0292 |
| True-label reference | Gives the model some of the reference labels to show what happens when true labels are available for training. | 0.9367 ± 0.0072 |

Each row summarizes **25 fits**. A higher standard deviation (SD) means F1 varied more across runs.

**Conclusion.** MOB2 had the highest mean F1 among the automatic strategies. Compared with MOB, it made more reliable plasmid calls, and the gain in precision outweighed a drop in recall. The true-label reference had higher F1 and used some evaluation labels that the automatic strategies did not receive. [Results at both lengths and the separate single-fit comparison](experiments/weak-supervision/README.md#results)

### C. Training repeatability

**Question.** Do repeated plASgraph2 fits give similar results, and can fixing validation or disabling dropout make them more consistent?

Each repeat fits a new model on the same data. A **random seed** sets the starting point for random choices such as initial weights or data selection. **Validation data** are set aside to monitor training or decide when to stop. **Network dropout** randomly switches off values inside the model during training. Feature masking changes the input scores.

The study compared a reference series with two interventions. The first was meant to fix the validation data. The second was meant to also disable network dropout. All groups used the same six conditions, data, and scoring rules. The table uses five MOB2 fits per group with recorded seed 123.

| Protocol | F1 SD at ≥100 bp | F1 SD at ≥1 kb |
| --- | ---: | ---: |
| Reference | 0.0372 | 0.0090 |
| Fixed validation | 0.0507 | 0.0185 |
| Fixed validation with network dropout disabled | 0.1464 | 0.0293 |

**Conclusion.** Neither intervention consistently reduced variability across the six conditions. The variation persisted on the same assemblies. The exact intervention code and other random choices cannot be verified for every fit, leaving the cause unresolved. [Protocol details and results by condition](experiments/robustness/README.md#results)

## Overall findings

On these assemblies, more plasmid calls did not necessarily mean better classification. The plASgraph2 comparisons show that training labels can shift the balance between missed contigs and false positives. Repeat runs show how much that balance can vary under the same condition.

## Installation and reproducibility

From the repository root, check the results and recompute the summaries with Python 3.8 or later. These commands need no machine-learning packages.

```sh
python3 scripts/results.py validate
python3 scripts/results.py summary weak-supervision
python3 scripts/results.py summary robustness
```

For the custom model, create a Python 3.8 environment and install PyTorch and the experiment dependencies. The following is a Linux x86-64 CPU example. The smoke test runs one training epoch on a small synthetic graph.

```sh
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==1.13.1+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/graphsage/requirements.txt
python tests/smoke_graphsage.py
```

The GraphSAGE guide covers public data preparation and full training. The plASgraph2 guide explains which external code and prepared tables are required for training and how to verify predictions independently. [GraphSAGE instructions](experiments/graphsage/README.md#how-to-run-or-verify) · [plASgraph2 instructions](experiments/weak-supervision/README.md#how-to-run-or-verify) · [Repeatability analysis](experiments/robustness/README.md#how-to-run-or-verify)

## Repository organization

- [GraphSAGE experiment](experiments/graphsage/) contains the custom model, input manifest, launcher, and baseline results.
- [Pseudolabel and feature-input experiment](experiments/weak-supervision/) contains the plASgraph2 condition settings, external launcher, and results.
- [Repeatability experiment](experiments/robustness/) contains the comparison protocols and intervention results.
- [Scoring and validation](scripts/results.py) and [tests](tests/) provide shared checks.

## Limitations

Models were trained and evaluated using contigs from the same collection of bacterial assemblies. These results do not establish how accurately the models would classify DNA fragments from new bacterial samples. The two pipelines also use different assemblies and reference rules, so their scores cannot determine which model architecture is better. Missing training states and intervention code limit which original fits can be reproduced exactly.

## References and acknowledgments

- GraSSRep provided the original graph-based repeat-detection workflow. Hamilton, Ying, and Leskovec introduced GraphSAGE. [GraSSRep](https://github.com/aliaaz99/GraSSRep) · [GraphSAGE paper](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
- Sielemann, Sielemann, Brejová, Vinař, and Chauve developed plASgraph2. Mane and colleagues developed PLASEVAL. [plASgraph2 paper](https://doi.org/10.3389/fmicb.2023.1267695) · [PLASEVAL paper](https://doi.org/10.1186/s12859-024-05941-0)
- The research was conducted at Simon Fraser University with Cédric Chauve, Mahsa Feizrahnemoon, and collaborators. The classifier authors and data contributors retain their respective attribution and terms.
