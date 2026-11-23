# GraSSPlas

GraSSPlas investigates whether short-read assembly graphs can improve **plasmid contig classification**. Fragmentation can leave individual contigs with weak or conflicting classifier scores. The project tested whether nearby contigs and partial training labels could add enough context to improve on the original classifiers.

This portfolio documents Pegah’s 2024 research work at Simon Fraser University. It includes the custom PyTorch Geometric pipeline, the collaborative Fall experiments adapting plASgraph2, and the evidence behind the reported scores. See the [contribution and source record](docs/provenance.md).

## From classifiers to graph predictions

The August pipeline combines 52 *E. coli* assemblies into a disjoint graph of 12,176 contigs. Each contig has ten features from PlasForest, plASgraph2, Platon, RFPlasmid and mlplasmids. The model uses a single GraphSAGE layer, batch normalization, ReLU and a binary classification head. The loss uses 1,093 contigs with exactly one chromosome or plasmid pseudolabel. The classifier predicts across the full graph. The historical driver also contains an optional random forest over the original features and learned embeddings.

| Research stage | Implementation |
| --- | --- |
| Inherited GraSSRep | Repeat detection, graph construction and simulation code inherited through the original fork |
| July 2024 prototype | Two GraphSAGE layers and baseline agreement analyses, preserved from `Experiments` |
| August 2024 pipeline | One GraphSAGE layer, filtered inputs, revised pseudolabel mask and five final saved predictions |
| Fall 2024 adaptation | TensorFlow/Spektral plASgraph2 backbone with external scores, six training conditions and randomness diagnostics |
| Later work | PlasBin-flow seeding and LongClass appear in neighboring projects with separate evidence and evaluation cohorts |

See the [August methodology](docs/august.md) and the [Fall experiment definitions](experiments/fall/README.md).

## Results

### August GraphSAGE and classifiers

Pooled plasmid F1 uses the same gold and length cutoff for each method. The GraphSAGE range covers the last fit saved by each of five array jobs. Baseline scores are constant across those jobs.

| Method | ≥100 bp | ≥1 kb |
| --- | ---: | ---: |
| GraSSPlas GraphSAGE | 0.3353–0.4640 | 0.5624–0.6068 |
| PlasForest | **0.6001** | 0.7146 |
| plASgraph2 | 0.4548 | 0.4876 |
| Platon | 0.4043 | **0.8137** |
| RFPlasmid | 0.4712 | 0.8037 |
| mlplasmids | 0.3493 | 0.6912 |

The strongest baseline outperformed GraphSAGE at both cutoffs. In the best saved GraphSAGE run at 1 kb, the model recovered 571 of 611 plasmid contigs and made 700 false-positive calls. The corrected evaluated total is **4,440 contigs**, including **3,129 true negatives**. The 1 kb cutoff selects a subset of existing predictions. See the [full confusion matrices](results/august_metrics.csv) and [error analysis](docs/august.md#evaluation-and-errors).

### Fall plASgraph2 adaptation

The initial series had 25 fits per condition. Values are mean plasmid F1 ± sample standard deviation across fits on the same graphs.

| Condition | ≥100 bp | ≥1 kb |
| --- | ---: | ---: |
| MOB | 0.6504 ± 0.0317 | 0.7402 ± 0.0271 |
| MOB2 | 0.6839 ± 0.0375 | 0.8597 ± 0.0146 |
| Pfam | 0.5514 ± 0.0304 | 0.5872 ± 0.0346 |
| Feature dropout | 0.6308 ± 0.0202 | 0.6499 ± 0.0306 |
| Random-label control | 0.2018 ± 0.0264 | 0.1727 ± 0.0292 |
| True-label reference | 0.8547 ± 0.0080 | 0.9367 ± 0.0072 |

MOB2 had the highest mean F1 among the weak-supervision conditions. The true-label reference uses a subset of the evaluation labels as training targets. Together with 60 diagnostic fits, the Fall record contains 210 model/prediction pairs. Variation persisted after fixing the validation split and after disabling architectural dropout. See the [Fall analysis and results](experiments/fall/README.md).

Both evaluations are transductive. The August and Fall models use different features, cohorts and gold, so their F1 scores cannot isolate an architectural effect. The experiments do not measure performance on unseen isolates.

## Reproduce the verified checks

Run from a Git checkout with Python 3.8 or newer. These checks use the standard library.

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/validate_results.py
python3 scripts/summarize_seeds.py --group-by saved_seed --threshold 1000
```

The input preparer reads pinned data from the existing GitHub branches and writes a portable runtime manifest. With the August environment installed, the synthetic smoke test exercises the actual historical driver on CPU.

```sh
git fetch origin data pseudolabels
python3 scripts/prepare_inputs.py work/august
python3 tests/smoke_august.py
python3 scripts/inspect_august.py work/august/manifest.csv
```

See [installation, input preparation and training commands](docs/reproduction.md) and the [historical evidence gaps](docs/missing-evidence.md).

## Repository map

| Path | Contents |
| --- | --- |
| [src/august-2024/](src/august-2024/) | Exact historical GraphSAGE driver and helper |
| [legacy/](legacy/) | Exact July prototype and inherited GraSSRep source |
| [experiments/](experiments/) | Historical August launcher and manifest, Fall fit ledger and recorded configuration |
| [data/](data/) | Ordered August sample manifest and input checksums |
| [scripts/](scripts/) | Portable manifests, execution wrappers and output validation |
| [results/](results/) | Verified aggregate metrics, figures and error counts |
| [docs/](docs/) | Methodology, provenance, attribution and recovery limits |
| [tests/](tests/) | Lightweight result, input and model smoke tests |

## Credits

The project builds on GraSSRep, GraphSAGE, the PLASEVAL data and evaluation work, plASgraph2, and the five classifier tools evaluated here. Fall experiments depended on shared laboratory inputs and training infrastructure. See [upstream sources, project roles and licensing](docs/provenance.md).
