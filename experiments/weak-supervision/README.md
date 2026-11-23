# Weak-supervision strategies

How does the choice of training target affect contig classification on the same graphs? This experiment adapts the TensorFlow/Spektral plASgraph2 architecture to use nine external scores and compares training targets or feature inputs. It uses the upstream backbone with six tied graph layers and a two-score sigmoid head. The project-specific work is the integration of external features, assignment of training targets and experimental comparisons. [plASgraph2 paper](https://doi.org/10.3389/fmicb.2023.1267695) · [Upstream implementation](https://github.com/cchauve/plASgraph2)

## Conditions

| Condition | Definition |
| --- | --- |
| MOB | Use MOB-recon plasmid matches and strict Pfam chromosome calls. |
| MOB2 | Keep MOB plasmid targets and accept Pfam chromosome scores above 0.1. The study notes describe the reference cutoff as 0.5. |
| Pfam | Use Pfam plasmid and chromosome calls. |
| Feature masking | Use MOB targets and replace the external scores with defaults for a randomly selected subset of pseudolabeled contigs. The selection fraction is 0.2. |
| Random-label control | Assign random classes to a sample of about 10% of contigs. |
| True-label reference | Disclose gold labels for a random subset of about 10% of contigs. |

The shared [model configuration](model.yaml) records the architecture and optimizer settings. The [condition definitions](conditions.json) identify preparation arguments and the corresponding input filenames. The preparation command can reproduce the intended rule when an authorized copy of the adaptation and inputs is available. Random selections depend on the historical inputs and random state.

## Evaluation and results

This transductive experiment uses 51 *E. coli* isolates and produces 11,984 predictions per fit. The binary chromosome and plasmid scores are evaluated separately. Ambiguous gold counts positive for both. Unlabeled contigs are excluded, leaving 11,766 labeled contigs at 100 bp and 4,348 at 1 kb. Precision, recall, F1, accuracy and ROC AUC are pooled per fit. Called positives have scores above 0.5. For a zero denominator, the original optimistic evaluator sets the undefined precision or recall to 1.

The table shows plasmid F1 as mean ± sample standard deviation across 25 fits per condition on the same biological cohort.

| Condition | F1 ≥100 bp | F1 ≥1 kb |
| --- | ---: | ---: |
| MOB | 0.6504 ± 0.0317 | 0.7402 ± 0.0271 |
| MOB2 | 0.6839 ± 0.0375 | 0.8597 ± 0.0146 |
| Pfam | 0.5514 ± 0.0304 | 0.5872 ± 0.0346 |
| Feature masking | 0.6308 ± 0.0202 | 0.6499 ± 0.0306 |
| Random-label control | 0.2018 ± 0.0264 | 0.1727 ± 0.0292 |
| True-label reference | 0.8547 ± 0.0080 | 0.9367 ± 0.0072 |

MOB2 has the best mean F1 of the weak-supervision strategies at both cutoffs. The true-label reference exposes some evaluation labels during fitting. Its higher F1 provides a supervision reference within the same cohort. This experiment uses different inputs and gold from [GraphSAGE](../graphsage/README.md), including 116 shared contigs labeled chromosome-only here and plasmid-positive in the custom gold.

The [results CSV](results.csv) contains all 600 pooled rows from the repeated study. It also preserves a separate six-condition single-fit comparison scored on gold and selected pseudolabel sets, plus ten paired external-baseline rows on this cohort. The column `protocol` distinguishes these comparisons. Each row uses the same pooled metric definitions. The `expected` statistic sums the continuous positive scores. Blank fields are inapplicable or unavailable.

## Execution and verification

Recreating the published summaries requires only Python. With access to the original score and gold files, the scorer can independently recompute the per-fit measures.

```sh
python scripts/results.py summary weak-supervision
python scripts/results.py score-plasgraph /path/to/gold.csv /path/to/predictions.csv
```

Training needs an authorized development checkout that implements the external feature and pseudolabel arguments, its `sets/plaseval-nolabels.csv` set and GFA data, plus the prepared condition tables. The study used TensorFlow 2.8.0, Spektral 1.1.0, NumPy 1.22.2, pandas 1.4.1 and scikit-learn 1.0.2 in a separate environment. Follow the adaptation's dependency file. The public upstream release provides the base architecture, and compatibility with the adapted CLI is unverified.

With the study-specific contig table and preparation script available, preparation uses `process-files.pl [preparation_args] input_prefix`. It produces the feature and target tables described in [conditions.json](conditions.json). The launcher expects these tables and `mob-ground-truth.csv` in the input directory. For example, run MOB2 in the adaptation's environment from the repository root.

```sh
python experiments/weak-supervision/run.py /path/to/adaptation /path/to/inputs mob2 work/mob2
```

This command calls the adaptation's trainer, classifier and evaluator and creates a model directory, loss trace, prediction CSV and both threshold statistics. The authorized inputs and code are needed to run the full pipeline. Exact fit reproduction additionally requires the matching code revision, original feature tables and random state. Only aggregate results are available for the single-fit comparison. Generalization to unseen isolates was outside the design.
