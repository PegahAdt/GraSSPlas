# B. Comparison of pseudolabeling strategies

## Research question

With the plASgraph2 architecture and evaluation cohort held constant, how does performance change when training uses different automatic labels, altered classifier inputs, random labels, or a small number of true labels?

## Plain-English explanation

The same assembly graphs can support several choices of training examples. MOB and Pfam use different clues to mark some contigs as plasmid or chromosome. MOB2 changes how many contigs qualify for a chromosome target. The feature-masking condition keeps MOB's labels but substitutes default input scores for some labeled contigs. Random labels and a subset of true labels give two reference points for interpreting those choices.

## Model and implementation

The adapted model uses the TensorFlow/Spektral **plASgraph2** architecture. Its graph convolutions combine node information along assembly links. The saved settings specify six graph layers that reuse weights, 32 internal graph features and two independent sigmoid outputs. Each output is a score between 0 and 1. The trainer uses weighted binary cross-entropy, which measures error against the two binary targets. It assigns zero training weight to unlabeled nodes while leaving them and their links available to the graph model. A contig can be labeled as both plasmid and chromosome. [plASgraph2 paper](https://doi.org/10.3389/fmicb.2023.1267695) · [Upstream implementation](https://github.com/cchauve/plASgraph2)

| Included here | What it does |
| --- | --- |
| [`conditions.json`](conditions.json) | Specifies each input prefix and the arguments used to describe its preparation. |
| [`model.yaml`](model.yaml) | Records shared model and optimizer settings and an example seed. |
| [`run.py`](run.py) | Passes prepared tables to an external trainer, classifier and evaluator. |
| [`scripts/results.py`](../../scripts/results.py) | Scores supplied predictions independently and summarizes the committed metrics. |

The adaptation's feature assembly and label-generation script `process-files.pl`, graph builder, TensorFlow model, trainer and classifier require an authorized external checkout. The JSON and launcher do not implement those steps. The public upstream release explains the base model. Support for the adaptation's external-feature and pseudolabel arguments must be checked in the authorized checkout.

## Dataset and training labels

The study uses 51 *E. coli* isolates and predicts for 11,984 contigs per fit. The nine model inputs are scores from the same five classifier families as Experiment A, using only the plasmid score from mlPlasmids. Training targets come from the condition tables. MOB-recon supplies plasmid matches and Pfam supplies evidence based on protein families. Reference labels used for evaluation are separate except where the true-label reference deliberately reuses them as training targets.

The evaluator treats ambiguous reference contigs as positive for both plasmid and chromosome. It omits contigs with unknown reference status. This leaves 11,766 contigs at 100 bp and 4,348 at 1 kb. Among contigs shared with GraphSAGE, 116 called chromosome-only in this reference are plasmid-positive in the custom binary reference. The scores from the two model families use different evaluation definitions.

## Experimental design

| Condition | What changes |
| --- | --- |
| MOB | Use MOB-recon matches for plasmid labels and strict Pfam calls for chromosome labels. |
| MOB2 | Keep the MOB plasmid criterion and flag chromosome when the Pfam chromosome score exceeds 0.1. This changes the training labels. The evaluation score cutoff stays at 0.5. |
| Pfam | Use Pfam for both plasmid and chromosome labels. |
| Feature masking | Keep MOB labels. Replace classifier scores with defaults for pseudolabeled contigs selected with probability 0.2. The same modified table is used in training and prediction. |
| Random-label control | Sample contigs with probability 0.1 and assign random plasmid or chromosome targets. |
| True-label reference | Sample contigs with probability 0.1 and supply their reference labels when available. |

The repeated comparison uses the same model architecture, assemblies, reference labels, evaluation rules and length cutoffs. Each condition has 25 fits. The conditions change the source or number of selected targets, the score inputs in the masking condition, or which reference labels can enter training. The single-fit comparison is a separate series. The 25-fit means use only the repeated comparison.

## Results

Scores above 0.5 are called positive for the molecule being evaluated. Metrics pool contigs within a fit before averaging across fits. The evaluator's optimistic convention assigns precision or recall a value of 1 when its denominator is zero. The table shows plasmid F1 as mean ± sample standard deviation.

| Condition | F1 ≥100 bp | F1 ≥1 kb |
| --- | ---: | ---: |
| MOB | 0.6504 ± 0.0317 | 0.7402 ± 0.0271 |
| MOB2 | 0.6839 ± 0.0375 | 0.8597 ± 0.0146 |
| Pfam | 0.5514 ± 0.0304 | 0.5872 ± 0.0346 |
| Feature masking | 0.6308 ± 0.0202 | 0.6499 ± 0.0306 |
| Random-label control | 0.2018 ± 0.0264 | 0.1727 ± 0.0292 |
| True-label reference | 0.8547 ± 0.0080 | 0.9367 ± 0.0072 |

At 1 kb, MOB2 increased mean precision from **0.6039** to **0.9396** relative to MOB, while mean recall decreased from **0.9588** to **0.7929**. In the separate single-fit comparison, MOB2 had 1 kb F1 **0.8519** on the reference labels, and the true-label condition had **0.9299**. The single-fit Pfam model reached **1.0000** F1 against its own pseudolabels at 100 bp, compared with **0.4918** against the reference. Matching the provisional training labels can give a very different result from measuring classification against the reference. [Per-fit results, single-fit comparison and external baselines](results.csv)

## Conclusions

MOB2 had the best mean plasmid F1 of the automatic-label strategies at both cutoffs on this cohort. Its higher precision outweighed its lower recall relative to MOB. The true-label reference scored higher than MOB2 while having access to some of the labels used for evaluation. The chromosome cutoff changed which contigs received a target and how many targets were available. The comparison cannot tell which of those changes drove the result.

## How to run or verify

From the repository root, use Python to reproduce the summary. With access to score and reference files, independently score each prediction vector.

```sh
python scripts/results.py summary weak-supervision
python scripts/results.py score-plasgraph /path/to/gold.csv /path/to/predictions.csv
```

Training requires an authorized adaptation checkout, its `sets/plaseval-nolabels.csv`, assembly graphs, and prepared feature, pseudolabel and reference tables. The study used TensorFlow 2.8.0, Spektral 1.1.0, NumPy 1.22.2, pandas 1.4.1 and scikit-learn 1.0.2. Follow the adaptation's dependency file in a separate environment.

In the adaptation, `process-files.pl` builds the condition tables with the arguments in [`conditions.json`](conditions.json). Supply the resulting `<prefix>-extfeatures.csv`, `<prefix>-pseudolabels.csv` and `mob-ground-truth.csv` in the input directory. For example, the command below calls the adaptation's trainer and classifier once with the included example settings, then writes threshold metrics.

```sh
python experiments/weak-supervision/run.py /path/to/adaptation /path/to/inputs mob2 work/mob2
```

## Limitations

All reported fits evaluate the same graphs used during training. No independent test on unseen isolates is established. The single-fit comparison supplies aggregate measurements, while its original predictions, model artifacts and random selections are unavailable. Reproducing each repeated fit also requires its original source and random state. [Repeatability and intervention limits](../robustness/README.md#limitations)
