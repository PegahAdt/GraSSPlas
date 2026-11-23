# C. Training repeatability

## Research question

When the adapted plASgraph2 model is fitted repeatedly on the same assemblies, how much do the scores change? Do the observed differences narrow under protocols intended to hold the validation split fixed or to also disable network dropout?

## Plain-English explanation

A model can produce different predictions when fitted twice, even with the same input data. A **seed** is the number used to initialize a random-number generator. Recording one seed does not by itself control every source of randomness. A **validation split** holds aside part of the labeled data to check training behavior. **Network dropout** randomly sets some intermediate model values to zero during training. The interventions were intended to check whether these two mechanisms explained the run-to-run variation.

## Model and implementation

All three protocols use the same adapted plASgraph2 model family and six conditions as Experiment B. Network dropout concerns activations inside the model. Experiment B's feature masking alters the classifier scores supplied as input. The feature-masking condition is present in all three protocols.

[`protocols.json`](protocols.json) describes the experimental groups and repeat IDs. [`scripts/results.py`](../../scripts/results.py) groups and summarizes the scores. [`scripts/plot_results.py`](../../scripts/plot_results.py) creates the variability figure. These analysis files do not implement the intervention changes. [Model and implementation for Experiment B](../weak-supervision/README.md#model-and-implementation)

## Dataset and training labels

The study repeats training on the 51-isolate cohort from Experiment B, with the same six condition definitions and the same reference labels for evaluation. The analysis scores the same two minimum lengths and measures variability of pooled plasmid F1 across fits. The repeats do not sample new isolates or perturb the data. [Cohort and labels](../weak-supervision/README.md#dataset-and-training-labels)

## Experimental design

| Protocol | Intended change | Fits |
| --- | --- | ---: |
| Reference | Five repeats for each condition and each seed group named 13, 20, 29, 37 and 123. | 150 |
| Fixed validation | Hold the training/validation split fixed while leaving other randomness active. Five repeats per condition at seed 123. | 30 |
| Fixed validation with network dropout disabled | Keep the intended fixed split and disable network dropout layers. Five repeats per condition at seed 123. | 30 |

The reference measurements are stored with [Experiment B's results](../weak-supervision/results.csv). The [diagnostic table](results.csv) stores the two intervention groups. Each records both the seed used to name the run and the seed value saved in its configuration. For the direct comparison below, both must equal 123.

## Results

The figure compares the sample standard deviation of F1 across five fits per condition and protocol at each cutoff. The table gives both the mean and standard deviation for MOB2.

![Standard deviation of plasmid F1 by protocol for all six conditions at both minimum contig lengths](variability.png)

| MOB2 at seed 123 | Mean F1 ± SD ≥100 bp | Mean F1 ± SD ≥1 kb |
| --- | ---: | ---: |
| Reference | 0.6838 ± 0.0372 | 0.8462 ± 0.0090 |
| Fixed validation | 0.7004 ± 0.0507 | 0.8484 ± 0.0185 |
| Fixed validation with network dropout disabled | 0.6156 ± 0.1464 | 0.8539 ± 0.0293 |

In the matched-seed comparison the fixed-validation group had higher variability than the reference in all six conditions at both cutoffs. For the group that also disabled dropout, only Pfam at 1 kb had lower variability than the reference. For MOB2 at 100 bp, the standard deviation increased from 0.0372 to 0.0507 to 0.1464 across the three protocols.

## Conclusions

Repeat-to-repeat variation remained even at the same recorded seed. Neither intervention consistently reduced the spread in these five-fit comparisons. Differences in initialization, random-number handling, the validation code and execution conditions were not independently controlled, so the observed spread does not identify its cause. The experiment addresses repeatability of training on this cohort.

## How to run or verify

From the repository root, reproduce the matched seed-123 comparison and inspect all fits grouped by their saved seed.

```sh
python scripts/results.py summary robustness
python scripts/results.py summary robustness --by-seed configured_seed
```

If the original prediction and reference files are available, the same independent scoring command used for Experiment B applies here. These commands reproduce the analysis. Replaying an intervention requires its exact model and training code, in addition to the [external adaptation and input tables](../weak-supervision/README.md#how-to-run-or-verify).

## Limitations

In 33 of 210 fits, the saved configuration seed differs from the run name. All 30 fits named for seed 29 record seed 20. Three more fits named for seeds 13 or 37 record seed 123. The matched-seed comparison excludes these mismatches.

The experiment notes describe the intended interventions. The base revision `127692edfb821a1d6eff9aa081cf594b48e12ff3` and the exact locally edited versions are unavailable. Saved configurations retain a dropout setting of 0.1 even for the no-dropout protocol. They also show `use_validation: false`, which in the available trainer selects training loss for early stopping. That trainer assigns zero validation weights, while several saved runs have nonzero validation loss. The settings and available source alone cannot establish what each run did. With only five fits per condition in the matched comparison, the standard deviations describe the measurements and provide limited evidence about a causal change in variability.
