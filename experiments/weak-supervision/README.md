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

The graph convolutions and TensorFlow/Spektral model are inherited from plASgraph2. The study uses the laboratory adaptation for joining classifier features and training labels to graph nodes, training, classification and evaluation. Its `exp/grassplas/process-files.pl` is specific to this study and prepares the condition tables. The public plASgraph2 release lacks the required training and classification options and the compatible evaluator. The launcher here requires the laboratory adaptation.

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

### Compatible implementation

With project access on Fir, use the laboratory's `plasgraph2-working` checkout. Otherwise, request checkout and input-table access from the laboratory maintainers. The compatible Fir checkout is based on `3d35f0074e66f3382fbef79a14cc24db479d03e4` and has local edits in `src/architecture.py` and `src/plASgraph2_train.py`. This identifies the compatible interface. Exact source states for the historical fits remain uncertain. No public redistribution grant for the laboratory adaptation was established.

Use a checkout whose `src/plASgraph2_train.py` accepts `-f` for external features and `-G` for training targets, whose `src/plASgraph2_classify.py set` accepts `-f`, and whose `src/eval.py` supports `-m`, `-o` and `-n`. Keep the rest of the adapter's `src/` alongside these entry points for their imports.

Use the adapter's pinned `requirements.txt` in a separate Python 3.10 environment. It specifies TensorFlow 2.8.0, Spektral 1.1.0, NumPy 1.22.2, pandas 1.4.1, scikit-learn 1.0.2, SciPy 1.8.0, NetworkX 2.8.3, Biopython 1.79, Matplotlib 3.5.1, protobuf 3.19.4 and PyYAML 6.0.

```sh
PLASGRAPH2_ADAPTER=/path/to/authorized/plasgraph2-working
PLASGRAPH2_INPUTS="$PLASGRAPH2_ADAPTER/exp/grassplas"
mkdir -p work
python3.10 -m venv work/plasgraph2-env
. work/plasgraph2-env/bin/activate
python -m pip install -r "$PLASGRAPH2_ADAPTER/requirements.txt"
```

### Prepared inputs

The `pseudolabels` branch at `0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6` provides the 51 compressed graphs as `data/Ecoli_AssemblyGraph/<Assembly_Name>/short_read.gfa.gz`, the prepared contig score and label source table `data/Ecoli_contig_details.tsv`, and the sample mapping `data/EColi_unambiguous.tsv`. [Pinned public inputs](https://github.com/PegahAdt/GraSSPlas/tree/0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6/data)

| Input | Expected format |
| --- | --- |
| Cohort manifest | `sets/plaseval-nolabels.csv` in the adapter checkout. A headerless CSV of `relative_gfa,,sample_id`. Graph paths are relative to the adapter's `data/`. The middle field is empty because `-G` supplies a common training-target table. |
| Assembly graphs | The original manifest points to `data/plaseval/ecol-<BioSample>/plaseval-short.gfa.gz`. To stage a public copy, use `Assembly_Name` and `NCBI BioSample` from the sample mapping and place each `short_read.gfa.gz` at its manifest path. Sample IDs are `ecol-<BioSample>`. |
| Feature CSV | `<prefix>-extfeatures.csv` with `sample,contig` followed by the nine numeric score columns in [`model.yaml`](model.yaml). |
| Training targets | `<prefix>-pseudolabels.csv` with `sample,contig,length,label`. Length is in base pairs. Labels are `plasmid`, `chromosome`, `ambiguous` or `unlabeled`. |
| Evaluation reference | `mob-ground-truth.csv`, with the same columns and label vocabulary as the target table. This reference is shared across all conditions. |

Use consistent `sample,contig` identifiers across the graph and the three tables. The launcher takes a condition name and finds its matching file pair in `PLASGRAPH2_INPUTS`.

| Launcher condition | File prefix | Prepared inputs |
| --- | --- | --- |
| `mob` | `mob` | Saved tables or deterministic preparation |
| `mob2` | `mob2` | Saved tables or deterministic preparation |
| `pfam` | `pfam` | Saved tables or deterministic preparation |
| `feature_masking` | `dropout` | Saved feature and target tables preserve the original masking draw |
| `random` | `random` | Saved feature and target tables preserve the original random labels |
| `true` | `true` | Saved feature and target tables preserve the original sampled true labels |

The authorized checkout's `exp/grassplas/` contains the saved condition tables. Use them directly when available. If only the public source tables are available, the authorized `process-files.pl` can regenerate the MOB, MOB2 and Pfam tables. In a scratch directory, save `Ecoli_contig_details.tsv` as `plaseval-contigs.tsv` and `EColi_unambiguous.tsv` as `plaseval-sample-overview.tsv`. Run the Perl script from that directory with `mob`, `-c 0.1 mob2` and `-p p pfam` respectively. The MOB invocation also writes the common reference. Re-running the randomized conditions with the arguments in [`conditions.json`](conditions.json) makes new draws.

### Train or score

From the repository root, supply the authorized adapter, prepared input directory and a new output directory. The example uses [`model.yaml`](model.yaml). Pass `--config /path/to/model.yaml` to choose a different settings file. A successful run writes the model, `predictions.csv`, training logs and both threshold metric files. Graph path resolution has been checked for all 51 entries for both launcher calls. Full training through this launcher remains unverified.

```sh
python experiments/weak-supervision/run.py "$PLASGRAPH2_ADAPTER" "$PLASGRAPH2_INPUTS" mob2 work/mob2
```

The summary and independent scoring commands use Python's standard library. To score an existing prediction CSV, provide `sample,contig,plasmid_score,chrom_score` columns and the common reference. The adapter also writes `length` and `label` in its predictions. Scoring joins by sample and contig, uses the reference lengths, and reports metrics at 100 bp and 1 kb.

```sh
python scripts/results.py summary weak-supervision
python scripts/results.py score-plasgraph "$PLASGRAPH2_INPUTS/mob-ground-truth.csv" /path/to/predictions.csv
```

## Limitations

All reported fits evaluate the same graphs used during training. No independent test on unseen isolates is established. The single-fit comparison supplies aggregate measurements, while its original predictions, model artifacts and random selections are unavailable. Reproducing each repeated fit also requires its original source and random state. [Repeatability and intervention limits](../robustness/README.md#limitations)
