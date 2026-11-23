# Portfolio branch report

The branch `portfolio/plasmid-research` is based on `Experiments` at `2c991711a7301fc7c25d023f55704dba358ffeba`. This base preserves the July plasmid-specific commits and the inherited GraSSRep history. The `data` and `pseudolabels` branches serve as pinned input sources.

## What the branch contains

- The final August PyTorch Geometric driver and helper from Fir, alongside the exact Slurm launcher and sample manifest. The original file hashes are recorded. The July prototype, analysis PDF and inherited GraSSRep files were relocated byte-for-byte.
- A portable ordered manifest and 208 input checksums. The preparer pulls verified graphs, features and labels from `pseudolabels` and gold from `data`. The new preflight checks historical table alignment. The local recovery tool verifies the original graph filtering, experiment and analysis files against a checksum catalog.
- The five saved August result sets summarized as 120 metric rows, along with a baseline figure and misclassification counts. The corrected run-1 evaluation at 1 kb uses 4,440 contigs, including 3,129 true negatives.
- The Fall plASgraph2 condition definitions, 210-fit ledger, configuration profile, randomness analyses and 840 pooled metric rows. The verified tables and figure distinguish the initial study, fixed-validation diagnostic and no-dropout diagnostic. The earlier six-condition aggregate stats are included separately.
- Runnable utilities for preparing inputs, checking results, comparing seeds and launching the original August driver. The Fall wrapper documents the lab adaptation’s CLI and accepts an authorized checkout, prepared condition tables and a configuration.

See [provenance and licensing](docs/provenance.md) and the [file-level source map](docs/provenance.csv).

## Deliberate exclusions

The branch uses references and a local checksum catalog for collaborator implementation files, raw lab materials and private experiment inputs. The public history supplies data already available on GitHub. The branch contains no bulk raw data, training checkpoints, per-contig prediction dumps, cluster logs, full shared checkout or private correspondence. External baseline runs and later related studies are described separately. Exact historical copies retain their original academic contact and cluster metadata from the already-public launcher history.

## Verification

| Check | Result |
| --- | --- |
| Python syntax, shell syntax, relative Markdown links, source hashes and curated file-size scan | Passed |
| Standard-library unit tests | Passed |
| Synthetic August model CPU run using the exact historical driver | Passed for a one-epoch fit and output export |
| August input preparation | All 208 files matched the Fir SHA-256 digests |
| Historical August loader preflight | Rebuilt the 12,176 feature, label and gold rows and matched the saved run-1 export |
| Local recovery and graph contraction fixture | Verified all cataloged source hashes and contracted a synthetic short segment through the historical filter |
| Custom metric recount | All 120 saved rows matched confusion counts recomputed from the five stored final predictions |
| Fall metric recount | Re-evaluated stored scores against gold and matched 840 rows for 210 predictions, then checked the summary statistics |

Input reconstruction, evaluation and lightweight execution are reproducible. Exact historical training is partially recoverable. No full-cohort training was performed for this branch. The Fall base commit and the intervention-specific source revisions are unresolved. The August record has only the final vector from each ten-fit job and contains no trained weights. See [missing evidence](docs/missing-evidence.md).

## Publication

The isolated checkout has the complete branch. The target is `PegahAdt/GraSSPlas`, under `refs/heads/portfolio/plasmid-research`. Fir allowed the public fetch, while direct push transport was unable to resolve `github.com` during the publication check. From a GitHub-authenticated shell with outbound connectivity, use the exact single-ref push in the handoff. Access and redistribution terms for additional lab files are independent of publishing this curated branch.
