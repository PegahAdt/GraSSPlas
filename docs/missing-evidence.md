# Limits of the historical record

The reconstructed input tables, aggregate results and synthetic driver behavior are verified. Exact retraining of the historic models remains incomplete.

| Missing or incomplete evidence | Surviving evidence and consequence |
| --- | --- |
| Fall base commit `127692edfb821a1d6eff9aa081cf594b48e12ff3` | All three Fall experiment notes cite it. Neither recovered checkout, its inspected archive nor the public GraSSPlas history yielded the object. The collaborator upstream required credentials during the audit. A collaborator’s older authorized clone or pack is the best lead. |
| Exact intervention-specific sources | Fixed-validation and no-dropout notes describe manual edits. Current shared architecture and training files contain uncommitted changes whose relationship to every fit is unverified. Configuration fields and saved loss traces survive. |
| Early model1–model3 artifacts | The old README contains aggregate tables and names three per-contig predictions, model directories and evaluation outputs. The YAMLs and README remain locally recoverable. The predictions, model binaries and exact result CSVs for those fits remain unlocated. |
| Earlier August plASgraph2 fits | Seven historical statistics files survive in local Git, while the exact six fitted models, predictions and random input selections are unlocated. |
| Earlier custom project revisions | An older log reached epoch 3000, while both recovered drivers set 2,001 epochs. Older outputs also used a different feature set and cohort. A May helper survives as bytecode with a 7,219-byte source size recorded in the header. The original source text and run-to-version mapping remain incomplete. |
| Forty-five intermediate custom predictions | Five array jobs each ran ten fits. Each writes only its last prediction vector. Intermediate predictions appear to have been transient. |
| Custom trained GraphSAGE weights | No weight export was identified. The historical `.pt` files contain inputs, edges, masks or labels. They cannot restore a trained model. |
| Complete data and execution provenance | The 208 August inputs are byte-verified. Original baseline database versions, all gold derivation choices, the complete execution command record and source-to-run bindings remain incomplete. |
| 2024 research correspondence | No original research email or attachment was available in the audited material. Claims here use the recovered research record and the researcher’s description of her role. |

An isolate-disjoint evaluation, corrected-loss comparison, symmetric-edge experiment and a matched RF-only comparison would answer new research questions. Recovered evidence does not establish that those experiments were completed. The later related work and the inherited plASgraph2 paper benchmarks are outside the 2024 evaluation corpus.
