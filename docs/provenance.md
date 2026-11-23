# Contributions, provenance and attribution

This record describes the 2024 SFU research work with Cédric Chauve and Mahsa Feizrahnemoon. Appointment and supervision context follows the researcher’s account. The research artifacts support the development of the custom GraphSAGE workflow, integration of graph and classifier inputs, design and comparison of pseudolabel conditions, threshold evaluation, and investigation of repeat variability. The project used inherited methods and shared materials. Git authors, workspace ownership and launcher paths do not establish sole authorship of individual routines.

| Material | Provenance and treatment |
| --- | --- |
| Inherited GraSSRep | Existing repeat-detection implementation preserved under `legacy/grassrep/`, including its README, environment file, simulation code and original sample placeholders. |
| July prototype and analysis guide | Public `Experiments` tip `2c991711a7301fc7c25d023f55704dba358ffeba`. The July source and launcher are preserved byte-for-byte in `legacy/july-2024/`. |
| August custom implementation and launcher | `/project/6069942/paa40/{mainPlasDetection.py,toolsDetection.py,run.sh}`. These are the latest recovered loose copies linked by the audit to the final array workflow. The exact files are retained, including the historical launcher metadata. |
| Ordered August input definition | Original `/project/6069942/PLASMIDS/DATA/Paths/grassplas_filteredGFA_pseudo_060.txt` under `experiments/august/`. The portable manifest strips the Fir prefix and retains sample order. Public data commits resolve its 208 inputs. |
| Shared filtering and label integration | `/project/6069942/PLASMIDS/DATA/Filter_GFA/` and `/project/6069942/PLASMIDS/DATA/combine_pseudolabels.py`. The local source catalog supplies exact relative paths and hashes for authorized recovery. The portable combine utility is new code. |
| plASgraph2 adaptation | Collaborator development repository `fmfi-compbio/plasgraph2-working`. Personal Fir commit `a6d1b79964e3fdeff6e395df250cb6e8663d4b02` preserves the August stats. Shared commit `31845067` introduces those statistics and `45ee98e6` adds experiment launchers. The shared Fall and randomness branch names point to `3d35f0074e66f3382fbef79a14cc24db479d03e4`. Equal branch tips cannot identify manual intervention edits. |
| Fall experiment files | `/project/6069942/PLASMIDS/tools/plasgraph2-working/exp/grassplas/`. The relative path of a saved YAML is `<family>/Seed<folder_seed>_Run<run>/model-<method>_<folder_seed>_<run>/plasgraph_config.yaml`. The compressed fit ledger preserves each YAML hash and recorded settings. |
| Historical findings | Fir reconstruction evidence under `/home/paa40/plasmid_project_reconstruction/Report_Source/evidence/` and the corrected report. The public portfolio retains aggregate results and curated figures. |
| Earlier bioinformatic analyses | `Experiments_and_Data_Analysis`, `data` and `pseudolabels` already carry the public analyses. Later adjacent projects such as PlasBin-flow seeding and LongClass are documented separately in the Fir reconstruction. |

See the [file-level source mapping and SHA-256 digests](provenance.csv) and [checksum catalog for locally recoverable files](local_sources.csv).

## References

- GraSSRep provides the original graph-based repeat-detection code and the fork history. Its original README is preserved under [legacy/grassrep](../legacy/grassrep/README.md). [GraSSRep source](https://github.com/aliaaz99/GraSSRep)
- Hamilton, Ying and Leskovec introduced GraphSAGE. The custom classifier uses the PyG implementation of the GraphSAGE convolution. [GraphSAGE paper](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
- Sielemann, Sielemann, Brejová, Vinař and Chauve developed plASgraph2. The Fall work reuses that model family through a lab development checkout. [Paper](https://doi.org/10.3389/fmicb.2023.1267695) · [Public source](https://github.com/cchauve/plASgraph2)
- Mane and colleagues describe PLASEVAL, the source of the assembly cohort context and baseline tool comparison. [PLASEVAL paper](https://doi.org/10.1186/s12859-024-05941-0)

## Redistribution

No general redistribution license was identified in the inspected GraSSPlas base tree, GraSSRep project listing, or Fir adaptation copies. The historical files already tracked by the chosen GraSSPlas base retain their original contents and attribution within the same repository history. This branch adds no blanket license for that code or the data it references. For reuse outside the repository, follow the respective upstream terms and seek a grant where terms are absent.

Shared lab source, raw datasets, training outputs and third-party binaries require their own redistribution basis. This branch ships portable interfaces, source references and aggregate metrics for the shared work. Other nonpublic artifacts are available through authorized copies. The exact Slurm scripts include previously public academic contact metadata and Fir-specific paths. The portable wrappers accept caller-supplied paths and contain neither.
