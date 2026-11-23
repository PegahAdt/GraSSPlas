# August inputs

The ordered manifest has 52 rows. Its four input columns are relative to a common data root. The historical parser assigns node IDs using the one-based row index, so preserve the ordering.

| Column | Format | Pinned public source |
| --- | --- | --- |
| `gfa_path` | GFA with segments and links | `pseudolabels`, `data/Ecoli_AssemblyGraph_Filtered` |
| `features_path` | TSV with contig ID, ten classifier scores, length, read depth and GC content | `pseudolabels`, `data/Ecoli_Features` |
| `labels_path` | TSV with contig ID and binary chromosome and plasmid calls | `pseudolabels`, `data/Ecoli_Pseudolabels` |
| `ground_truth` | TSV with plasmid assignments, using the contig column for membership | `data`, `Data/Ecoli_GroundTruth` |

All 208 SHA-256 digests match the files selected by the historical manifest on Fir. Use the gold source on `data` to obtain exact bytes. The files of the same name on `pseudolabels` differ. Git can extract the pinned inputs without placing the data branches in the working tree. See [August sample order](august_samples.csv), [input digests](input_hashes.csv), [pseudolabels data tree](https://github.com/PegahAdt/GraSSPlas/tree/0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6/data) and [gold data tree](https://github.com/PegahAdt/GraSSPlas/tree/eb264d9a45ac278fae37a678e4440cf23145801a/Data/Ecoli_GroundTruth).

The underlying data come from the PLASEVAL *E. coli* assemblies. The versioned portfolio maintains references to the public inputs and includes only the manifest and hashes. The PLASEVAL publication cohort, the 52 selected August isolates and the 51 isolates in the shared plASgraph2 study describe different evaluation scopes. [PLASEVAL article](https://doi.org/10.1186/s12859-024-05941-0)

The public generator writes `pseudolabels_060.tsv` from processed MOB marker mappings and graph-derived chromosome heuristics. The README gives the older filename `generate_pseudolabels.py`, while the actual entry point is `code/generate_labels.py`. The BLAST reader accepts `--pid_thr` and `--cov_thr`. Conflicting threshold notes and incomplete command history make the existing hash-verified tables the strongest record of the final inputs. [Generator](https://github.com/PegahAdt/GraSSPlas/blob/0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6/code/generate_labels.py) · [BLAST reader](https://github.com/PegahAdt/GraSSPlas/blob/0c9af7c11ab54eab1a36ed39d9ff3b04b17c40d6/code/read_blast_output.py)
