# Recovered results

| File | Contents |
| --- | --- |
| [august_metrics.csv](august_metrics.csv) | 120 rows of exact metric values and confusion counts covering five runs, three thresholds and eight call sets |
| [august_plasgraph2.csv](august_plasgraph2.csv) | Pooled aggregate rows recovered from the earlier committed six-condition sweep, including pseudolabel evaluation files |
| [fall_metrics.csv](fall_metrics.csv) | 840 pooled rows, covering two lengths and two molecular labels for 210 saved fits |
| [fall_summary.csv](fall_summary.csv) | Means, sample standard deviations and ranges by Fall experiment family, length, condition and molecular label |
| [fall_baselines.csv](fall_baselines.csv) | Derived baseline scores on the shared cohort |
| [figures/](figures/) | Verified August baseline and Fall intervention plots from the Fir reconstruction |

The original values are preserved at saved precision. The repository’s validation tool checks August metric arithmetic and recomputes Fall summaries from the per-fit aggregate rows. The Fir reconstruction also rescored the stored Fall predictions against gold. The repository data omits individual contig records, checkpoints and stdout.
