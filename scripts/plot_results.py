"""Regenerate the GraphSAGE comparison and the matched-seed robustness figure."""
import json
from statistics import stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results import BASELINES, EXPERIMENTS, experiment_rows, f1_groups

COLORS = ("#344f75", "#337f76", "#b65b3f")
BASELINE_NAMES = ("PlasForest", "plASgraph2", "Platon", "RFPlasmid", "mlplasmids")


def graph_comparison():
    folder = EXPERIMENTS / "graphsage"
    records = experiment_rows("graphsage")
    tools = ("GrassPlas", *BASELINES)
    labels = ("GraphSAGE", *BASELINE_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for axis, minimum in zip(axes, ("100", "1000")):
        for i, tool in enumerate(tools):
            values = [float(r["f1"]) for r in records
                      if r["threshold"] == minimum and r["tool"] == tool]
            color = COLORS[0] if i == 0 else COLORS[1]
            axis.hlines(i, min(values), max(values), color=color, linewidth=2)
            axis.scatter(sorted(set(values)), [i] * len(set(values)), color=color, s=26, zorder=2)
        axis.set(yticks=range(len(labels)), yticklabels=labels, xlim=(0.25, 0.9),
                 xlabel="Plasmid F1", title=f"Contigs ≥{int(minimum):,} bp")
        axis.grid(axis="x", color="#e6e9ed")
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=180, metadata={"Software": "GraSSPlas"})
    plt.close(fig)


def robustness():
    folder = EXPERIMENTS / "robustness"
    records = [r for r in experiment_rows("weak-supervision") + experiment_rows("robustness")
               if r["nominal_seed"] == r["configured_seed"] == "123"]
    groups = f1_groups(records, ("protocol", "condition", "minimum_bp"))
    with (EXPERIMENTS / "weak-supervision/conditions.json").open() as stream:
        conditions = json.load(stream)
    protocols = {
        "repeated_fits": "Reference",
        "fixed_validation": "Fixed validation",
        "fixed_validation_no_dropout": "Fixed validation + no dropout",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, minimum in zip(axes, ("100", "1000")):
        for i, (protocol, label) in enumerate(protocols.items()):
            positions = [x + (i - 1) * 0.25 for x in range(len(conditions))]
            values = [stdev(groups[protocol, condition, minimum]) for condition in conditions]
            axis.bar(positions, values, width=0.23, color=COLORS[i], label=label)
        axis.set(xticks=range(len(conditions)), title=f"Contigs ≥{int(minimum):,} bp", ylim=(0, 0.16))
        axis.set_xticklabels(("MOB", "MOB2", "Pfam", "Mask", "Random", "True"),
                             rotation=30, ha="right")
        axis.grid(axis="y", color="#e6e9ed")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Sample SD of plasmid F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(folder / "variability.png", dpi=180, metadata={"Software": "GraSSPlas"})
    plt.close(fig)


if __name__ == "__main__":
    graph_comparison()
    robustness()
