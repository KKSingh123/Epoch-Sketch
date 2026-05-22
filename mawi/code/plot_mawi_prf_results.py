"""
plot_mawi_prf_results.py
========================
Load saved MAWI precision/recall/F1 data and plot the metrics without
rerunning the sketch experiment.
"""

# python3 mawi/code/plot_mawi_prf_results.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS_TO_PLOT = ["CountLess", "Elastic", "Stable", "Aegis"]
METHOD_COLORS = {
    "CountLess": "tab:blue",
    "Elastic": "tab:orange",
    "Stable": "tab:red",
    "Aegis": "tab:green",
}
METHOD_MARKERS = {
    "CountLess": "o",
    "Elastic": "s",
    "Stable": "^",
    "Aegis": "D",
}
METHOD_LINESTYLES = {
    "CountLess": "-",
    "Elastic": "--",
    "Stable": "-.",
    "Aegis": ":",
}


def _selected_values(results: dict, metric: str) -> list[float]:
    return [
        value
        for name in METHODS_TO_PLOT
        if name in results
        for value in results[name][metric]
    ]


def _adjust_y_axis(ax, results: dict, metric: str) -> None:
    values = _selected_values(results, metric)
    if not values:
        return
    ymin = min(values)
    ymax = max(values)
    margin = max((ymax - ymin) * 0.15, 0.01)
    ax.set_ylim(max(0, ymin - margin), min(1.02, ymax + margin))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot saved MAWI precision/recall/F1 results.")
    ap.add_argument("--input", default=str(REPO_ROOT / "mawi/results/mawi_prf_results.json"),
                    help="Path to saved MAWI precision/recall/F1 JSON file")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "mawi/results"),
                    help="Directory where PDF plots will be saved")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    memories = data["memories"]
    results = data["results"]
    x_positions = list(range(len(memories)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "f1"]
    ylabels = ["Precision", "Recall", "F1 Score"]
    filenames = [
        output_dir / "mawi_precision.pdf",
        output_dir / "mawi_recall.pdf",
        output_dir / "mawi_f1_score.pdf",
    ]

    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 16,
    })

    display_names = {
        "Aegis": "HFH Sketch",
        "CMS": "CMS",
        "CountLess": "CountLess",
        "Elastic": "Elastic",
        "Nitro": "Nitro",
        "Waving": "Waving",
        "UnivMon": "UnivMon",
        "SktVisor": "SktVisor",
        "Stable": "Stable",
    }
    for metric, ylabel, filename in zip(metrics, ylabels, filenames):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in METHODS_TO_PLOT:
            if name not in results:
                print(f"Warning: {name} not found in results; skipping.")
                continue
            values = results[name]
            label = display_names.get(name, name)
            ax.plot(x_positions, values[metric],
                    marker=METHOD_MARKERS.get(name, "o"),
                    linestyle=METHOD_LINESTYLES.get(name, "-"),
                    linewidth=2.5,
                    markersize=7,
                    color=METHOD_COLORS.get(name),
                    label=label)
        ax.set_xlabel("Memory (KB)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(x) for x in memories])
        ax.set_xlim(-0.25, len(memories) - 0.75)
        _adjust_y_axis(ax, results, metric)
        ax.grid(True)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(filename)
        print(f"Saved plot: {filename}")

    plt.show()


if __name__ == "__main__":
    main()
