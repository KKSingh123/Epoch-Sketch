"""
mawi_memory_experiment.py
=========================
Run all sketches on MAWI trace with varying memory budgets (16KB to 256KB)
and collect metrics: throughput, ARE top-20, ARE all flows, ARE on HH set.
Then plot the results for each sketch.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
import matplotlib.pyplot as plt

from sketch import (
    CountMinSketch, CountLessSketch, AegisSketch,
    ElasticSketch, NitroSketch, WavingSketch,
    UnivMonSketch, SketchVisorSketch, StableSketch,
)

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
    if metric.startswith("are_"):
        positive_values = [value for value in values if value > 0]
        linthresh = max(min(positive_values) / 2, 1e-6) if positive_values else 1e-6
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(bottom=0)
        return
    ymin = min(values)
    ymax = max(values)
    margin = max((ymax - ymin) * 0.15, 0.02)
    ax.set_ylim(max(0, ymin - margin), ymax + margin)

_RECORD = 13  # bytes per packet record


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_trace(path: str, limit: int | None = None) -> list[bytes]:
    with open(path, "rb") as f:
        raw = f.read()
    n = min(len(raw) // _RECORD, limit) if limit else len(raw) // _RECORD
    return [raw[i * _RECORD : (i + 1) * _RECORD] for i in range(n)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def are(truth: dict, estimates: dict) -> float:
    errs = [abs(estimates.get(k, 0) - v) / v for k, v in truth.items() if v > 0]
    return sum(errs) / len(errs) if errs else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Sketch comparison memory experiment on MAWI trace.")
    ap.add_argument("--dat",     default="dataset/Mawi/mawi.dat",  help="Path to .dat trace file")
    ap.add_argument("--topk",    type=int, default=20,   help="Heavy-hitter top-K")
    ap.add_argument("--rows",    type=int, default=4,  help="Sketch depth (rows)")
    ap.add_argument("--sample",  type=int, default=0,  help="Cap packets (0=all)")
    ap.add_argument("--hh-pct",  type=float, default=0.1,
                                 help="Heavy-hitter threshold as %% of total traffic (default 0.1)")

    # NitroSketch
    ap.add_argument("--nitro-rate", type=int, default=16,
                    help="NitroSketch sampling rate (higher = faster, less accurate)")

    # UnivMon
    ap.add_argument("--univmon-levels", type=int, default=8)
    ap.add_argument("--univmon-rows",   type=int, default=2)

    # SketchVisor
    ap.add_argument("--sktvisor-fast-ratio", type=float, default=0.25)

    args = ap.parse_args()
    _run(args)


def _run(args) -> None:
    # ------------------------------------------------------------------
    # Load trace
    # ------------------------------------------------------------------
    packets = load_trace(args.dat, args.sample or None)
    truth   = Counter(packets)
    print(f"Packets: {len(packets):,}   Unique flows: {len(truth):,}")

    total_pkts = len(packets)
    hh_threshold = total_pkts * args.hh_pct / 100.0
    hh_keys = [
        k for k, c in sorted(truth.items(), key=lambda x: x[1], reverse=True)
        if c >= hh_threshold
    ]
    gt_hh = {k: truth[k] for k in hh_keys}
    print(
        f"HH threshold: >={args.hh_pct}% of traffic "
        f"({hh_threshold:,.0f} packets)   HH flows: {len(hh_keys):,}"
    )

    # Memory sizes in KB
    memories = [16, 32, 64, 128, 256]

    # Sketch configs
    sketches_config = [
        ("CMS", lambda W: CountMinSketch(num_rows=args.rows, num_buckets=W)),
        ("CountLess", lambda W: CountLessSketch(num_rows=args.rows, row_width=W)),
        ("Aegis", lambda W: AegisSketch(_capacity_cells=args.rows * W)),
        ("Elastic", lambda W: ElasticSketch(num_rows=args.rows, row_width=W)),
        ("Nitro", lambda W: NitroSketch(num_rows=args.rows, row_width=W, sampling_rate=args.nitro_rate)),
        ("Waving", lambda W: WavingSketch(num_rows=args.rows, row_width=W)),
        ("UnivMon", lambda W: UnivMonSketch(num_levels=args.univmon_levels, num_rows=args.univmon_rows, row_width=W)),
        ("SktVisor", lambda W: SketchVisorSketch(num_rows=args.rows, row_width=W, fast_ratio=args.sktvisor_fast_ratio)),
        ("Stable", lambda W: StableSketch(num_rows=args.rows, row_width=W)),
    ]

    results = {name: {'memory': [], 'throughput': [], 'are_top20': [], 'are_all': [], 'are_hh': []} for name, _ in sketches_config}

    for mem_kb in memories:
        print(f"\n--- Running with {mem_kb} KB memory ---")
        target = mem_kb * 1024   # bytes
        R  = args.rows
        UR = args.univmon_rows
        UL = args.univmon_levels
        FR = args.sktvisor_fast_ratio

        # row_width formulas
        W_base     = max(1, target // (R * 4))
        W_countless = max(1, target // (R * 4))
        W_elastic  = max(1, target // ((1 + R) * 4))
        W_waving   = max(1, target // (R * 8))
        W_univmon  = max(1, target // (UL * UR * 4))
        W_sktvisor = max(1, int(target / (4 * (FR + R * (1 - FR)))))
        W_stable   = max(1, target // (R * 12))

        widths = {
            "CMS": W_base,
            "CountLess": W_countless,
            "Aegis": W_base,  # _capacity_cells = R * W_base
            "Elastic": W_elastic,
            "Nitro": W_base,
            "Waving": W_waving,
            "UnivMon": W_univmon,
            "SktVisor": W_sktvisor,
            "Stable": W_stable,
        }

        sketches = [(config(widths[name]), name) for name, config in sketches_config]

        print(f"  {'Sketch':<12} {'Memory':>12}")
        print(f"  {'-'*26}")
        for sk, name in sketches:
            print(f"  {name:<12} {sk.memory_bytes:>10} B")

        # ------------------------------------------------------------------
        # Throughput
        # ------------------------------------------------------------------
        print("\n--- Throughput ---")
        for sk, name in sketches:
            t0 = time.perf_counter()
            for p in packets:
                sk.update(p)
            tput = len(packets) / (time.perf_counter() - t0) / 1e6
            print(f"  {name:<10}: {tput:.2f} Mops/s")
            results[name]['throughput'].append(tput)

        # ------------------------------------------------------------------
        # Accuracy — top-K heavy hitters
        # ------------------------------------------------------------------
        top_keys = [k for k, _ in truth.most_common(args.topk)]
        gt_top = {k: truth[k] for k in top_keys}

        print(f"\n--- ARE top-{args.topk} ---")
        for sk, name in sketches:
            est = {k: sk.query(k) for k in top_keys}
            are_val = are(gt_top, est)
            print(f"  {name:<10}: {are_val:.4f}")
            results[name]['are_top20'].append(are_val)

        # ------------------------------------------------------------------
        # Accuracy — all flows
        # ------------------------------------------------------------------
        all_k = list(truth.keys())
        gt_all = dict(truth)

        print(f"\n--- ARE all {len(all_k):,} flows ---")
        for sk, name in sketches:
            est = {k: sk.query(k) for k in all_k}
            are_val = are(gt_all, est)
            print(f"  {name:<10}: {are_val:.4f}")
            results[name]['are_all'].append(are_val)

        # ------------------------------------------------------------------
        # Heavy Hitters: flows with true count >= hh_pct% of total traffic
        # ------------------------------------------------------------------
        print(
            f"\n--- ARE on HH set "
            f"(>={args.hh_pct}% traffic, {len(hh_keys):,} flows) ---"
        )
        for sk, name in sketches:
            est = {k: sk.query(k) for k in hh_keys}
            are_val = are(gt_hh, est)
            print(f"  {name:<10}: {are_val:.4f}")
            results[name]['are_hh'].append(are_val)

        # Store memory
        for name in results:
            results[name]['memory'].append(mem_kb)

    # ------------------------------------------------------------------
    # Save results for separate plotting
    # ------------------------------------------------------------------
    output_file = 'sketch_results.json'
    save_data = {
        'dataset': 'MAWI',
        'dat': args.dat,
        'memories': memories,
        'topk': args.topk,
        'rows': args.rows,
        'hh_pct': args.hh_pct,
        'hh_threshold': hh_threshold,
        'hh_flow_count': len(hh_keys),
        'nitro_rate': args.nitro_rate,
        'univmon_levels': args.univmon_levels,
        'univmon_rows': args.univmon_rows,
        'sktvisor_fast_ratio': args.sktvisor_fast_ratio,
        'results': results,
    }
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved experiment data to {output_file}")

    # ------------------------------------------------------------------
    # Plot results separately
    # ------------------------------------------------------------------
    metrics = ['throughput', 'are_top20', 'are_all', 'are_hh']
    ylabels = ['Throughput', 'ARE', 'ARE', 'ARE']
    filenames = ['mawi_throughput.png', 'mawi_are_top20.png', 'mawi_are_all.png', 'mawi_are_hh.png']
    x_positions = list(range(len(memories)))

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
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in METHODS_TO_PLOT:
            if name not in results:
                print(f"Warning: {name} not found in results; skipping.")
                continue
            label = display_names.get(name, name)
            ax.plot(x_positions, results[name][metric],
                    marker=METHOD_MARKERS.get(name, "o"),
                    linestyle=METHOD_LINESTYLES.get(name, "-"),
                    linewidth=2.5,
                    markersize=7,
                    color=METHOD_COLORS.get(name),
                    label=label)
        ax.set_xlabel('Memory (KB)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(x) for x in memories])
        ax.set_xlim(-0.25, len(memories) - 0.75)
        _adjust_y_axis(ax, results, metric)
        ax.grid(True)
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(filename)
        print(f"Saved plot: {filename}")

    plt.show()

    print("\nAll cycles completed. All plots displayed and saved.")


if __name__ == "__main__":
    main()
