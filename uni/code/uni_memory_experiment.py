"""
uni_memory_experiment.py
========================
Run all sketches on the UNI trace with varying memory budgets (16KB to 256KB)
and collect metrics: throughput, ARE top-K, ARE all flows, ARE on HH set.
Then save the results to JSON and plot each metric.
"""

from __future__ import annotations

import argparse
import csv
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


def load_trace(path: str, limit: int | None = None) -> list[bytes]:
    packets: list[bytes] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_ip = row.get("ip.src", "").strip()
            dst_ip = row.get("ip.dst", "").strip()
            src_port = row.get("tcp.srcport", "0").strip() or "0"
            dst_port = row.get("tcp.dstport", "0").strip() or "0"
            proto = row.get("ip.proto", "0").strip() or "0"

            if not src_ip or not dst_ip:
                continue

            packets.append(f"{src_ip}:{src_port}->{dst_ip}:{dst_port}/{proto}".encode())
            if limit and len(packets) >= limit:
                break

    return packets


def are(truth: dict, estimates: dict) -> float:
    errs = [abs(estimates.get(k, 0) - v) / v for k, v in truth.items() if v > 0]
    return sum(errs) / len(errs) if errs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Sketch comparison memory experiment on UNI trace.")
    ap.add_argument("--csv", default="dataset/UNI/univ1_pt0.csv",
                    help="Path to UNI CSV trace file")
    ap.add_argument("--output", default="uni_sketch_results.json",
                    help="Path to save JSON results")
    ap.add_argument("--plot-prefix", default="uni_",
                    help="Prefix for saved plot filenames")
    ap.add_argument("--topk", type=int, default=20, help="Heavy-hitter top-K")
    ap.add_argument("--rows", type=int, default=4, help="Sketch depth (rows)")
    ap.add_argument("--sample", type=int, default=0, help="Cap packets (0=all)")
    ap.add_argument("--hh-pct", type=float, default=0.1,
                    help="Heavy-hitter threshold as %% of total traffic (default 0.1)")
    ap.add_argument("--nitro-rate", type=int, default=16,
                    help="NitroSketch sampling rate (higher = faster, less accurate)")
    ap.add_argument("--univmon-levels", type=int, default=8)
    ap.add_argument("--univmon-rows", type=int, default=2)
    ap.add_argument("--sktvisor-fast-ratio", type=float, default=0.25)

    args = ap.parse_args()
    _run(args)


def _run(args) -> None:
    print(f"Loading {args.csv}", flush=True)
    packets = load_trace(args.csv, args.sample or None)
    truth = Counter(packets)
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

    memories = [16, 32, 64, 128, 256]

    sketches_config = [
        ("CMS", lambda W: CountMinSketch(num_rows=args.rows, num_buckets=W)),
        ("CountLess", lambda W: CountLessSketch(num_rows=args.rows, row_width=W)),
        ("Aegis", lambda W: AegisSketch(_capacity_cells=args.rows * W)),
        ("Elastic", lambda W: ElasticSketch(num_rows=args.rows, row_width=W)),
        ("Nitro", lambda W: NitroSketch(num_rows=args.rows, row_width=W, sampling_rate=args.nitro_rate)),
        ("Waving", lambda W: WavingSketch(num_rows=args.rows, row_width=W)),
        ("UnivMon", lambda W: UnivMonSketch(num_levels=args.univmon_levels,
                                            num_rows=args.univmon_rows,
                                            row_width=W)),
        ("SktVisor", lambda W: SketchVisorSketch(num_rows=args.rows,
                                                 row_width=W,
                                                 fast_ratio=args.sktvisor_fast_ratio)),
        ("Stable", lambda W: StableSketch(num_rows=args.rows, row_width=W)),
    ]

    results = {
        name: {"memory": [], "throughput": [], "are_top20": [], "are_all": [], "are_hh": []}
        for name, _ in sketches_config
    }

    for mem_kb in memories:
        print(f"\n--- Running with {mem_kb} KB memory ---")
        target = mem_kb * 1024
        R = args.rows
        UR = args.univmon_rows
        UL = args.univmon_levels
        FR = args.sktvisor_fast_ratio

        W_base = max(1, target // (R * 4))
        W_countless = max(1, target // (R * 4))
        W_elastic = max(1, target // ((1 + R) * 4))
        W_waving = max(1, target // (R * 8))
        W_univmon = max(1, target // (UL * UR * 4))
        W_sktvisor = max(1, int(target / (4 * (FR + R * (1 - FR)))))
        W_stable = max(1, target // (R * 12))

        widths = {
            "CMS": W_base,
            "CountLess": W_countless,
            "Aegis": W_base,
            "Elastic": W_elastic,
            "Nitro": W_base,
            "Waving": W_waving,
            "UnivMon": W_univmon,
            "SktVisor": W_sktvisor,
            "Stable": W_stable,
        }

        sketches = [(config(widths[name]), name) for name, config in sketches_config]

        print(f"  {'Sketch':<12} {'Memory':>12}")
        print(f"  {'-' * 26}")
        for sk, name in sketches:
            print(f"  {name:<12} {sk.memory_bytes:>10} B")

        print("\n--- Throughput ---")
        for sk, name in sketches:
            t0 = time.perf_counter()
            for p in packets:
                sk.update(p)
            tput = len(packets) / (time.perf_counter() - t0) / 1e6
            print(f"  {name:<10}: {tput:.2f} Mops/s")
            results[name]["throughput"].append(tput)

        top_keys = [k for k, _ in truth.most_common(args.topk)]
        gt_top = {k: truth[k] for k in top_keys}

        print(f"\n--- ARE top-{args.topk} ---")
        for sk, name in sketches:
            est = {k: sk.query(k) for k in top_keys}
            are_val = are(gt_top, est)
            print(f"  {name:<10}: {are_val:.4f}")
            results[name]["are_top20"].append(are_val)

        all_k = list(truth.keys())
        gt_all = dict(truth)

        print(f"\n--- ARE all {len(all_k):,} flows ---")
        for sk, name in sketches:
            est = {k: sk.query(k) for k in all_k}
            are_val = are(gt_all, est)
            print(f"  {name:<10}: {are_val:.4f}")
            results[name]["are_all"].append(are_val)

        print(
            f"\n--- ARE on HH set "
            f"(>={args.hh_pct}% traffic, {len(hh_keys):,} flows) ---"
        )
        for sk, name in sketches:
            est = {k: sk.query(k) for k in hh_keys}
            are_val = are(gt_hh, est)
            print(f"  {name:<10}: {are_val:.4f}")
            results[name]["are_hh"].append(are_val)

        for name in results:
            results[name]["memory"].append(mem_kb)

    save_data = {
        "dataset": "UNI",
        "csv": args.csv,
        "memories": memories,
        "topk": args.topk,
        "rows": args.rows,
        "hh_pct": args.hh_pct,
        "hh_threshold": hh_threshold,
        "hh_flow_count": len(hh_keys),
        "nitro_rate": args.nitro_rate,
        "univmon_levels": args.univmon_levels,
        "univmon_rows": args.univmon_rows,
        "sktvisor_fast_ratio": args.sktvisor_fast_ratio,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved experiment data to {args.output}")

    _plot_results(memories, results, args.plot_prefix)


def _plot_results(memories: list[int], results: dict, plot_prefix: str) -> None:
    metrics = ["throughput", "are_top20", "are_all", "are_hh"]
    ylabels = ["Throughput", "ARE", "ARE", "ARE"]
    filenames = [
        f"{plot_prefix}throughput.pdf",
        f"{plot_prefix}are_top20.pdf",
        f"{plot_prefix}are_all.pdf",
        f"{plot_prefix}are_hh.pdf",
    ]
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
        ax.set_xlabel("Memory (KB)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(x) for x in memories])
        ax.set_xlim(-0.25, len(memories) - 0.75)
        _adjust_y_axis(ax, results, metric)
        ax.grid(True)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(filename)
        print(f"Saved plot: {filename}")

    plt.show()
    print("\nAll cycles completed. All plots displayed and saved.")


if __name__ == "__main__":
    main()
