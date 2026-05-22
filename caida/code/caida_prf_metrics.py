"""
caida_prf_metrics.py
====================
Compute heavy-hitter precision, recall, and F1 score for CAIDA across
memory budgets from 16KB to 256KB.
"""

# python3 caida/code/caida_prf_metrics.py

# for plot diagram
# python3 caida/code/plot_caida_prf_results.py --input caida/results/caida_prf_results.json
# python3 uni/code/plot_uni_prf_results.py --input uni/results/uni_prf_results.json
# python3 plot_saved_results.py --input caida/results/sketch_results.json
# python3 plot_uni_saved_results.py --input uni_sketch_results.json


from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sketch import (  # noqa: E402
    CountMinSketch, CountLessSketch, AegisSketch,
    ElasticSketch, NitroSketch, WavingSketch,
    UnivMonSketch, SketchVisorSketch, StableSketch,
)

_RECORD = 13


def load_trace(path: Path, limit: int | None = None) -> list[bytes]:
    with open(path, "rb") as f:
        raw = f.read()
    n = min(len(raw) // _RECORD, limit) if limit else len(raw) // _RECORD
    return [raw[i * _RECORD: (i + 1) * _RECORD] for i in range(n)]


def precision_recall_f1(true_hh: set[bytes], pred_hh: set[bytes]) -> dict:
    tp = len(true_hh & pred_hh)
    fp = len(pred_hh - true_hh)
    fn = len(true_hh - pred_hh)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_hh_count": len(pred_hh),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CAIDA heavy-hitter precision/recall/F1 experiment.")
    ap.add_argument("--dat", default=str(REPO_ROOT / "dataset/CAIDA/0.dat"),
                    help="Path to CAIDA .dat trace file")
    ap.add_argument("--output", default=str(REPO_ROOT / "caida/results/caida_prf_results.json"),
                    help="Path to save precision/recall/F1 JSON")
    ap.add_argument("--rows", type=int, default=4, help="Sketch depth (rows)")
    ap.add_argument("--sample", type=int, default=0, help="Cap packets (0=all)")
    ap.add_argument("--hh-pct", type=float, default=0.1,
                    help="Heavy-hitter threshold as %% of total traffic")
    ap.add_argument("--nitro-rate", type=int, default=16)
    ap.add_argument("--univmon-levels", type=int, default=8)
    ap.add_argument("--univmon-rows", type=int, default=2)
    ap.add_argument("--sktvisor-fast-ratio", type=float, default=0.25)
    args = ap.parse_args()

    packets = load_trace(Path(args.dat), args.sample or None)
    truth = Counter(packets)
    keys = list(truth.keys())
    total_pkts = len(packets)
    hh_threshold = total_pkts * args.hh_pct / 100.0
    true_hh = {k for k, c in truth.items() if c >= hh_threshold}

    print(f"Packets: {total_pkts:,}   Unique flows: {len(truth):,}")
    print(f"HH threshold: >={hh_threshold:,.0f} packets   True HH flows: {len(true_hh):,}")

    memories = [16, 32, 64, 128, 256]
    results = _run_memory_sweep(args, packets, keys, hh_threshold, true_hh, memories)

    save_data = {
        "dataset": "CAIDA",
        "trace": args.dat,
        "memories": memories,
        "rows": args.rows,
        "hh_pct": args.hh_pct,
        "hh_threshold": hh_threshold,
        "true_hh_count": len(true_hh),
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved precision/recall/F1 results to {output}")


def _run_memory_sweep(args, packets, keys, hh_threshold, true_hh, memories) -> dict:
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
    results = {name: {"memory": [], "precision": [], "recall": [], "f1": [],
                      "tp": [], "fp": [], "fn": [], "predicted_hh_count": []}
               for name, _ in sketches_config}

    for mem_kb in memories:
        print(f"\n--- Running with {mem_kb} KB memory ---")
        widths = _widths_for_memory(mem_kb, args)
        sketches = [(config(widths[name]), name) for name, config in sketches_config]

        for sk, name in sketches:
            t0 = time.perf_counter()
            for p in packets:
                sk.update(p)
            update_seconds = time.perf_counter() - t0

            pred_hh = {k for k in keys if sk.query(k) >= hh_threshold}
            metrics = precision_recall_f1(true_hh, pred_hh)

            results[name]["memory"].append(mem_kb)
            for metric_name, value in metrics.items():
                results[name][metric_name].append(value)

            print(
                f"  {name:<10} P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
                f"pred={metrics['predicted_hh_count']:,} "
                f"update={update_seconds:.2f}s"
            )

    return results


def _widths_for_memory(mem_kb: int, args) -> dict:
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

    return {
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


if __name__ == "__main__":
    main()
