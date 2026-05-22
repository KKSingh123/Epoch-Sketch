"""
mawi26_demo.py
==============
Compare sketches on a MAWI26 trace:

  CMS        CountMin Sketch              (Cormode & Muthukrishnan, 2005)
  Aegis      Aegis Array                  (this work)
  Elastic    ElasticSketch                (Yang et al., SIGCOMM 2018)
  Nitro      NitroSketch                  (Liu et al., SIGCOMM 2019)
  Waving     WavingSketch                 (Li et al., KDD 2020)
  UnivMon    Universal Monitoring         (Liu et al., SIGCOMM 2016)
  SktVisor   SketchVisor                  (Huang et al., SIGCOMM 2017)
  Stable     StableSketch                 (Li & Patras, WWW 2024)

Trace format: 13-byte records per packet
    bytes 0-3   srcIP   bytes 4-5  srcPort
    bytes 6-9   dstIP   bytes 10-11 dstPort
    byte  12    protocol

Usage
-----
    python mawi/code/mawi26_demo.py --dat dataset/Mawi/mawi26.dat --sample 100000
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from collections import Counter

from sketch import (
    CountMinSketch, CountLessSketch, AegisSketch,
    ElasticSketch, NitroSketch, WavingSketch,
    UnivMonSketch, SketchVisorSketch, StableSketch,
)

_RECORD = 13  # bytes per packet record


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_trace(path: str, limit: int | None = None) -> list[bytes]:
    with open(path, "rb") as f:
        raw = f.read()
    n = min(len(raw) // _RECORD, limit) if limit else len(raw) // _RECORD
    return [raw[i * _RECORD : (i + 1) * _RECORD] for i in range(n)]


def flow_str(b: bytes) -> str:
    src = f"{socket.inet_ntoa(b[0:4])}:{struct.unpack('>H', b[4:6])[0]}"
    dst = f"{socket.inet_ntoa(b[6:10])}:{struct.unpack('>H', b[10:12])[0]}"
    return f"{src}->{dst}/{b[12]}"


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
    ap = argparse.ArgumentParser(description="Sketch comparison on MAWI26 trace.")
    ap.add_argument("--dat",     default="dataset/Mawi/mawi26.dat",  help="Path to .dat trace file")
    ap.add_argument("--topk",    type=int, default=20,   help="Heavy-hitter top-K")
    ap.add_argument("--rows",    type=int, default=4,  help="Sketch depth (rows)")
    ap.add_argument("--memory",  type=int, default=64, help="Memory budget per sketch in KB (all sketches use the same budget)")
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

    # ------------------------------------------------------------------
    # Compute per-sketch row widths so every sketch uses the same memory
    # ------------------------------------------------------------------
    target = args.memory * 1024   # bytes
    R  = args.rows
    UR = args.univmon_rows
    UL = args.univmon_levels
    FR = args.sktvisor_fast_ratio

    # row_width formulas derived from each sketch's memory_bytes property:
    #   CMS/Nitro       : R * W * 4
    #   CountLess       : conservative-update CMS, 4 bytes/cell
    #   Aegis           : exact slots (fingerprint/count/shield) + slow-path CMS
    #   Elastic         : (1 + R) * W * 4   (heavy array + light CMS)
    #   Waving          : R * W * 8         (int32 counts + uint32 fps)
    #   UnivMon         : UL * UR * W * 4
    #   SketchVisor     : W * 4 * (FR + R*(1-FR))
    W_base     = max(1, target // (R * 4))
    W_countless = max(1, target // (R * 4))
    W_elastic  = max(1, target // ((1 + R) * 4))
    W_waving   = max(1, target // (R * 8))
    W_univmon  = max(1, target // (UL * UR * 4))
    W_sktvisor = max(1, int(target / (4 * (FR + R * (1 - FR)))))
    W_stable   = max(1, target // (R * 12))   # 12 B per cell: fp + count + stability

    # ------------------------------------------------------------------
    # Build sketches
    # ------------------------------------------------------------------
    cms = CountMinSketch(num_rows=R, num_buckets=W_base)
    countless = CountLessSketch(num_rows=R, row_width=W_countless)
    aegis = AegisSketch(_capacity_cells=R * W_base)

    elastic  = ElasticSketch(num_rows=R, row_width=W_elastic)
    nitro    = NitroSketch(num_rows=R, row_width=W_base,
                           sampling_rate=args.nitro_rate)
    waving   = WavingSketch(num_rows=R, row_width=W_waving)
    univmon  = UnivMonSketch(num_levels=UL, num_rows=UR, row_width=W_univmon)
    sktvisor = SketchVisorSketch(num_rows=R, row_width=W_sktvisor, fast_ratio=FR)
    stable   = StableSketch(num_rows=R, row_width=W_stable)

    try:
        import mmh3  # noqa: F401
        hash_backend = "mmh3"
    except ImportError:
        hash_backend = "sha256-fallback"

    # ------------------------------------------------------------------
    # Run Config
    # ------------------------------------------------------------------
    print(
        f"\nRun Config:  data={args.dat}  sample={'all' if not args.sample else args.sample}"
        f"  topk={args.topk}  hash={hash_backend}  budget={args.memory} KB\n"
        f"  {'Sketch':<12} {'Memory':>12}\n"
        f"  {'-'*26}"
    )
    for sk, nm in [(cms,"CMS"),(countless,"CountLess"),(aegis,"Aegis"),(elastic,"Elastic"),
                   (nitro,"Nitro"),(waving,"Waving"),(univmon,"UnivMon"),(sktvisor,"SktVisor"),
                   (stable,"Stable")]:
        print(f"  {nm:<12} {sk.memory_bytes:>10,} B")

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------
    print("\n--- Throughput ---")
    sketches = [
        (cms,      "CMS"),
        (countless,"CountLess"),
        (aegis,    "Aegis"),
        (elastic,  "Elastic"),
        (nitro,    "Nitro"),
        (waving,   "Waving"),
        (univmon,  "UnivMon"),
        (sktvisor, "SktVisor"),
        (stable,   "Stable"),
    ]
    for sk, name in sketches:
        t0   = time.perf_counter()
        for p in packets:
            sk.update(p)
        tput = len(packets) / (time.perf_counter() - t0) / 1e6
        print(f"  {name:<10}: {tput:.2f} Mops/s")

    # ------------------------------------------------------------------
    # Accuracy — top-K heavy hitters
    # ------------------------------------------------------------------
    top_keys = [k for k, _ in truth.most_common(args.topk)]
    gt_top   = {k: truth[k] for k in top_keys}

    print(f"\n--- ARE top-{args.topk} ---")
    for sk, name in sketches:
        est = {k: sk.query(k) for k in top_keys}
        print(f"  {name:<10}: {are(gt_top, est):.4f}")

    # ------------------------------------------------------------------
    # Accuracy — all flows
    # ------------------------------------------------------------------
    all_k  = list(truth.keys())
    gt_all = dict(truth)

    print(f"\n--- ARE all {len(all_k):,} flows ---")
    for sk, name in sketches:
        est = {k: sk.query(k) for k in all_k}
        print(f"  {name:<10}: {are(gt_all, est):.4f}")

    # ------------------------------------------------------------------
    # Per-flow table for top-K  (abbreviated column headers)
    # ------------------------------------------------------------------
    FL = 44   # flow label width
    CW = 7    # per-sketch column width
    hdrs = ["True", "CMS", "CLss", "Aegs", "Elst", "Ntro", "Wavg", "UMon", "SVsr", "Stbl"]
    header = f"{'Flow':<{FL}}" + "".join(f"{h:>{CW}}" for h in hdrs)
    print(f"\n{header}")
    print("-" * len(header))
    for k in top_keys:
        vals = [truth[k]] + [sk.query(k) for sk, _ in sketches]
        label = flow_str(k)[:FL]
        row = f"{label:<{FL}}" + "".join(f"{v:>{CW},}" for v in vals)
        print(row)

    # ------------------------------------------------------------------
    # Heavy Hitters: flows with true count > hh_pct% of total traffic
    # ------------------------------------------------------------------
    total_pkts = len(packets)
    hh_threshold = total_pkts * args.hh_pct / 100.0
    hh_flows = sorted(
        [(k, c) for k, c in truth.items() if c >= hh_threshold],
        key=lambda x: x[1], reverse=True
    )
    print(
        f"\n--- Heavy Hitters (>={args.hh_pct}% of {total_pkts:,} pkts  "
        f"threshold={hh_threshold:,.0f})  count={len(hh_flows)} ---"
    )
    PCT = 6   # width of traffic% column
    hdrs_hh = ["True", "%traf", "CMS", "CLss", "Aegs", "Elst", "Ntro", "Wavg", "UMon", "SVsr", "Stbl"]
    hh_header = f"{'Flow':<{FL}}" + "".join(f"{h:>{CW}}" for h in hdrs_hh)
    print(hh_header)
    print("-" * len(hh_header))
    for k, c in hh_flows:
        pct_str = f"{100*c/total_pkts:.2f}%"
        vals = [c, pct_str] + [sk.query(k) for sk, _ in sketches]
        label = flow_str(k)[:FL]
        row = f"{label:<{FL}}{vals[0]:>{CW},}{vals[1]:>{CW}}" + "".join(f"{v:>{CW},}" for v in vals[2:])
        print(row)
    # Per-sketch ARE on the HH set
    if hh_flows:
        gt_hh = {k: c for k, c in hh_flows}
        print(f"  ARE on HH set:")
        for sk, name in sketches:
            est_hh = {k: sk.query(k) for k in gt_hh}
            print(f"    {name:<10}: {are(gt_hh, est_hh):.4f}")


if __name__ == "__main__":
    main()
