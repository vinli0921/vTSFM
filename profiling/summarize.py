"""Markdown tables for one round of harness results, with an optional baseline to compare.

    python profiling/summarize.py --round 0b [--baseline other/timings.csv] [--traces]

Prints per-model latency, throughput, memory and quality, the phase split from
<round>/<model>.json, ratios against the baseline CSV's matching (round, model, dtype,
batch, context) row, and with --traces the top CUDA kernels from each Chrome trace.
Findings are written by hand under these tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = ("round", "model", "dtype", "batch", "context")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v, fmt="{:.1f}"):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "–"


def ratio(base, new):
    try:
        return f"{float(base) / float(new):.2f}×"
    except (TypeError, ValueError, ZeroDivisionError):
        return "–"


def top_kernels(trace_path, n=8):
    """(total CUDA kernel ms, [(name, ms, share)]) over the profiled requests."""
    events = json.loads(Path(trace_path).read_text())["traceEvents"]
    by_name = defaultdict(float)
    for e in events:
        if e.get("cat") == "kernel" and e.get("ph") == "X":
            by_name[e["name"]] += e.get("dur", 0) / 1000
    total = sum(by_name.values())
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:n]
    return total, [(k, v, v / total if total else 0) for k, v in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--baseline", help="timings.csv from another GPU")
    ap.add_argument("--traces", action="store_true")
    args = ap.parse_args()
    res = Path(args.results)
    csv_name = "timings_round5.csv" if args.round == "5" else "timings.csv"
    rows = [r for r in read_csv(res / csv_name) if r["round"] == args.round]
    # a rerun appends; keep the last row per configuration
    rows = list({tuple(r.get(k) for k in KEY): r for r in rows}.values())
    base = {}
    if args.baseline:
        base = {tuple(r.get(k) for k in KEY): r for r in read_csv(args.baseline)}

    print(f"### Round {args.round}: results\n")
    print("| model | path | dtype | batch | context | status | p50 ms | p95 ms | series/s "
          "| peak MB | MASE | CRPS (WQL) | tiled |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['path']} | {r['dtype']} | {r['batch']} | {r['context']} "
              f"| {r['status'][:60]} | {num(r['p50_ms'])} | {num(r['p95_ms'])} "
              f"| {num(r['throughput_series_per_s'])} | {num(r['peak_mem_mb'], '{:.0f}')} "
              f"| {num(r['mase'], '{:.3f}')} | {num(r['crps_wql'], '{:.3f}')} "
              f"| {r.get('context_tiled') or ''} |")

    if base:
        print("\n### Against the baseline\n")
        print("| model | dtype | batch | context | base p50 ms | p50 ms | speedup "
              "| base peak MB | peak MB | base MASE | MASE |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            b = base.get(tuple(r.get(k) for k in KEY))
            if b is None:
                print(f"| {r['model']} | {r['dtype']} | {r['batch']} | {r['context']} "
                      f"| no baseline row | {num(r['p50_ms'])} | – | – "
                      f"| {num(r['peak_mem_mb'], '{:.0f}')} | – | {num(r['mase'], '{:.3f}')} |")
                continue
            print(f"| {r['model']} | {r['dtype']} | {r['batch']} | {r['context']} "
                  f"| {num(b['p50_ms'])} | {num(r['p50_ms'])} | {ratio(b['p50_ms'], r['p50_ms'])} "
                  f"| {num(b['peak_mem_mb'], '{:.0f}')} | {num(r['peak_mem_mb'], '{:.0f}')} "
                  f"| {num(b['mase'], '{:.3f}')} | {num(r['mase'], '{:.3f}')} |")

    rdir = res / ("round5" if args.round == "5" else args.round)
    phase_rows = []
    for jf in sorted(rdir.glob("*.json")):
        if jf.name.endswith(".trace.json"):
            continue
        for r in json.loads(jf.read_text()):
            pm = r.get("phase_mean_ms") or {}
            if pm:
                phase_rows.append((r["model"], r.get("dtype"), r.get("batch"), r.get("context"), pm))
    if phase_rows:
        names = sorted({k for *_, pm in phase_rows for k in pm})
        print("\n### Phase split (mean ms per request)\n")
        print("| model | dtype | batch | context | " + " | ".join(names) + " |")
        print("|---|---|---|---|" + "---|" * len(names))
        for model, dtype, batch, context, pm in phase_rows:
            print(f"| {model} | {dtype} | {batch} | {context} | "
                  + " | ".join(num(pm.get(k)) for k in names) + " |")

    if args.traces:
        for tf in sorted(rdir.glob("*.trace.json")):
            total, top = top_kernels(tf)
            print(f"\n### Top CUDA kernels: {tf.stem.removesuffix('.trace')} "
                  f"({total:.1f} ms of kernel time over the profiled requests)\n")
            print("| kernel | ms | share |")
            print("|---|---|---|")
            for name, ms, share in top:
                print(f"| `{name[:90]}` | {ms:.1f} | {share:.0%} |")


if __name__ == "__main__":
    main()
