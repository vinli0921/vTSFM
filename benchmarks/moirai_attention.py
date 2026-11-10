"""Issue #24: native Moirai 2.0 attention memory/latency, unpatched vs univariate patch.

Runs the pinned ``Moirai2Forecast.forward`` directly (the #45 tensor path), not HTTP.
Each grid point runs in a fresh subprocess so an OOM is isolated and peak memory
counters start clean.

  python -m benchmarks.moirai_attention grid --output results/
  python -m benchmarks.moirai_attention parity --output results/
  python -m benchmarks.moirai_attention accuracy --output results/
  python -m benchmarks.moirai_attention profile --output results/ --batch 64 --context 7168 --horizon 96
  python -m benchmarks.moirai_attention snapshot --output results/ --batch 64 --context 7168 --horizon 96
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID = 'Salesforce/moirai-2.0-R-small'
MODEL_REVISION = '30f43ff08c8494f4943ae1521e9d4e94a0fbb389'
ARMS = ('unpatched', 'patched')
GRID_BATCH = (1, 16, 64, 256)
GRID_CONTEXT = (512, 2048, 4096, 7168)
GRID_HORIZON = (64, 96)
LONG_HORIZON = [(b, c, 1024) for b in (1, 16, 64) for c in (512, 2048, 6144)]
DEVICE = os.environ.get('VTSFM_BENCH_DEVICE', 'cuda')  # 'cpu' only for smoke tests


def _setup(arm, device=None):
    device = device or DEVICE
    import torch
    from uni2ts.model.moirai2 import Moirai2Module

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = Moirai2Module.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    if arm == 'patched':
        from vtsfm.adapters.moirai2.attention import install_univariate_attention
        install_univariate_attention(module)
    elif arm != 'unpatched':
        raise ValueError(f'unknown arm {arm}')
    return module


def series(rows, length, seed, device=None):
    device = device or DEVICE
    """Seasonal random-walk histories; deterministic and strictly finite."""
    import torch

    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    base = 10 + 3 * torch.sin(2 * torch.pi * t / 24)[None] + torch.sin(2 * torch.pi * t / 168)[None]
    return (base + torch.cumsum(0.3 * torch.randn(rows, length, generator=g), 1)).to(device)


def forecast(module, context, horizon, padding=None):
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast

    with torch.inference_mode():
        observed = torch.isfinite(context)
        padding = ~observed if padding is None else padding
        runner = Moirai2Forecast(module=module, prediction_length=horizon, context_length=context.shape[-1],
                                 target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0
                                 ).to(context.device).eval()
        return runner(past_target=context.masked_fill(~observed, 0).unsqueeze(-1),
                      past_observed_target=observed.unsqueeze(-1), past_is_pad=padding)


def tokens(context, horizon):
    return -(-context // 16) + -(-horizon // 16)


def point(arm, batch, context, horizon, reps, budget_s):
    """One grid point in this process; prints one JSON line."""
    import torch

    record = dict(arm=arm, batch=batch, context=context, horizon=horizon, tokens=tokens(context, horizon))
    try:
        module = _setup(arm)
        x = series(batch, context, 7)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            forecast(module, x, horizon)
        torch.cuda.synchronize()
        times, started = [], time.perf_counter()
        while len(times) < reps and (not times or time.perf_counter() - started < budget_s):
            t0 = time.perf_counter()
            forecast(module, x, horizon)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        record.update(status='ok', reps=len(times), p50_ms=1e3 * times[len(times) // 2],
                      min_ms=1e3 * times[0], series_per_s=batch / times[len(times) // 2],
                      peak_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_alloc_above_weights_mib=(torch.cuda.max_memory_allocated() - base) / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    except torch.OutOfMemoryError as exc:
        record.update(status='oom', error=str(exc).splitlines()[0][:200])
    except Exception as exc:  # report and continue the grid
        record.update(status='error', error=f'{type(exc).__name__}: {exc}'[:300])
    print(json.dumps(record), flush=True)


def grid(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'grid.csv'
    configs = [(b, c, h) for h in GRID_HORIZON for c in GRID_CONTEXT for b in GRID_BATCH] + LONG_HORIZON
    fields = ['arm', 'batch', 'context', 'horizon', 'tokens', 'status', 'reps', 'p50_ms', 'min_ms',
              'series_per_s', 'peak_alloc_mib', 'peak_alloc_above_weights_mib', 'peak_reserved_mib', 'error']
    done = set()
    if path.exists():
        done = {(r['arm'], int(r['batch']), int(r['context']), int(r['horizon'])) for r in csv.DictReader(path.open())}
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not done:
            writer.writeheader()
        for batch, context, horizon in configs:
            for arm in ARMS:
                if (arm, batch, context, horizon) in done:
                    continue
                reps = 10 if horizon <= 96 else 3
                proc = subprocess.run([sys.executable, '-m', 'benchmarks.moirai_attention', 'point', arm,
                                       str(batch), str(context), str(horizon), str(reps), str(args.budget)],
                                      capture_output=True, text=True, timeout=args.timeout)
                lines = [line for line in proc.stdout.splitlines() if line.startswith('{')]
                record = json.loads(lines[-1]) if lines else dict(
                    arm=arm, batch=batch, context=context, horizon=horizon, tokens=tokens(context, horizon),
                    status='crash', error=(proc.stderr.strip().splitlines() or ['?'])[-1][:300])
                writer.writerow({k: record.get(k, '') for k in fields})
                stream.flush()
                print(json.dumps(record), flush=True)


def _compare(ref, out):
    diff = (out - ref).abs()
    return dict(max_abs=float(diff.max()), max_rel=float((diff / (ref.abs() + 1e-6)).max()),
                ref_std=float(ref.std()),
                frac_outside_1e4=float((diff > 1e-4 + 1e-4 * ref.abs()).float().mean()),
                allclose_1e4=bool(diff.le(1e-4 + 1e-4 * ref.abs()).all()))


def parity(args):
    import torch
    from vtsfm.adapters.moirai2.attention import install_univariate_attention

    module = _setup('unpatched')
    ragged = series(4, 512, 4)
    ragged[0, :300] = float('nan')
    ragged[1, :17] = float('nan')
    cases = [('B4 C512 H64 (single pass)', series(4, 512, 1), 64),
             ('B4 C512 H96 (recursive x9)', series(4, 512, 2), 96),
             ('B4 C512 H96 ragged left padding', ragged, 96),
             *([('B16 C7168 H96', series(16, 7168, 3), 96)] if DEVICE == 'cuda' else []),
             ('B2 C2048 H1024', series(2, 2048, 5), 1024)]
    rows = []
    for name, context, horizon in cases:
        ref = forecast(module, context, horizon)
        rerun = forecast(module, context, horizon)
        g = torch.Generator().manual_seed(11)
        noise = (1e-7 * torch.randn(context.shape, generator=g)).to(context.device)
        noisy = forecast(module, context * (1 + noise), horizon)
        uninstall = install_univariate_attention(module, verify=True)
        try:
            out = forecast(module, context, horizon)
        finally:
            uninstall()
        row = dict(case=name, **{f'patched_{k}': v for k, v in _compare(ref, out).items()},
                   rerun_max_abs=float((rerun - ref).abs().max()),
                   noise1e7_max_abs=_compare(ref, noisy)['max_abs'])
        row['within_noise'] = row['patched_max_abs'] <= max(row['noise1e7_max_abs'], 1e-4 + 1e-4 * float(ref.abs().max()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    _write(Path(args.output) / 'parity.csv', rows)


def _mase_crps(quantiles, target, history, season=24):
    """quantiles [rows, 9, H] (levels 0.1..0.9), target [rows, H], history [rows, C]."""
    import torch

    levels = torch.arange(1, 10, device=target.device, dtype=torch.float32) / 10
    median = quantiles[:, 4]
    scale = (history[:, season:] - history[:, :-season]).abs().mean(-1).clamp_min(1e-8)
    mase = ((median - target).abs().mean(-1) / scale).mean()
    error = target[:, None] - quantiles
    pinball = torch.maximum(levels[None, :, None] * error, (levels[None, :, None] - 1) * error)
    wql = 2 * pinball.sum((-1, -2)).sum() / (len(levels) * target.abs().sum())
    return float(mase), float(wql)


def accuracy(args):
    from vtsfm.adapters.moirai2.attention import install_univariate_attention

    module = _setup('unpatched')
    rows = []
    for context, horizon in (((512, 96), (2048, 96), (2048, 1024), (7168, 96)) if DEVICE == 'cuda' else ((512, 96),)):
        full = series(32, context + horizon, 23)
        history, target = full[:, :context], full[:, context:]
        ref = forecast(module, history, horizon)
        uninstall = install_univariate_attention(module)
        try:
            out = forecast(module, history, horizon)
        finally:
            uninstall()
        (m0, c0), (m1, c1) = _mase_crps(ref, target, history), _mase_crps(out, target, history)
        row = dict(context=context, horizon=horizon, rows=32, mase_unpatched=m0, mase_patched=m1,
                   crps_wql_unpatched=c0, crps_wql_patched=c1, mase_delta=m1 - m0, crps_delta=c1 - c0)
        rows.append(row)
        print(json.dumps(row), flush=True)
    _write(Path(args.output) / 'accuracy.csv', rows)


def profile(args):
    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile

    rows = []
    for arm in ARMS:
        module = _setup(arm)
        x = series(args.batch, args.context, 7)
        forecast(module, x, args.horizon)
        torch.cuda.synchronize()
        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            forecast(module, x, args.horizon)
            torch.cuda.synchronize()
        events = [e for e in prof.key_averages() if e.device_time_total > 0]
        total = sum(e.self_device_time_total for e in events) or 1
        for e in sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:15]:
            name = e.key
            kind = ('fused-attention' if any(s in name.lower() for s in ('fmha', 'efficient_attention', 'flash'))
                    else 'gemm' if 'gemm' in name.lower() else 'softmax' if 'softmax' in name.lower() else 'other')
            rows.append(dict(arm=arm, kernel=name[:120], kind=kind, calls=e.count,
                             self_ms=e.self_device_time_total / 1e3, share=e.self_device_time_total / total))
        del module
        torch.cuda.empty_cache()
    _write(Path(args.output) / f'kernels_B{args.batch}_C{args.context}_H{args.horizon}.csv', rows)
    for row in rows:
        print(json.dumps(row))


def snapshot(args):
    """Record allocator history and report what is live at the peak, grouped by source frame."""
    import pickle
    import torch

    out = Path(args.output)
    rows = []
    for arm in ARMS:
        module = _setup(arm)
        x = series(args.batch, args.context, 7)
        torch.cuda.synchronize()
        torch.cuda.memory._record_memory_history(max_entries=200000)
        try:
            forecast(module, x, args.horizon)
            torch.cuda.synchronize()
            snap = torch.cuda.memory._snapshot()
        except torch.OutOfMemoryError:
            snap = torch.cuda.memory._snapshot()
            print(json.dumps(dict(arm=arm, status='oom during snapshot')))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        stem = f'snapshot_{arm}_B{args.batch}_C{args.context}_H{args.horizon}'
        with open(out / f'{stem}.pickle', 'wb') as stream:
            pickle.dump(snap, stream)
        live, total, peak, peak_live = {}, 0, 0, {}
        for event in snap['device_traces'][0]:
            action, size = event['action'], event['size']
            if action == 'alloc':
                live[event['addr']] = (size, event.get('frames', []))
                total += size
                if total > peak:
                    peak, peak_live = total, dict(live)
            elif action in ('free_completed',):
                if event['addr'] in live:
                    total -= live.pop(event['addr'])[0]
        groups = {}
        for size, frames in peak_live.values():
            site = next((f"{Path(f['filename']).name}:{f['line']}" for f in frames
                         if any(k in f['filename'] for k in ('uni2ts', 'vtsfm', 'benchmarks'))), 'other')
            groups[site] = groups.get(site, 0) + size
        for site, size in sorted(groups.items(), key=lambda kv: -kv[1])[:12]:
            rows.append(dict(arm=arm, site=site, live_mib_at_peak=size / 2**20, peak_traced_mib=peak / 2**20))
        del module
        torch.cuda.empty_cache()
    _write(out / f'snapshot_top_B{args.batch}_C{args.context}_H{args.horizon}.csv', rows)
    for row in rows:
        print(json.dumps(row))


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('point')
    p.add_argument('arm')
    for name in ('batch', 'context', 'horizon', 'reps'):
        p.add_argument(name, type=int)
    p.add_argument('budget', type=float)
    for name in ('grid', 'parity', 'accuracy', 'profile', 'snapshot'):
        q = sub.add_parser(name)
        q.add_argument('--output', required=True)
        q.add_argument('--batch', type=int, default=64)
        q.add_argument('--context', type=int, default=7168)
        q.add_argument('--horizon', type=int, default=96)
        q.add_argument('--budget', type=float, default=60.0)
        q.add_argument('--timeout', type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.command == 'point':
        return point(args.arm, args.batch, args.context, args.horizon, args.reps, args.budget)
    return dict(grid=grid, parity=parity, accuracy=accuracy, profile=profile, snapshot=snapshot)[args.command](args)


if __name__ == '__main__':
    main()
