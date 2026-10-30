"""Baseline profiling harness for Chronos-2, TimesFM-2.5, Toto-2 and Moirai-2.0 on one GPU.

Rounds (one trace per model each):
  0a  AssetOpsBench run_recipe forecast, batch 1, zero-shot
  0b  native pipeline forward, batch 64, context 2048
  1   streaming: batch 1, window advances one patch per request, 20 requests
  2   native forward, batch 64, context 8192 (on OOM: largest batch that fits)
  3   0b in bf16
  4   run_recipe tsfm_anomaly_detection (conformal), batch 1
  sweep  timing only: batch x context x model, fp32
  5   optional large-GPU extension, timing only, own CSV: batch 512/1024 at context
      512/2048 (fp32), and the sweep grid in bf16

Usage: python profiling/harness.py --round 0 [--models chronos2,timesfm2,toto2,moirai2]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
AOB = Path(os.environ.get("AOB_ROOT", HERE.parent.parent / "AssetOpsBench"))
DATA = AOB / "src/couchdb/scenarios_data/shared/iot/chiller_6.json"
# Condenser Water Flow is ~53% zeros (chiller off) and ends in an off period, which makes
# holdout MASE meaningless; Supply Temperature is continuous. Override with --target.
TARGET = "Chiller 6 Supply Temperature"
HORIZON = 96
SEED = 0
N_WARM, N_TIMED, N_PROF = 5, 20, 3

CHECKPOINTS = {
    "chronos2": "amazon/chronos-2",
    "timesfm2": "google/timesfm-2.5-200m-transformers",
    "toto2": "Datadog/Toto-2.0-313m",
    "moirai2": "Salesforce/moirai-2.0-R-small",
}
SKTIME_CLASS = {
    "chronos2": "sktime.forecasting.chronos2.Chronos2Forecaster",
    "timesfm2": "sktime.forecasting.timesfm2.TimesFM2Forecaster",
    "toto2": "sktime.forecasting.toto2.Toto2Forecaster",
    "moirai2": "sktime.forecasting.moirai2.Moirai2Forecaster",
}


# ---------------------------------------------------------------- utilities

def seed_all():
    import random

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


@contextlib.contextmanager
def phase(name, timings=None):
    """NVTX range (for nsys) + record_function (for the Chrome trace) + optional wall time."""
    torch.cuda.nvtx.range_push(name)
    t0 = time.perf_counter()
    try:
        with record_function(name):
            yield
    finally:
        if timings is not None:
            torch.cuda.synchronize()
            timings.setdefault(name, []).append(time.perf_counter() - t0)
        torch.cuda.nvtx.range_pop()


def load_series(target=TARGET):
    """Target column in time order. The file's last rows are out of order and empty, so sort
    and drop rows without a value instead of interpolating them."""
    df = pd.DataFrame(json.loads(DATA.read_text()))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df[target].dropna().astype(float).to_numpy()


def windows(y, batch, context, *, stride=None, offset=0):
    """`batch` (context, future) pairs ending at different points of the series.

    The chiller series has ~2.9k points; when context + horizon does not fit, the series is
    tiled end-to-end, so long-context windows contain repeats (fine for timing, flagged for MASE).
    """
    need = context + HORIZON + (batch - 1) * (stride or 1) + offset
    reps = int(np.ceil(need / len(y)))
    ext = np.tile(y, reps) if reps > 1 else y
    room = len(ext) - context - HORIZON - offset
    stride = stride or max(1, room // max(batch, 1))
    ends = [context + offset + i * stride for i in range(batch)]
    ctx = np.stack([ext[e - context:e] for e in ends]).astype(np.float32)
    fut = np.stack([ext[e:e + HORIZON] for e in ends]).astype(np.float32)
    return ctx, fut, bool(context + HORIZON + offset + (batch - 1) * stride > len(y))


def mase(ctx, fut, pred):
    scale = np.mean(np.abs(np.diff(ctx, axis=1)), axis=1)
    return float(np.mean(np.mean(np.abs(fut - pred), axis=1) / np.maximum(scale, 1e-8)))


def crps_wql(fut, qpred, levels):
    """Weighted quantile loss (the usual CRPS approximation). qpred: (B, Q, H)."""
    loss = 0.0
    for i, q in enumerate(levels):
        d = fut - qpred[:, i, :]
        loss += np.sum(np.maximum(q * d, (q - 1) * d))
    return float(2 * loss / len(levels) / np.sum(np.abs(fut)))


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


# ---------------------------------------------------------------- native runners

class Runner:
    """prepare(ctx np (B,C)) -> inputs; forward(inputs) -> raw; post(raw) -> (median, quantiles)."""

    name = ""
    patch = 16
    bf16_mode = "weights"

    def autocast(self):
        return contextlib.nullcontext()


class Chronos2(Runner):
    name = "chronos2"

    def __init__(self, dtype):
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained(
            CHECKPOINTS["chronos2"], device_map="cuda", torch_dtype=dtype)
        self.levels = list(self.pipe.quantiles)
        self.patch = int(self.pipe.model.chronos_config.input_patch_size)

    def prepare(self, ctx):
        return torch.from_numpy(ctx).unsqueeze(1)  # (B, 1, C); the pipeline moves it to the GPU

    def forward(self, x):
        return self.pipe.predict(x, prediction_length=HORIZON, batch_size=x.shape[0],
                                 context_length=x.shape[-1])

    def post(self, raw):
        q = torch.stack([r[0] for r in raw]).float().cpu().numpy()  # (B, Q, H)
        return q[:, self.levels.index(0.5), :], q


class TimesFM2(Runner):
    name = "timesfm2"
    patch = 32

    def __init__(self, dtype):
        from sktime.forecasting.timesfm2 import TimesFM2Forecaster

        # reuse sktime's own loader so native and run_recipe paths load the same model class
        self.model = TimesFM2Forecaster(
            model_path=CHECKPOINTS["timesfm2"], device_map="cuda", dtype=dtype)._load_model()
        self.model.eval()
        cfg = self.model.config
        self.patch = int(getattr(cfg, "patch_length", None) or getattr(cfg, "patch_len", 32))
        self.levels = list(getattr(cfg, "quantiles", [0.1 * i for i in range(1, 10)]))

    def prepare(self, ctx):
        return torch.from_numpy(ctx).to("cuda", self.model.dtype, non_blocking=True)

    @torch.no_grad()
    def forward(self, x):
        return self.model(past_values=x)

    def post(self, out):
        median = out.mean_predictions[:, :HORIZON].float().cpu().numpy()
        full = getattr(out, "full_predictions", None)
        q = None
        if full is not None and full.shape[-1] == len(self.levels) + 1:
            q = full[:, :HORIZON, 1:].float().cpu().numpy().transpose(0, 2, 1)
            median = q[:, self.levels.index(min(self.levels, key=lambda v: abs(v - 0.5))), :]
        return median, q


class Toto2(Runner):
    name = "toto2"

    def __init__(self, dtype):
        from toto2 import Toto2Model

        model = Toto2Model.from_pretrained(CHECKPOINTS["toto2"]).to("cuda").eval()
        self.dtype = dtype
        if dtype == torch.bfloat16:
            # Toto's forecast loop builds fp32 tensors internally; run bf16 via autocast
            self.bf16_mode = "autocast"
        self.model = model
        self.patch = int(model.config.patch_size)
        self.levels = list(model.output_head.knots)

    def autocast(self):
        if self.dtype == torch.bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def prepare(self, ctx):
        target = torch.from_numpy(ctx).unsqueeze(1).to("cuda", non_blocking=True)  # (B, 1, C)
        return {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(target.shape[:2], dtype=torch.long, device="cuda"),
        }

    @torch.no_grad()
    def forward(self, inputs):
        with self.autocast():
            return self.model.forecast(inputs, horizon=HORIZON,
                                       has_missing_values=inputs["target"].shape[-1] % self.patch != 0)

    def post(self, q):
        q = q[:, :, 0, :HORIZON].float().cpu().numpy().transpose(1, 0, 2)  # (B, Q, H)
        return q[:, self.levels.index(0.5), :], q


class Moirai2(Runner):
    name = "moirai2"

    def __init__(self, dtype):
        # sktime's vendored uni2ts copy: the same code Moirai2Forecaster runs by default, and the
        # uni2ts package itself pins torch<2.5 and numpy 1.26
        from sktime.libs.uni2ts.moirai2_forecast import Moirai2Forecast
        from sktime.libs.uni2ts.moirai2_module import Moirai2Module

        module = Moirai2Module.from_pretrained(CHECKPOINTS["moirai2"])
        self.model = Moirai2Forecast(
            prediction_length=HORIZON, target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0, context_length=2048, module=module).to("cuda").eval()
        self.dtype = dtype
        if dtype == torch.bfloat16:
            # forward builds fp32 tensors internally (masks, quantile buffers); run bf16 via autocast
            self.bf16_mode = "autocast"
        self.patch = int(module.patch_size)
        self.levels = list(module.quantile_levels)

    def autocast(self):
        if self.dtype == torch.bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def prepare(self, ctx):
        # forward reads the context length from hparams to split context and prediction tokens
        self.model.hparams.context_length = ctx.shape[-1]
        target = torch.from_numpy(ctx).unsqueeze(-1).to("cuda", non_blocking=True)  # (B, C, 1)
        return {
            "past_target": target,
            "past_observed_target": torch.ones_like(target, dtype=torch.bool),
            "past_is_pad": torch.zeros(target.shape[:2], dtype=torch.bool, device="cuda"),
        }

    @torch.no_grad()
    def forward(self, inputs):
        with self.autocast():
            return self.model(**inputs)

    def post(self, out):
        # sktime's copy returns ((preds (B, H, Q),), None, None) where uni2ts returns (B, Q, H)
        q = out[0][0].float().cpu().numpy().transpose(0, 2, 1)  # (B, Q, H)
        return q[:, self.levels.index(0.5), :], q


RUNNERS = {"chronos2": Chronos2, "timesfm2": TimesFM2, "toto2": Toto2, "moirai2": Moirai2}


# ---------------------------------------------------------------- measurement

def measure(step, trace_path=None, n_warm=N_WARM, n_timed=N_TIMED, n_prof=N_PROF):
    """step(i, timings) runs one request. Returns latency stats, peak memory and phase means."""
    for i in range(n_warm):
        step(i, None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat, phases = [], {}
    for i in range(n_timed):
        t0 = time.perf_counter()
        step(n_warm + i, phases)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20
    if trace_path and n_prof:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            for i in range(n_prof):
                step(n_warm + n_timed + i, None)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace_path))
    lat_ms = sorted(x * 1000 for x in lat)
    return {
        "n_timed": n_timed,
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "mean_ms": statistics.mean(lat_ms),
        "peak_mem_mb": peak,
        "reserved_mem_mb": torch.cuda.max_memory_reserved() / 2**20,
        "phase_mean_ms": {k: 1000 * statistics.mean(v) for k, v in phases.items()},
    }


def native_config(runner, y, batch, context, trace_path=None, streaming=False, **kw):
    """Time the native pipeline on `batch` windows of length `context`."""
    if streaming:
        def make(i):  # window slides one model patch per request
            return windows(y, 1, context, stride=1, offset=i * runner.patch)
    else:
        fixed = windows(y, batch, context)

        def make(i):
            return fixed

    last = {}

    def step(i, timings):
        ctx, fut, tiled = make(i)
        with phase("preprocess", timings):
            x = runner.prepare(ctx)
        with phase("forward", timings):
            raw = runner.forward(x)
        with phase("postprocess", timings):
            median, q = runner.post(raw)
        last.update(ctx=ctx, fut=fut, median=median, q=q, tiled=tiled)

    stats = measure(step, trace_path, **kw)
    stats["throughput_series_per_s"] = batch * 1000 / stats["mean_ms"]
    stats["mase"] = mase(last["ctx"], last["fut"], last["median"])
    stats["crps_wql"] = (crps_wql(last["fut"], last["q"], runner.levels)
                         if last["q"] is not None else None)
    stats["context_tiled"] = last["tiled"]
    return stats


def recipe_estimator(model, dtype):
    params = {
        "chronos2": {"model_path": CHECKPOINTS["chronos2"], "seed": SEED,
                     "config": {"device_map": "cuda", "torch_dtype": dtype}},
        "timesfm2": {"model_path": CHECKPOINTS["timesfm2"], "device_map": "cuda", "dtype": dtype},
        "toto2": {"model_path": CHECKPOINTS["toto2"], "device": "cuda", "seed": SEED},
        # the estimator's default context_length is 200; match the other models' 2048
        "moirai2": {"checkpoint_path": CHECKPOINTS["moirai2"], "context_length": 2048,
                    "map_location": "cuda"},
    }[model]
    spec = {"name": model, "sktime_class": SKTIME_CLASS[model], "params": params}
    if model == "toto2":
        # the resolver's foundation-model keyword list lacks "toto", so without this the engine
        # treats Toto as fit_on_series and runs an expanding-window refit backtest
        spec["training_regime"] = "zero_shot"
    return spec


def recipe_config(model, y, context, task, trace_path=None, **kw):
    """Time AssetOpsBench run_recipe end to end at batch 1."""
    sys.path.insert(0, str(AOB / "src"))
    from servers.tsfm.engine import composition as C
    from servers.tsfm.substrate import resolver as R

    # NVTX/record_function around the engine's backtest folds and final fit/predict
    for fn in ("_backtest", "_backtest_zero_shot"):
        orig = getattr(C, fn)
        if not getattr(orig, "_wrapped", False):
            def wrapped(*a, _orig=orig, _n=fn, **k):
                with phase(f"backtest:{_n}"):
                    return _orig(*a, **k)
            wrapped._wrapped = True
            setattr(C, fn, wrapped)

    spec = recipe_estimator(model, torch.float32)
    default_spec = {k: v for k, v in spec.items() if k != "training_regime"}
    recipe = {"estimator": spec, "fh": list(range(1, HORIZON + 1)),
              "eval": {"metrics": ["mase"]}}
    if task == "anomaly":
        recipe.update(task="tsfm_anomaly_detection", method="conformal",
                      conformal={"coverage": 0.9})
    series = pd.Series(y[-(context + HORIZON):])
    last = {}

    def step(i, timings):
        with phase("run_recipe", timings):
            last["out"] = C.run_recipe(None, series, recipe, asset_id="Chiller 6")

    stats = measure(step, trace_path, **kw)
    out = last["out"]
    stats["throughput_series_per_s"] = 1000 / stats["mean_ms"]
    stats["engine_default_regime"] = R.training_regime(default_spec)
    stats["regime_used"] = out.get("training_regime")
    if task == "anomaly":
        stats["n_anomalies"] = out.get("n_anomalies")
        stats["n_observations"] = out.get("n_observations")
    else:
        stats["mase"] = out.get("backtest_score")
        stats["folds"] = out.get("folds")
    return stats


# ---------------------------------------------------------------- rounds

CSV_FIELDS = ["round", "model", "path", "dtype", "batch", "context", "status", "p50_ms",
              "p95_ms", "mean_ms", "throughput_series_per_s", "peak_mem_mb", "mase", "crps_wql",
              "context_tiled"]


def round_dir(rnd):
    return RESULTS / ("round5" if rnd == "5" else rnd)


def append_csv(row):
    # round 5 is an extension beyond the A6000 baseline; keep it out of the comparable CSV
    path = RESULTS / ("timings_round5.csv" if row["round"] == "5" else "timings.csv")
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(rnd, model, path, dtype, batch, context, fn, trace=True):
    out_dir = round_dir(rnd)
    out_dir.mkdir(parents=True, exist_ok=True)
    dname = "bf16" if dtype == torch.bfloat16 else "fp32"
    row = {"round": rnd, "model": model, "path": path, "dtype": dname,
           "batch": batch, "context": context}
    free_cuda()
    seed_all()
    try:
        stats = fn(out_dir / f"{model}.trace.json" if trace else None)
        row.update(status="ok", **stats)
    except Exception as exc:  # record OOMs and adapter failures instead of aborting the round
        row.update(status="oom" if is_oom(exc) else f"error: {type(exc).__name__}: {exc}"[:300])
        free_cuda()
    append_csv(row)
    print(json.dumps({k: row.get(k) for k in CSV_FIELDS}), flush=True)
    return row


def write_json(rnd, model, rows):
    (round_dir(rnd) / f"{model}.json").write_text(json.dumps(rows, indent=2, default=str))


SWEEP_CONTEXTS, SWEEP_BATCHES = (512, 2048, 8192), (1, 4, 16, 64, 256)


def timing_grid(rnd, model, runner, y, dtype, batches, contexts, path="native"):
    """Timing-only rows (no trace) over batch x context, context-major."""
    rows = []
    for context in contexts:
        for batch in batches:
            rows.append(run_one(rnd, model, path, dtype, batch, context,
                                lambda t, b=batch, c=context: native_config(runner, y, b, c, None),
                                trace=False))
    return rows


def run_round5(model, y):
    fp32, bf16 = torch.float32, torch.bfloat16
    rows = []
    runner = RUNNERS[model](fp32)
    # (a) batches past the sweep's 256; an OOM row marks the largest batch that fits
    rows += timing_grid("5", model, runner, y, fp32, (512, 1024), (512, 2048), path="native_bigbatch")
    meta_fp32 = {"bf16_mode": None, "patch": runner.patch, "quantile_levels": runner.levels}
    rows = [{**r, **meta_fp32} for r in rows]
    del runner
    free_cuda()
    # (b) the sweep grid in bf16
    runner = RUNNERS[model](bf16)
    meta_bf16 = {"bf16_mode": runner.bf16_mode, "patch": runner.patch,
                 "quantile_levels": runner.levels}
    rows += [{**r, **meta_bf16} for r in
             timing_grid("5", model, runner, y, bf16, SWEEP_BATCHES, SWEEP_CONTEXTS)]
    write_json("5", model, rows)
    del runner
    free_cuda()


def run_round(rnd, models, y):
    fp32, bf16 = torch.float32, torch.bfloat16
    for model in models:
        rows = []
        if rnd == "5":
            run_round5(model, y)
            continue
        if rnd in ("0a", "4"):
            task = "forecast" if rnd == "0a" else "anomaly"
            kw = {} if rnd == "0a" else {"n_warm": 1, "n_timed": 5, "n_prof": 1}
            rows.append(run_one(rnd, model, "run_recipe", fp32, 1, 2048,
                                lambda t: recipe_config(model, y, 2048, task, t, **kw)))
            write_json(rnd, model, rows)
            continue

        dtype = bf16 if rnd == "3" else fp32
        runner = RUNNERS[model](dtype)
        if rnd in ("0b", "3"):
            rows.append(run_one(rnd, model, "native", dtype, 64, 2048,
                                lambda t: native_config(runner, y, 64, 2048, t)))
        elif rnd == "1":
            rows.append(run_one(rnd, model, "native_streaming", dtype, 1, 2048,
                                lambda t: native_config(runner, y, 1, 2048, t, streaming=True)))
        elif rnd == "2":
            row = run_one(rnd, model, "native", dtype, 64, 8192,
                          lambda t: native_config(runner, y, 64, 8192, t))
            rows.append(row)
            batch = 64
            while row["status"] == "oom" and batch > 1:
                batch //= 2
                row = run_one(rnd, model, "native", dtype, batch, 8192,
                              lambda t, b=batch: native_config(runner, y, b, 8192, t))
                rows.append(row)
        elif rnd == "sweep":
            rows += timing_grid(rnd, model, runner, y, dtype, SWEEP_BATCHES, SWEEP_CONTEXTS)
        rows_meta = {"bf16_mode": runner.bf16_mode if dtype == bf16 else None,
                     "patch": runner.patch, "quantile_levels": runner.levels}
        write_json(rnd, model, [{**r, **rows_meta} for r in rows])
        del runner
        free_cuda()


def write_env(target):
    import importlib.metadata as md

    lines = [
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"host: {platform.node()}",
        f"gpu: {torch.cuda.get_device_name(0)}",
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}  cuda: {torch.version.cuda}  cudnn: {torch.backends.cudnn.version()}",
    ]
    try:
        import subprocess

        lines.append("driver: " + subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip())
    except Exception as exc:
        lines.append(f"driver: unavailable ({exc})")
    for pkg in ("sktime", "skpro", "chronos-forecasting", "transformers", "toto-2",
                "toto-models", "gluonts", "lightning", "numpy", "pandas"):
        try:
            lines.append(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            lines.append(f"{pkg}: not installed")
    lines.append(f"data: {DATA.relative_to(AOB)} column '{target}', horizon {HORIZON}")
    lines.append(f"method: {N_WARM} warm-up, {N_TIMED} timed, profiler over {N_PROF}, seed {SEED}")
    (RESULTS / "env.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True,
                    help="0 (=0a+0b), 0a, 0b, 1, 2, 3, 4, sweep, 5, or a comma list")
    ap.add_argument("--models", default=",".join(RUNNERS))
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_env(args.target)
    y = load_series(args.target)
    rounds = []
    for r in args.round.split(","):
        rounds += ["0a", "0b"] if r == "0" else [r]
    models = args.models.split(",")
    if len(rounds) == 1 and len(models) == 1:
        run_round(rounds[0], models, y)
        return
    # one process per (round, model): sktime caches loaded models process-wide, which would
    # leak earlier models into later peak-memory readings
    import subprocess

    for rnd in rounds:
        for model in models:
            rc = subprocess.run([sys.executable, __file__, "--round", rnd, "--models", model,
                                 "--target", args.target], check=False).returncode
            if rc != 0:  # e.g. killed by the job's memory cgroup: nothing else gets recorded
                append_csv({"round": rnd, "model": model, "status": f"crashed: exit {rc}"})
                print(f"round {rnd} model {model} crashed with exit code {rc}", flush=True)


if __name__ == "__main__":
    main()
