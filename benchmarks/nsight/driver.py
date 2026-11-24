"""Issue #16: in-process drivers for nsys (harness rounds 1, 0b, 0b16 and 0b8) on the vLLM 0.33 stack.

  --path native   official upstream forecast API (timesfm.TimesFM_2p5 forecast, Toto2Model.forecast,
                  uni2ts Moirai2Forecast.forward), as harness.py's native runners do
  --path runtime  the vTSFM adapter runtime (runtime.forecast) that ForecastModel.forward calls,
                  without vLLM, HTTP or the scheduler

Rounds follow profiling/harness.py (branch profiling-baseline; see workload.py): chiller_6 supply
temperature, the same windows(), horizon 96, NVTX preprocess / forward / postprocess per request.
Round 1: batch 1, context 2048, window advances one model patch per request.
Round 0b: batch 64, context 2048, fixed windows. Rounds 0b16 and 0b8 use 16 or 8 fixed windows, spread
evenly over the series by the same windows() rule (not a prefix of 0b).
Timed requests are bracketed by cudaProfilerStart/Stop so `nsys --capture-range=cudaProfilerApi`
records steady state only (model load and warm-up excluded).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload  # noqa: E402

HORIZON = workload.HORIZON


def nvtx(name):
    class R:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)

        def __exit__(self, *a):
            torch.cuda.nvtx.range_pop()
    return R()


def snapshot(adapter):
    from huggingface_hub import snapshot_download
    from vtsfm.adapters import get_adapter
    spec = get_adapter(adapter)
    return spec, snapshot_download(spec.model_id, revision=spec.model_revision)


# ------------------------------------------------------------------ native (upstream APIs)

class NativeTimesFM:
    def __init__(self, batch):
        import timesfm
        _, d = snapshot('timesfm25')
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(d, torch_compile=False)
        self.m.compile(timesfm.ForecastConfig(
            max_context=2048, max_horizon=128, per_core_batch_size=batch,
            normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))
        self.patch = 32

    def prepare(self, ctx):
        return list(ctx)  # the upstream API takes NumPy histories and moves them itself

    def forward(self, x):
        return self.m.forecast(horizon=HORIZON, inputs=x)

    def post(self, out):
        return np.asarray(out[1][..., 1:]).transpose(0, 2, 1)


class NativeToto:
    def __init__(self, batch):
        from toto2 import Toto2Model
        _, d = snapshot('toto2')
        self.m = Toto2Model.from_pretrained(d).to('cuda').eval()
        self.patch = int(self.m.config.patch_size)

    def prepare(self, ctx):
        t = torch.from_numpy(ctx).unsqueeze(1).to('cuda', non_blocking=True)
        return {'target': t, 'target_mask': torch.ones_like(t, dtype=torch.bool),
                'series_ids': torch.zeros(t.shape[:2], dtype=torch.long, device='cuda')}

    @torch.no_grad()
    def forward(self, x):
        return self.m.forecast(x, horizon=HORIZON, decode_block_size=768)

    def post(self, q):
        return q[:, :, 0, :HORIZON].float().cpu().numpy().transpose(1, 0, 2)


class NativeMoirai:
    def __init__(self, batch):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        _, d = snapshot('moirai2')
        module = Moirai2Module.from_pretrained(d)
        self.m = Moirai2Forecast(module=module, prediction_length=HORIZON, context_length=2048,
                                 target_dim=1, feat_dynamic_real_dim=0,
                                 past_feat_dynamic_real_dim=0).to('cuda').eval()
        self.patch = int(module.patch_size)

    def prepare(self, ctx):
        t = torch.from_numpy(ctx).unsqueeze(-1).to('cuda', non_blocking=True)
        return {'past_target': t, 'past_observed_target': torch.ones_like(t, dtype=torch.bool),
                'past_is_pad': torch.zeros(t.shape[:2], dtype=torch.bool, device='cuda')}

    @torch.no_grad()
    def forward(self, x):
        return self.m(**x)

    def post(self, out):
        return out.float().cpu().numpy()


# ------------------------------------------------------------------ vTSFM adapter runtime

class Runtime:
    """Build the vTSFM model wrapper exactly as tests/test_forecast_runtime.py does, then call
    wrapper.runtime.forecast(context, horizon) as ForecastModel.forward does per request."""

    def __init__(self, adapter):
        import importlib
        from types import SimpleNamespace
        from safetensors.torch import load_file
        from vllm.engine.arg_utils import EngineArgs
        os.environ['VTSFM_ADAPTER'] = adapter
        os.environ['VLLM_PLUGINS'] = 'vtsfm'
        spec, d = snapshot(adapter)
        mc = EngineArgs(
            model=d, config_format='vtsfm', runner='pooling', dtype='float32',
            skip_tokenizer_init=True, enforce_eager=True, max_model_len=8192,
            hf_overrides=json.loads(spec.vllm_args[spec.vllm_args.index('--hf-overrides') + 1]),
            io_processor_plugin='vtsfm', enable_mm_embeds=True).create_model_config()
        module, name = spec.model_class.split(':')
        w = getattr(importlib.import_module(module), name)(vllm_config=SimpleNamespace(model_config=mc))
        w.load_weights(load_file(str(Path(d) / 'model.safetensors')).items())
        self.w = w.to('cuda').eval()
        self.runtime = w.runtime
        cfg = mc.hf_config.vtsfm_config
        self.patch = {'timesfm25': 32}.get(adapter) or int(cfg.get('patch_size', 16))

    def prepare(self, ctx):
        # Same construction as ForecastIOProcessor.pre_process (NaN-left-padded FP32 rows);
        # vLLM's runner then moves the multimodal tensor to the GPU.
        c = torch.full(ctx.shape, float('nan'))
        for row, item in zip(c, ctx):
            row[-len(item):] = torch.tensor(item.tolist(), dtype=torch.float32)
        return c.to('cuda')

    @torch.inference_mode()
    def forward(self, c):
        return self.runtime.forecast(c, HORIZON).to(c.device)

    def post(self, out):
        return out.cpu().numpy()


NATIVE = {'timesfm25': NativeTimesFM, 'toto2': NativeToto, 'moirai2': NativeMoirai}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(NATIVE))
    ap.add_argument('--path', required=True, choices=['native', 'runtime'])
    ap.add_argument('--round', required=True, choices=['1', '0b', '0b16', '0b8'])
    ap.add_argument('--warm', type=int, default=5)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batch = {'1': 1, '0b': 64, '0b16': 16, '0b8': 8}[a.round]
    r = NATIVE[a.model](batch) if a.path == 'native' else Runtime(a.model)
    make = workload.round_inputs(workload.load_series(), a.round, r.patch)

    lat = []

    def step(i, timed):
        ctx = make(i)
        t0 = time.perf_counter()
        with nvtx('request'):
            with nvtx('preprocess'):
                x = r.prepare(ctx)
            with nvtx('forward'):
                raw = r.forward(x)
            with nvtx('postprocess'):
                q = r.post(raw)
        torch.cuda.synchronize()
        if timed:
            lat.append(time.perf_counter() - t0)
        return q

    for i in range(a.warm):
        q = step(i, False)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    for i in range(a.n):
        q = step(a.warm + i, True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    torch.cuda.profiler.stop()
    ms = sorted(1e3 * x for x in lat)
    res = {'model': a.model, 'path': a.path, 'round': a.round, 'batch': batch, 'context': 2048,
           'horizon': HORIZON, 'n': a.n, 'wall_s': wall, 'mean_ms': float(np.mean(ms)),
           'p50_ms': float(np.percentile(ms, 50)), 'p95_ms': float(np.percentile(ms, 95)),
           'output_shape': list(np.asarray(q).shape), 'torch': torch.__version__}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
