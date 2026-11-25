"""Factorial eager check of TimesFM flip packing and quantile assembly on CUDA."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from benchmarks.timesfm_compile_gate import compare


def case_from_archive(native_path, name_filter):
    archive = json.loads(native_path.read_bytes())
    matches = [entry for entry in archive['cases'] if name_filter in entry['case']['name']]
    if len(matches) != 1:
        raise ValueError(f'Expected one native case matching {name_filter!r}, got {len(matches)}')
    return matches[0]


def decode_pair(runtime, values, padding, horizon, split_flips):
    import torch

    def decode(inputs, mask):
        prefill, spread, autoregressive = runtime.model.decode(horizon, inputs, mask)
        blocks = [prefill[:, -1]]
        if autoregressive is not None:
            blocks.append(autoregressive.flatten(1, 2))
        return torch.cat(blocks, dim=1), spread

    if split_flips:
        positive, positive_spread = decode(values, padding)
        negative, negative_spread = decode(-values, padding)
    else:
        forecasts, spreads = decode(torch.cat((values, -values)), padding.repeat(2, 1))
        rows = values.shape[0]
        positive, negative = forecasts[:rows], forecasts[rows:]
        positive_spread, negative_spread = spreads[:rows], spreads[rows:]
    return positive, negative, positive_spread, negative_spread


def native_assembly(parts, horizon):
    # Adapted from the pinned TimesFM 2.5 forecast closure (Google, Apache-2.0).
    import torch

    def flip_quantiles(tensor):
        return torch.cat((tensor[..., :1], tensor[..., 1:].flip(-1)), dim=-1)

    forecast, flipped_forecast, spread, flipped_spread = parts
    spread = (spread - flip_quantiles(flipped_spread)) / 2
    forecast = (forecast - flip_quantiles(flipped_forecast)) / 2
    for index in (1, 2, 3, 4, 6, 7, 8, 9):
        forecast[..., index] = spread[..., index] - spread[..., 5] + forecast[..., 5]
    forecast = forecast[:, :horizon, :]
    for index in (4, 3, 2, 1):
        forecast[..., index] = torch.where(
            forecast[..., index] < forecast[..., index + 1],
            forecast[..., index], forecast[..., index + 1])
    for index in (6, 7, 8, 9):
        forecast[..., index] = torch.where(
            forecast[..., index] > forecast[..., index - 1],
            forecast[..., index], forecast[..., index - 1])
    return forecast


def vector_assembly(parts, horizon):
    import torch

    forecast, flipped_forecast, spread, flipped_spread = (
        tensor[:, :horizon, 1:] for tensor in parts
    )
    forecast = (forecast - flipped_forecast.flip(-1)) / 2
    spread = (spread - flipped_spread.flip(-1)) / 2
    median = forecast[..., 4:5]
    continuous = spread - spread[..., 4:5] + median
    lower = torch.cat((median, continuous[..., :4].flip(-1)), dim=-1)
    upper = torch.cat((median, continuous[..., 5:]), dim=-1)
    lower = lower.cummin(dim=-1).values[..., 1:].flip(-1)
    upper = upper.cummax(dim=-1).values[..., 1:]
    return torch.cat((lower, median, upper), dim=-1)


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from timesfm.torch.util import revin

    from model_integration_support import NativeForecaster
    from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime

    entry = case_from_archive(args.archived_native, args.case_filter)
    case = entry['case']
    native = NativeForecaster('timesfm25')
    expected = native.predict_array(case['data']).copy()
    np.testing.assert_array_equal(expected, np.asarray(entry['expected'], dtype=np.float32))
    metadata = native.metadata()
    checkpoint_config = Path(metadata['model_path']) / 'config.json'
    runtime = TimesFMRuntime(json.loads(checkpoint_config.read_bytes()))
    if runtime.compile_scope != 'eager':
        raise ValueError('The flip ablation requires the eager runtime')
    runtime.model.load_state_dict(native.reference.model.state_dict(), strict=True)
    runtime.model.to(device=native.device, dtype=torch.float32).eval()
    del native

    histories = [item['target'] for item in case['data']['inputs']]
    if len({len(history) for history in histories}) != 1:
        raise ValueError('This ablation requires equal original history lengths')
    context = torch.as_tensor(histories, device='cuda', dtype=torch.float32)
    observed = torch.isfinite(context)
    horizon = case['data']['parameters']['prediction_length']
    left = -context.shape[-1] % runtime.model.p

    with torch.inference_mode():
        def forecast(split_flips, use_native_assembly):
            values = F.pad(context.masked_fill(~observed, 0), (left, 0))
            padding = F.pad(~observed, (left, 0), value=True)
            positive = (values >= 0).all(dim=-1, keepdim=True).unsqueeze(-1)
            mean = values.mean(dim=-1, keepdim=True)
            std = values.std(dim=-1, keepdim=True)
            normalized = revin(values, mean, std, reverse=False)
            parts = decode_pair(runtime, normalized, padding, horizon, split_flips)
            quantiles = (native_assembly if use_native_assembly else vector_assembly)(parts, horizon)
            quantiles = revin(quantiles, mean, std, reverse=True)
            quantiles = torch.where(positive, quantiles.clamp_min(0), quantiles)
            if use_native_assembly:
                quantiles = quantiles[..., 1:]
            return quantiles.transpose(1, 2), parts

        actual = runtime.forecast(context, horizon)
        results = {}
        decoder_outputs = {}
        for split_flips in (False, True):
            for use_native_assembly in (False, True):
                label = ('split' if split_flips else 'packed') + (
                    '-native-assembly' if use_native_assembly else '-vector-assembly')
                output, parts = forecast(split_flips, use_native_assembly)
                if not split_flips and not use_native_assembly:
                    torch.testing.assert_close(output, actual, atol=0, rtol=0)
                if not use_native_assembly:
                    decoder_outputs[split_flips] = parts
                samples = []
                for _ in range(args.timed_calls):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    forecast(split_flips, use_native_assembly)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                results[label] = {
                    'parity': compare(output.cpu().numpy(), expected),
                    'direct_samples_ms': samples,
                    'direct_p50_ms': statistics.median(samples),
                }
        differences = {
            label: {
                'unequal_values': int((packed != split).sum().item()),
                'max_abs_error': (packed - split).abs().max().item(),
            }
            for label, packed, split in zip(
                ('positive_forecast', 'negative_forecast', 'positive_spread', 'negative_spread'),
                decoder_outputs[False], decoder_outputs[True], strict=True)
        }
    result = {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'case': case['name'], 'geometry': [*context.shape, horizon], 'left_padding': left,
        'archive_sha256': hashlib.sha256(args.archived_native.read_bytes()).hexdigest(),
        'fresh_native_matches_archive': True, 'native': metadata,
        'decoder_geometry_differences': differences, 'arms': results,
    }
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for name, result in results.items():
        print(name, result['parity'], round(result['direct_p50_ms'], 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archived-native', type=Path, required=True)
    parser.add_argument('--case-filter', default='condenser_water_flow')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timed-calls', type=int, default=5)
    args = parser.parse_args()
    if args.timed_calls < 1:
        parser.error('--timed-calls must be positive')
    run(args)
