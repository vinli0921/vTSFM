"""Explicitly opted-in, untimed native CUDA correctness for issue #47.

VTSFM_TEST_MOIRAI_MULTIVARIATE_ATTENTION=1 VTSFM_MOIRAI_CHECKPOINT=/local/snapshot \
  PYTHONPATH=src:tests .venv-moirai2/bin/python -m pytest -m gpu \
  tests/test_moirai_multivariate_attention_gpu.py -q -s

Set VTSFM_ATTENTION_RESULTS and TORCHINDUCTOR_CACHE_DIR to ignored task-local
directories. No downloads, server, benchmark samples, or public schema changes.
"""

from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_TEST_MOIRAI_MULTIVARIATE_ATTENTION') != '1',
    reason='Explicit native multivariate CUDA correctness opt-in is required')]
RTOL = ATOL = 1e-4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


@pytest.fixture
def gpu(request):
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from uni2ts.common.torch_util import packed_causal_attention_mask

    assert torch.cuda.is_available(), 'The explicit GPU opt-in requires CUDA'
    fraction = float(os.environ.get('VTSFM_ATTENTION_MEMORY_FRACTION', '0.18'))
    assert 0 < fraction <= 0.18
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.set_num_threads(2)
    torch.manual_seed(47)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Each test is an independent compilation experiment. This avoids combining
    # unrelated tiny/checkpoint shapes into the process-global default limit.
    torch._dynamo.reset()
    torch.cuda.reset_peak_memory_stats()
    directory = Path(os.environ.get('VTSFM_ATTENTION_RESULTS', 'results/moirai-multivariate-attention'))
    directory.mkdir(parents=True, exist_ok=True)
    name = request.node.name
    record = {
        'name': name, 'pid': os.getpid(), 'started_unix': time.time(),
        'boundary': 'untimed native CUDA numerical/kernel correctness; no HTTP or performance claim',
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'memory_fraction': fraction, 'rtol': RTOL, 'atol': ATOL,
        'commit': command('git', 'rev-parse', 'HEAD'), 'dirty': command('git', 'status', '--short'),
        'diff': command('git', 'diff'), 'cases': [],
        'occupancy_before': command('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'),
        'source_sha256': {p: digest(p) for p in (
            'src/vtsfm/adapters/moirai2/multivariate_attention.py',
            'tests/test_moirai_multivariate_attention.py', __file__)},
        'flags': {'matmul_precision': torch.get_float32_matmul_precision(),
                  'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                  'compile_reset_before_test': True,
                  'recompile_limit': torch._dynamo.config.recompile_limit,
                  'accumulated_recompile_limit': torch._dynamo.config.accumulated_recompile_limit,
                  'inductor_cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
                  'triton_cache': os.environ.get('TRITON_CACHE_DIR')},
    }

    def flush():
        (directory / f'{name}.json').write_text(json.dumps(record, indent=2, default=str) + '\n')

    @contextmanager
    def case(label):
        entry = {'label': label, 'status': 'running'}
        record['cases'].append(entry)
        flush()
        try:
            yield entry
            entry['status'] = 'passed'
        except BaseException:
            entry.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            flush()

    def save(entry, *, expected=None, actual=None, **inputs):
        arrays = {key: value.detach().cpu().numpy() for key, value in inputs.items()}
        if expected is not None:
            arrays['expected'] = expected.detach().cpu().numpy()
        if actual is not None:
            arrays['actual'] = actual.detach().cpu().numpy()
        path = directory / f'{name}-{entry["label"]}.npz'
        np.savez_compressed(path, **arrays)
        entry['raw'] = {'path': str(path), 'sha256': digest(path)}
        if expected is not None and actual is not None:
            entry.update(shape=list(actual.shape), max_abs=float((actual - expected).abs().max()),
                         allclose=bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
                         finite=bool(torch.isfinite(actual).all()))
        flush()

    before = {group: dict(values) for group, values in counters.items()}
    kernels_before = metrics.generated_kernel_count
    flush()
    yield torch, Moirai2Module, Moirai2Forecast, packed_causal_attention_mask, case, save, record
    torch.cuda.synchronize()
    record['compiler_counters_delta'] = {
        group: {key: value - before.get(group, {}).get(key, 0) for key, value in values.items()
                if value != before.get(group, {}).get(key, 0)}
        for group, values in counters.items()}
    record['generated_kernel_count_delta'] = metrics.generated_kernel_count - kernels_before
    record['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    record['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    record['occupancy_after_test'] = command(
        'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv')
    record['finished_unix'] = time.time()
    flush()


def tiny(Module, torch, *, heads=1):
    model = Module(d_model=heads * 64, d_ff=96, num_layers=2, patch_size=4,
                   max_seq_len=128, attn_dropout_p=0, dropout_p=0, num_predict_token=2).eval()
    with torch.no_grad():
        for layer in model.encoder.layers:
            weight = layer.self_attn.var_attn_bias.emb.weight
            weight[0] = torch.linspace(-1.9, -0.7, heads)
            weight[1] = torch.linspace(1.1, 2.3, heads)
    return model.cuda()


def forecast(Forecast, model, target, observed, padding, horizon):
    return Forecast(module=model, prediction_length=horizon, context_length=target.shape[1],
                    target_dim=target.shape[-1], feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0).to(target.device).eval()(
                        past_target=target, past_observed_target=observed, past_is_pad=padding)


def test_tiny_cuda_packed_bias_kernel_and_native_forecasts(gpu, monkeypatch):
    from vtsfm.adapters.moirai2.multivariate_attention import install_multivariate_attention

    torch, Module, Forecast, packed_mask, case, save, record = gpu
    reference = tiny(Module, torch, heads=2)
    candidate = copy.deepcopy(reference)
    samples = torch.tensor([[0, 0, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2, 2]], device='cuda')
    times = torch.tensor([[0, 1, 0, 2, 1, 0, 1], [2, 0, 1, 1, 0, 2, 1]], device='cuda')
    variates = torch.tensor([[0, 1, 0, 1, 2, 0, 1], [0, 1, 2, 0, 0, 1, 2]], device='cuda')
    mask = packed_mask(samples, times)
    hidden = torch.randn(2, 7, 128, device='cuda')
    kwargs = dict(attn_mask=mask, query_var_id=variates, kv_var_id=variates,
                  query_time_id=times, kv_time_id=times)
    with torch.inference_mode():
        expected = reference.encoder.layers[0].self_attn(hidden, hidden, hidden, **kwargs)
        undo = install_multivariate_attention(candidate)
        attention = candidate.encoder.layers[0].self_attn
        def forbidden(*args, **kwargs):
            raise AssertionError('candidate called the dense native float-bias builder')
        monkeypatch.setattr(attention, '_update_attn_mask', forbidden)
        with case('packed-head-bias') as entry:
            actual = attention(hidden, hidden, hidden, **kwargs)
            save(entry, expected=expected, actual=actual, hidden=hidden, sample_id=samples,
                 time_id=times, variate_id=variates, packed_mask=mask)
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        with case('compiled-kernel-activity') as entry:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                attention(hidden, hidden, hidden, **kwargs)
                torch.cuda.synchronize()
            entry['cuda_kernel_names'] = sorted({event.name for event in profile.events()
                                                if event.device_type == torch.autograd.DeviceType.CUDA})
            assert any('flex' in name.lower() for name in entry['cuda_kernel_names'])
        with case('drop-bias-negative-control') as entry:
            attention.var_attn_bias.emb.weight.zero_()
            dropped = attention(hidden, hidden, hidden, **kwargs)
            save(entry, expected=expected, actual=dropped, hidden=hidden)
            assert not entry['allclose'] and entry['max_abs'] > 1e-2
        undo()
        # Independent pristine control; the prior negative control modified only
        # a different candidate. Use another copy of the reference for forecasts.
        candidate = copy.deepcopy(reference)
        undo = install_multivariate_attention(candidate)
        for variables, horizon in [(3, 8), (1, 17)]:
            target = torch.randn(2, 19, variables, device='cuda')
            observed = torch.ones_like(target, dtype=torch.bool)
            padding = torch.zeros(2, 19, dtype=torch.bool, device='cuda')
            target[0, :7], observed[0, :7], padding[0, :7] = 0, False, True
            original = target.clone()
            with case(f'native-v{variables}-h{horizon}') as entry:
                expected = forecast(Forecast, reference, target, observed, padding, horizon)
                actual = forecast(Forecast, candidate, target, observed, padding, horizon)
                save(entry, expected=expected, actual=actual, target=target, observed=observed, padding=padding)
                torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
                assert torch.equal(target, original)
        undo()
    record['scope'] = 'Tiny official native models; multivariate first block and genuine univariate recursion'


def test_cached_checkpoint_cuda_native_parity_and_pristine_recursive_failure(gpu):
    from einops import EinopsError
    from safetensors.torch import load_file
    from vtsfm.adapters.moirai2.multivariate_attention import install_multivariate_attention
    from vtsfm.adapters.moirai2.spec import MODEL_REVISION

    torch, Module, Forecast, _, case, save, record = gpu
    checkpoint = Path(os.environ['VTSFM_MOIRAI_CHECKPOINT']).resolve()
    assert checkpoint.name == MODEL_REVISION, 'Use the pinned cached snapshot, not a floating revision'
    record['checkpoint'] = {'path': str(checkpoint), 'revision': MODEL_REVISION,
                            'sha256': {name: digest(checkpoint / name)
                                       for name in ('config.json', 'model.safetensors')}}
    reference = Module(**json.loads((checkpoint / 'config.json').read_text())).eval()
    reference.load_state_dict(load_file(checkpoint / 'model.safetensors'), strict=True)
    reference = reference.cuda()
    candidate = copy.deepcopy(reference)
    # All genuine references run before installing the candidate hook.
    cases = []
    with torch.inference_mode():
        for variables, context, horizon, ragged in [(3, 65, 64, False), (3, 65, 64, True),
                                                    (1, 65, 129, True)]:
            time_axis = torch.arange(context, device='cuda', dtype=torch.float32)
            target = torch.stack([2 + time_axis * 0.01 + torch.sin(time_axis / (7 + v)) + row * 0.2 + v
                                  for row in range(2) for v in range(variables)]).reshape(2, variables, context).transpose(1, 2).contiguous()
            observed = torch.ones_like(target, dtype=torch.bool)
            padding = torch.zeros(2, context, dtype=torch.bool, device='cuda')
            if ragged:
                target[0, :23], observed[0, :23], padding[0, :23] = 0, False, True
            expected = forecast(Forecast, reference, target, observed, padding, horizon)
            cases.append((f'checkpoint-v{variables}-c{context}-h{horizon}-ragged{ragged}',
                          target, observed, padding, expected))
        with case('pristine-multivariate-recursive-failure') as entry:
            target = torch.randn(1, 33, 2, device='cuda')
            assert all('forward' not in vars(layer.self_attn) for layer in reference.encoder.layers)
            try:
                forecast(Forecast, reference, target, torch.ones_like(target, dtype=torch.bool),
                         torch.zeros(1, 33, dtype=torch.bool, device='cuda'), 65)
            except EinopsError as exc:
                entry.update(expected_error=str(exc), native_traceback=traceback.format_exc())
                assert 'Shape mismatch, 8 != 4' in str(exc)
            else:
                pytest.fail('Revisit the documented upstream multivariate recursive limitation')
        _, target, observed, padding, expected = cases[0]
        changed = target + 0.75
        changed_expected = forecast(Forecast, reference, changed, observed, padding, 64)
        undo = install_multivariate_attention(candidate)
        for name, target, observed, padding, expected in cases:
            originals = [value.clone() for value in (target, observed, padding)]
            with case(name) as entry:
                actual = forecast(Forecast, candidate, target, observed, padding, expected.shape[2])
                save(entry, expected=expected, actual=actual, target=target, observed=observed, padding=padding)
                assert actual.shape == expected.shape and actual.dtype == torch.float32
                torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
                assert all(torch.equal(value, before) for value, before in zip((target, observed, padding), originals))
        # Revisit an earlier shape with changed values, then reproduce A exactly.
        name, target, observed, padding, expected = cases[0]
        with case('checkpoint-changed-input-and-aba') as entry:
            changed_actual = forecast(Forecast, candidate, changed, observed, padding, 64)
            replay = forecast(Forecast, candidate, target, observed, padding, 64)
            save(entry, expected=changed_expected, actual=changed_actual, target=changed, replay=replay,
                 original_expected=expected)
            torch.testing.assert_close(changed_actual, changed_expected, rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(replay, expected, rtol=RTOL, atol=ATOL)
        undo()
        dropped_model = copy.deepcopy(reference)
        undo_dropped = install_multivariate_attention(dropped_model)
        for layer in dropped_model.encoder.layers:
            layer.self_attn.var_attn_bias.emb.weight.zero_()
        with case('checkpoint-drop-bias-negative-control') as entry:
            dropped = forecast(Forecast, dropped_model, target, observed, padding, 64)
            save(entry, expected=expected, actual=dropped, target=target)
            assert not entry['allclose'] and entry['max_abs'] > 1e-3
        undo_dropped()


def test_tiny_cuda_multivariate_module_feedback_and_float_noise(gpu):
    """Recurrent native tensor contract; the unsupported forecast API is not repaired."""
    from vtsfm.adapters.moirai2.multivariate_attention import install_multivariate_attention

    torch, Module, _, _, case, save, record = gpu
    reference = tiny(Module, torch)
    candidate, noisy = copy.deepcopy(reference), copy.deepcopy(reference)
    target = torch.randn(2, 27, 4, device='cuda')
    observed = torch.ones_like(target, dtype=torch.bool)
    samples = torch.ones(2, 27, dtype=torch.long, device='cuda')
    variates = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [0] * 6 + [1] * 6 + [2] * 6,
                            device='cuda').expand(2, -1)
    times = torch.tensor(list(range(3)) * 3 + list(range(3, 9)) * 3, device='cuda').expand(2, -1)
    prediction = torch.zeros(2, 27, dtype=torch.bool, device='cuda')
    prediction[:, 9:], target[:, 9:] = True, 0
    target[0, [0, 3]], observed[0, [0, 3]], samples[0, [0, 3]] = 0, False, 0
    inputs = (target, observed, samples, times, variates, prediction)
    originals = [value.clone() for value in inputs]

    def rollout(model):
        x, obs, sample, time_id, var, pred = [value.clone() for value in inputs]
        anchors = torch.tensor([2, 5, 8], device='cuda')
        blocks = []
        for block in range(3):
            output = model(x, obs, sample, time_id, var, pred, training_mode=False)
            values = output.reshape(*output.shape[:-1], 2, 9, 4)[..., anchors, :, :, :].flatten(-4, -3)
            if block:
                values = torch.quantile(values.permute(0, 2, 1, 3, 4).flatten(2, 3),
                                        torch.tensor(model.quantile_levels, device='cuda'),
                                        dim=-2).permute(1, 2, 0, 3)
            blocks.append(values.clone())
            if block == 0:
                x, obs, sample, time_id, var, pred = [value.unsqueeze(1).expand(-1, 9, *value.shape[1:]).clone()
                                                     for value in (x, obs, sample, time_id, var, pred)]
            positions = torch.tensor([9 + block * 2, 10 + block * 2,
                                      15 + block * 2, 16 + block * 2,
                                      21 + block * 2, 22 + block * 2], device='cuda')
            x[..., positions, :] = values.permute(0, 2, 1, 3)
            pred[..., positions] = False
            anchors = positions.reshape(3, 2)[:, -1]
        return blocks

    def one_ulp(module, args, output):
        return torch.nextafter(output, torch.full_like(output, float('inf')))

    hooks = [layer.self_attn.register_forward_hook(one_ulp) for layer in noisy.encoder.layers]
    ranks = []
    shape_hook = candidate.register_forward_pre_hook(lambda module, args: ranks.append(list(args[0].shape)))
    with torch.inference_mode():
        expected, float_noise = rollout(reference), rollout(noisy)
        undo = install_multivariate_attention(candidate)
        with case('native-module-feedback-setup') as entry:
            actual = rollout(candidate)
            entry['native_input_shapes'] = ranks
            assert ranks == [[2, 27, 4], [2, 9, 27, 4], [2, 9, 27, 4]]
        for block, (golden, output, noise) in enumerate(zip(expected, actual, float_noise)):
            with case(f'native-module-feedback-block{block}') as entry:
                save(entry, expected=golden, actual=output, float_noise=noise, target=target,
                     observed=observed, sample_id=samples, time_id=times, variate_id=variates,
                     prediction_mask=prediction)
                entry['float_noise_max_abs'] = float((noise - golden).abs().max())
                torch.testing.assert_close(output, golden, rtol=RTOL, atol=ATOL)
                torch.testing.assert_close(noise, golden, rtol=RTOL, atol=ATOL)
        assert any(bool((noise != golden).any()) for noise, golden in zip(float_noise, expected))
        assert all(torch.equal(value, before) for value, before in zip(inputs, originals))
        undo()
    shape_hook.remove()
    for hook in hooks:
        hook.remove()
    record['scope'] = 'Manual native multivariate tensor feedback/81-sample quantile contract; not official multivariate forecast recursion'
