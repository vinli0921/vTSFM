"""Manual #14 backend/allocator feasibility checks; not HTTP runner support."""

import gc
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
native = pytest.mark.skipif(os.environ.get('VTSFM_RUN_TIMESFM_NATIVE_KV') != '1',
                          reason='Set VTSFM_RUN_TIMESFM_NATIVE_KV=1 on an idle GPU')


def test_native_kv_benchmark_help_is_cpu_safe():
    result = subprocess.run([sys.executable, '-m', 'benchmarks.timesfm_native_kv', '--help'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'experimental' in result.stdout.lower()


def test_native_kv_benchmark_requires_optin_before_creating_output(tmp_path):
    environment = os.environ.copy()
    environment.pop('VTSFM_RUN_TIMESFM_NATIVE_KV', None)
    output = tmp_path / 'not-submitted'
    result = subprocess.run([sys.executable, '-m', 'benchmarks.timesfm_native_kv',
                             '--backend', 'tensor', '--output', str(output)],
                            cwd=ROOT, env=environment, capture_output=True, text=True)
    assert result.returncode == 2 and 'VTSFM_RUN_TIMESFM_NATIVE_KV' in result.stderr
    assert not output.exists()


@pytest.fixture(scope='module')
def device_monitor():
    from benchmarks.run import GpuMonitor

    monitor = GpuMonitor(int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')))
    monitor.assert_idle()
    import torch

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield monitor
    gc.collect()
    torch.cuda.empty_cache()
    monitor.close()


@pytest.mark.gpu
@native
def test_paged_backend_causal_append_flip_isolation_release_and_reuse(device_monitor):
    import torch
    from torch.nn import functional as F
    from vtsfm.adapters.timesfm25.paged_cache import PagedCacheArena

    torch.manual_seed(14)
    arena = PagedCacheArena(1, 16, 80, 2, 32)
    free = arena.free_block_count
    arena.reserve('a', 32)
    arena.reserve('b', 32)
    assert set(arena.block_ids('a')).isdisjoint(arena.block_ids('b'))
    assert arena.free_block_count == free - 4
    history = {}

    def check(names, starts, sizes, stream):
        q, k, v = [torch.randn(sum(sizes), 16, 80, device='cuda') / 3 for _ in range(3)]
        actual = arena.attend(0, stream, q, k, v, names, starts, sizes)
        cursor = 0
        expected = []
        for name, start, size in zip(names, starts, sizes):
            section = slice(cursor, cursor + size)
            old = history.get((name, stream))
            keys = torch.cat((old[0], k[section])) if old else k[section]
            values = torch.cat((old[1], v[section])) if old else v[section]
            mask = torch.arange(start + size, device='cuda')[None] <= (
                start + torch.arange(size, device='cuda')[:, None])
            reference = F.scaled_dot_product_attention(
                q[section].transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1),
                attn_mask=mask, scale=1.0).transpose(0, 1)
            expected.append(reference)
            history[name, stream] = (keys, values)
            cursor += size
        torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-4, atol=1e-4)

    check(('a', 'b'), (0, 0), (3, 5), 0)
    check(('b', 'a'), (0, 0), (5, 3), 1)
    check(('a', 'b'), (3, 5), (4, 4), 0)
    check(('b', 'a'), (9, 7), (4, 4), 0)
    check(('a', 'b'), (3, 5), (4, 4), 1)
    arena.release('a')
    arena.release('a')
    assert arena.free_block_count == free - 2
    arena.reserve('reused', 32)
    check(('reused',), (0,), (3,), 0)
    check(('b',), (13,), (4,), 0)
    arena.release('b')
    arena.release('reused')
    assert not arena.allocated and arena.leased_cache_bytes == 0
    assert arena.free_block_count == free
    assert arena.cache_bytes > 0
    arena.close()
    arena.close()
    assert arena.cache_bytes == 0
    with pytest.raises(ValueError, match='closed'):
        arena.reserve('closed', 32)


@pytest.fixture(scope='module')
def runtimes(device_monitor):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime
    from vtsfm.adapters.timesfm25.spec import MODEL_ID, MODEL_REVISION

    directory = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                  allow_patterns=['config.json', 'model.safetensors'])
    reference, candidate = TimesFMRuntime({}), TimesFMRuntime({})
    weights = load_file(str(Path(directory) / 'model.safetensors'))
    for runtime in (reference, candidate):
        runtime.model.load_state_dict(weights)
        runtime.model.cuda().float().eval()
    return reference, candidate


def _context(lengths, offset=-1.):
    import torch

    value = torch.full((len(lengths), max(lengths)), float('nan'), device='cuda')
    for row, length in enumerate(lengths):
        time = torch.arange(length, device='cuda', dtype=torch.float32)
        value[row, -length:] = torch.sin(time / (7 + row)) + time / 1000 + offset
    return value


@pytest.mark.gpu
@native
@pytest.mark.parametrize('lengths,horizon', [((31,), 129), ((32, 32), 384),
                                           ((17, 65), 513), ((128,), 1024),
                                           ((2048,), 384), ((8192,), 1024)])
def test_native_pages_preserve_whole_forecast(runtimes, lengths, horizon):
    import torch
    from vtsfm.adapters.timesfm25.native_stepping import PagedTimesFMStepper

    reference, candidate = runtimes
    context = _context(lengths)
    expected = reference.forecast(context, horizon)
    capacity = (max(lengths) + 31) // 32 + 4 * ((horizon - 1) // 128)
    with PagedTimesFMStepper(candidate, max_sequences=2, max_tokens=capacity) as stepper:
        state = stepper.initialize(context, horizon)
        stepper.prefill([state])
        while not stepper.complete(state):
            stepper.advance([state])
        torch.testing.assert_close(stepper.finalize(state), expected, rtol=1e-4, atol=1e-4)
        assert not stepper.arena.allocated
        stepper.release(state)


@pytest.mark.gpu
@native
def test_native_page_lifetime_reorder_completion_cancel_and_reuse(runtimes):
    import torch
    from vtsfm.adapters.timesfm25.native_stepping import PagedTimesFMStepper

    reference, candidate = runtimes
    a, b = _context((33,)), _context((33,), 2.)
    expected_a, expected_b = reference.forecast(a, 513), reference.forecast(b, 257)
    with PagedTimesFMStepper(candidate, max_sequences=2, max_tokens=24) as stepper:
        free = stepper.arena.free_block_count
        first, second = stepper.initialize(a, 513), stepper.initialize(b, 257)
        assert set(stepper.sequence_ids(first)).isdisjoint(stepper.sequence_ids(second))
        stepper.prefill([first, second])
        stepper.advance([second, first])
        stepper.advance([first, second])
        torch.testing.assert_close(stepper.finalize(second), expected_b, rtol=1e-4, atol=1e-4)
        assert len(stepper.arena.allocated) == 1
        stepper.release(second)
        while not stepper.complete(first):
            stepper.advance([first])
        torch.testing.assert_close(stepper.finalize(first), expected_a, rtol=1e-4, atol=1e-4)
        stepper.release(first)
        assert stepper.arena.free_block_count == free
        cancelled = stepper.initialize(a, 513)
        stepper.prefill([cancelled])
        stepper.release(cancelled)
        again = stepper.initialize(b, 257)
        stepper.prefill([again])
        while not stepper.complete(again):
            stepper.advance([again])
        torch.testing.assert_close(stepper.finalize(again), expected_b, rtol=1e-4, atol=1e-4)
        stepper.release(again)
        assert stepper.arena.free_block_count == free and not stepper.arena.allocated


@pytest.mark.gpu
@native
def test_native_decode_batches_different_physical_cache_positions(runtimes):
    import torch
    from vtsfm.adapters.timesfm25.native_stepping import PagedTimesFMStepper

    reference, candidate = runtimes
    first, second = _context((160,), -2.), _context((33,), 1.)
    expected = [reference.forecast(context, 384) for context in (first, second)]
    with PagedTimesFMStepper(candidate, max_sequences=2, max_tokens=16) as stepper:
        a, b = [stepper.initialize(context, 384) for context in (first, second)]
        stepper.prefill([a])
        stepper.prefill([b])
        assert (a.cache_position, b.cache_position) == (5, 2)
        stepper.advance([b, a])
        stepper.advance([a, b])
        for state, values in zip((a, b), expected):
            torch.testing.assert_close(stepper.finalize(state), values, rtol=1e-4, atol=1e-4)
            stepper.release(state)
        assert not stepper.arena.allocated
