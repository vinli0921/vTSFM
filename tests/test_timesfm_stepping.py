"""Manual pinned-checkpoint parity for resumable TimesFM forecast blocks.

    CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_STEPPING=1 PYTHONPATH=src:tests:. \\
      .venv-timesfm25/bin/python -m pytest tests/test_timesfm_stepping.py -q -s

Synthetic boundary/state probes compare with the unchanged tensor runtime.
They establish numerical/state correctness, not accuracy or serving speedups.
"""

import gc
import os
from pathlib import Path

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_TIMESFM_STEPPING') != '1',
    reason='Set VTSFM_RUN_TIMESFM_STEPPING=1 in the pinned TimesFM CUDA environment')]


@pytest.fixture(scope='module')
def runtime():
    from benchmarks.run import GpuMonitor

    monitor = GpuMonitor(int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')))
    monitor.assert_idle()
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime
    from vtsfm.adapters.timesfm25.spec import MODEL_ID, MODEL_REVISION

    assert torch.cuda.is_available(), 'The stepper checks require CUDA'
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                  allow_patterns=['config.json', 'model.safetensors'])
    model = TimesFMRuntime({})
    model.model.load_state_dict(load_file(str(Path(checkpoint) / 'model.safetensors')))
    model.model.cuda().float().eval()
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    monitor.close()


def _context(lengths, offset=-1.0):
    import torch

    result = torch.full((len(lengths), max(lengths)), float('nan'), device='cuda')
    for row, length in enumerate(lengths):
        time = torch.arange(length, dtype=torch.float32, device='cuda')
        values = torch.sin(time / (7 + row)) + offset + row / 3
        result[row, -length:] = values
    return result


def _run(stepper, state):
    stepper.prefill([state])
    while not stepper.complete(state):
        stepper.advance([state])
    return stepper.finalize(state)


@pytest.mark.parametrize('length', [31, 32, 33])
@pytest.mark.parametrize('horizon', [127, 128, 129, 513, 1024])
def test_stepper_matches_whole_forecast_across_patch_and_block_boundaries(runtime, length, horizon):
    import torch
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    context = _context((length, length), offset=2.0 if length == 32 else -1.0)
    context[0].fill_(3.0)  # Constant and nonconstant histories share a call.
    expected = runtime.forecast(context, horizon)
    stepper = TimesFMStepper(runtime)
    state = stepper.initialize(context, horizon)
    assert state.blocks_done == state.cache_position == 0
    assert state.bucket == ('prefill', (length + 31) // 32 * 32)
    actual = _run(stepper, state)
    assert state.blocks_done == (horizon + 127) // 128
    assert state.cache_position == (length + 31) // 32 + 4 * (state.blocks_done - 1)
    assert actual.device.type == 'cuda' and actual.dtype == torch.float32
    assert state.cache_bytes == 0  # Finished requests need no KV state.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    stepper.release(state)
    stepper.release(state)
    assert state.released and state.output is None and state.passes == []
    assert state.cache_bytes == state.tensor_bytes == 0


def test_ragged_rows_preserve_original_batched_padding_semantics(runtime):
    import torch
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    context = _context((17, 32, 33), offset=1.0)
    expected = runtime.forecast(context, 513)
    stepper = TimesFMStepper(runtime)
    state = stepper.initialize(context, 513)
    torch.testing.assert_close(_run(stepper, state), expected, rtol=1e-4, atol=1e-4)
    stepper.release(state)


def test_late_prefill_shared_decode_reordering_and_early_retirement(runtime):
    import torch
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    cases = {'a': (_context((32, 32), -2), 513),
             'short': (_context((31,), 2), 127),
             'late': (_context((32,), -0.5), 257)}
    expected = {name: runtime.forecast(context, horizon) for name, (context, horizon) in cases.items()}
    stepper = TimesFMStepper(runtime)
    a = stepper.initialize(*cases['a'])
    short = stepper.initialize(*cases['short'])
    stepper.prefill([a, short])  # Same padded width, different raw C and H.
    assert not stepper.complete(a) and stepper.complete(short)
    assert a.cache_bytes == a.rows * a.cache_capacity * 409600
    assert a.tensor_bytes > a.cache_bytes
    torch.testing.assert_close(stepper.finalize(short), expected['short'], rtol=1e-4, atol=1e-4)
    stepper.release(short)

    late = stepper.initialize(*cases['late'])
    stepper.prefill([late])  # A stays resident while the later request starts.
    assert a.bucket == late.bucket
    assert stepper.model_call_work([a, late]) == 2 * 3 * max(a.cache_capacity, late.cache_capacity) ** 2
    stepper.advance([late, a])
    assert a.blocks_done == late.blocks_done == 2
    assert a.passes[0].caches[0].key.untyped_storage().data_ptr() != late.passes[0].caches[0].key.untyped_storage().data_ptr()
    assert a.passes[1].caches[0].key.untyped_storage().data_ptr() != late.passes[1].caches[0].key.untyped_storage().data_ptr()
    stepper.advance([a, late])
    assert not stepper.complete(a) and stepper.complete(late)
    torch.testing.assert_close(stepper.finalize(late), expected['late'], rtol=1e-4, atol=1e-4)
    stepper.release(late)
    while not stepper.complete(a):
        stepper.advance([a])
    torch.testing.assert_close(stepper.finalize(a), expected['a'], rtol=1e-4, atol=1e-4)
    stepper.release(a)
    # Reuse after a different cohort cannot carry normalization/cache state.
    again = stepper.initialize(*cases['a'])
    torch.testing.assert_close(_run(stepper, again), expected['a'], rtol=1e-4, atol=1e-4)
    stepper.release(again)


def test_normal_and_flipped_streams_use_their_own_feedback_statistics(runtime):
    import torch
    from timesfm.torch.util import update_running_stats
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    stepper = TimesFMStepper(runtime)
    state = stepper.initialize(_context((33, 33), -0.3), 257)
    stepper.prefill([state])
    normal, flipped = state.passes
    assert normal.feedback.untyped_storage().data_ptr() != flipped.feedback.untyped_storage().data_ptr()
    # A singleton keeps the stacked [x; -x] prefill allocation: the two flip
    # streams may share storage but must own disjoint row ranges of it.
    for mine, other in zip(normal.caches, flipped.caches, strict=True):
        for a, b in ((mine.key, other.key), (mine.value, other.value)):
            a_end = a.data_ptr() + a.numel() * a.element_size()
            b_end = b.data_ptr() + b.numel() * b.element_size()
            assert a_end <= b.data_ptr() or b_end <= a.data_ptr()
    expected = []
    for stream in state.passes:
        n, mu, sigma = stream.last_n, stream.last_mu, stream.last_sigma
        feedback = stream.feedback.reshape(state.rows, 4, 32)
        mask = torch.zeros_like(feedback, dtype=torch.bool)
        for patch in range(4):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, feedback[:, patch], mask[:, patch])
        expected.append((n, mu, sigma))
    stepper.advance([state])
    for stream, stats in zip(state.passes, expected, strict=True):
        for actual, reference in zip((stream.last_n, stream.last_mu, stream.last_sigma), stats, strict=True):
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    stepper.release(state)


def test_different_initial_contexts_can_merge_at_the_same_cache_position(runtime):
    import torch
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    long_context = _context((160, 160), -1.5)
    short_context = _context((32,), 0.5)
    expected_long = runtime.forecast(long_context, 513)
    expected_short = runtime.forecast(short_context, 513)
    stepper = TimesFMStepper(runtime)
    a = stepper.initialize(long_context, 513)
    b = stepper.initialize(short_context, 513)
    stepper.prefill([a])
    stepper.prefill([b])
    stepper.advance([b])
    assert (a.blocks_done, b.blocks_done) == (1, 2)
    assert a.cache_position == b.cache_position == 5
    assert a.bucket == b.bucket == ('decode',)
    stepper.advance([b, a])
    assert (a.blocks_done, b.blocks_done) == (2, 3)
    # Different progress counters and cache capacities retain their own meaning
    # even though the physical write index permits this shared model call.
    assert a.cache_capacity == 21 and b.cache_capacity == 17
    for state, expected in ((a, expected_long), (b, expected_short)):
        while not stepper.complete(state):
            stepper.advance([state])
        torch.testing.assert_close(stepper.finalize(state), expected, rtol=1e-4, atol=1e-4)
        stepper.release(state)
        assert state.tensor_bytes == 0


def test_incompatible_or_duplicate_cohorts_fail_before_mutating_progress(runtime):
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    stepper = TimesFMStepper(runtime)
    a = stepper.initialize(_context((32,)), 513)
    b = stepper.initialize(_context((33,)), 513)
    with pytest.raises(ValueError, match='buckets'):
        stepper.prefill([a, b])
    with pytest.raises(ValueError, match='unique'):
        stepper.prefill([a, a])
    assert a.blocks_done == b.blocks_done == 0
    stepper.prefill([a])
    with pytest.raises(ValueError, match='buckets'):
        stepper.advance([a, b])
    assert a.blocks_done == 1 and b.blocks_done == 0
    with pytest.raises(ValueError, match='completed'):
        stepper.finalize(a)
    stepper.release(a)
    stepper.release(b)
    assert a.cache_bytes == b.cache_bytes == 0
    assert all(getattr(a, field) is None for field in ('normalized', 'padding', 'mean', 'std', 'positive', 'output'))
    with pytest.raises(ValueError, match='released'):
        stepper.advance([a])


def test_staggered_physical_positions_share_decode_and_keep_owned_cache_storage(runtime):
    import torch
    from test_timesfm_block_state import assert_append_only, snapshot
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    cases = {'early': (_context((480, 512), -1.5), 1024),
             'late': (_context((512,), 0.5), 513)}
    expected = {name: runtime.forecast(values, horizon)
                for name, (values, horizon) in cases.items()}
    stepper = TimesFMStepper(runtime)
    early = stepper.initialize(*cases['early'])
    stepper.prefill([early])
    before, position = snapshot(early), early.cache_position
    stepper.advance([early])
    assert_append_only(early, before, position)
    late = stepper.initialize(*cases['late'])
    stepper.prefill([late])
    assert (early.cache_position, late.cache_position) == (20, 16)
    assert early.bucket == late.bucket == ('decode',)
    before = {id(state): (snapshot(state), state.cache_position) for state in (early, late)}
    stepper.advance([late, early])
    for state in (early, late):
        assert_append_only(state, *before[id(state)])
    active = [early, late]
    while active:
        stepper.advance(list(reversed(active)))
        for name, state in [('early', early), ('late', late)]:
            if any(value is state for value in active) and stepper.complete(state):
                torch.testing.assert_close(stepper.finalize(state), expected[name], rtol=1e-4, atol=1e-4)
                stepper.release(state)
                active = [value for value in active if value is not state]
    assert early.tensor_bytes == late.tensor_bytes == 0
    # Whole forecasting still calls the unchanged upstream forward after use.
    values, horizon = cases['late']
    torch.testing.assert_close(runtime.forecast(values, horizon), expected['late'], rtol=0, atol=0)


def test_maximum_capacity_layouts_match_independent_official_native(runtime, tmp_path, monkeypatch):
    """Full-corner native parity, including the 32-row internal AR batch."""
    import json
    import torch
    from model_integration_support import NativeForecaster
    from vtsfm.adapters.timesfm25 import block_decode
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    lengths = tuple(8192 - 17 * (row % 4) for row in range(16))
    context = _context(lengths, offset=-0.5)
    cpu_context = context.cpu()
    payload = {'inputs': [
        {'item_id': f'full-corner-{row}', 'target': values[torch.isfinite(values)].tolist()}
        for row, values in enumerate(cpu_context)],
        'parameters': {'prediction_length': 1024, 'quantile_levels': [i / 10 for i in range(1, 10)]}}
    assert torch.get_float32_matmul_precision() == 'highest'
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    native = NativeForecaster('timesfm25')  # Separate official model, torch_compile=False.
    expected = torch.from_numpy(native.predict_array(payload)).clone()
    metadata = native.metadata()
    assert expected.shape == (16, 9, 1024) and expected.dtype == torch.float32
    torch.save({'context': cpu_context, 'native': expected}, tmp_path / 'native-golden.pt')
    del native
    gc.collect()
    torch.cuda.empty_cache()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    control = runtime.forecast(context, 1024).cpu()
    torch.save(control, tmp_path / 'runtime-control.pt')
    evidence = {'native': metadata, 'lengths': lengths, 'rtol': 1e-4, 'atol': 1e-4,
                'runtime_max_abs': float((control - expected).abs().max()), 'layouts': []}
    report = tmp_path / 'maximum-native-parity.json'
    report.write_text(json.dumps(evidence, indent=2) + '\n')
    torch.testing.assert_close(control, expected, rtol=1e-4, atol=1e-4)
    ar_rows, original = [], block_decode.block_decode_forward

    def observe(model, inputs, masks, cache_groups, positions):
        ar_rows.append(inputs.shape[0])
        return original(model, inputs, masks, cache_groups, positions)

    monkeypatch.setattr(block_decode, 'block_decode_forward', observe)
    for layout in ((16,), (4, 4, 4, 4)):
        stepper, states, cursor = TimesFMStepper(runtime), [], 0
        ar_rows.clear()
        for rows in layout:
            states.append(stepper.initialize(context[cursor:cursor + rows], 1024))
            cursor += rows
        assert cursor == 16
        try:
            stepper.prefill(states)
            retained = sum(state.cache_bytes for state in states)
            assert retained == 1_861_222_400
            while not stepper.complete(states[0]):
                stepper.advance(states)
            actual = torch.cat([stepper.finalize(state) for state in states]).cpu()
            label = '-'.join(map(str, layout))
            torch.save(actual, tmp_path / f'layout-{label}-actual.pt')
            difference = (actual - expected).abs()
            evidence['layouts'].append({'owner_rows': layout, 'persistent_KV_bytes': retained,
                'AR_neural_rows': list(ar_rows), 'max_abs': float(difference.max()),
                'max_abs_by_quantile': difference.amax(dim=(0, 2)).tolist()})
            report.write_text(json.dumps(evidence, indent=2) + '\n')
            assert ar_rows == [32] * 7
            assert all(state.cache_bytes == 0 for state in states)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        finally:
            for state in states:
                stepper.release(state)
            assert all(state.tensor_bytes == 0 for state in states)
    evidence['status'] = 'passed'
    report.write_text(json.dumps(evidence, indent=2) + '\n')
