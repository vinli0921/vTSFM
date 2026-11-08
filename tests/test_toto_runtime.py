"""Toto 2 execution-contract tests (issue B1), written ahead of the implementation.

CPU tests check declared capabilities and run in the normal CPU environment.
GPU tests load the pinned checkpoint and compare with official Toto:

    VTSFM_TEST_ADAPTER=toto2 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:tests \\
      .venv-toto2/bin/python -m pytest -m gpu tests/test_toto_runtime.py -q -s

Assumed TotoRuntime behavior; update these tests if the design changes:

- ``forecast(context, horizon)`` is the unchanged live-serving path. ``context``
  is FP32 ``[rows, width]`` with left NaN padding; the result is
  ``[rows, 9, horizon]`` on the model device.
- ``execute(batch: PreparedBatch) -> ExecutionResult`` reads ``observed_mask``
  and ``context_lengths`` (never NaN detection), replaces unobserved values with
  zero as official ``nan_to_num`` inputs do, keeps every row an independent
  batch element, returns only the key's quantiles in key order, and stays on the
  model device. It rejects another model/revision, different capabilities
  (including grouped rows, which would enable variate attention), and inputs
  that are not on the model device.
- ``spec.py`` declares ``input_patch_size``; a GPU test checks the checkpoint.

The official reference calls ``Toto2Model.forecast`` with each row as its own
batch element (``series_ids`` zero, one variate), left padding to a patch
multiple, and ``decode_block_size=768``.
"""

from contextlib import contextmanager
import math
import os
from types import SimpleNamespace

import pytest

from vtsfm.adapters import get_adapter


ADAPTER = os.environ.get('VTSFM_TEST_ADAPTER')
requires_toto = pytest.mark.skipif(
    ADAPTER != 'toto2', reason='Set VTSFM_TEST_ADAPTER=toto2 in the Toto GPU environment')
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RTOL = ATOL = 1e-4  # Declared FP32 parity tolerance from the issue.


def signal(length, phase=0.0):
    """Deterministic synthetic history; not an accuracy workload."""
    return [2.0 + phase + 0.01 * step + math.sin(step / 7 + phase) + 0.2 * math.cos(step / 13)
            for step in range(length)]


# ---------------------------------------------------------------------------
# CPU: declared capabilities (no Torch or model import)
# ---------------------------------------------------------------------------

def test_capabilities_declare_patch_size_and_block_policy():
    capabilities = get_adapter('toto2').execution_capabilities
    assert capabilities.execution_options == (('decode_block_size', '768'),)
    assert capabilities.output_block_size == 768
    assert capabilities.input_patch_size is not None, (
        'Declare the checkpoint patch size; the GPU test verifies it against the loaded config')
    # Upstream asserts decode_block_size is divisible by patch_size.
    assert 768 % capabilities.input_patch_size == 0
    # Grouped rows would let Toto's variate attention mix independent series.
    assert not capabilities.supports_grouped_rows


@pytest.mark.parametrize('horizon, blocks', [(17, 1), (767, 1), (768, 1), (769, 2), (800, 2), (1024, 2)])
def test_workload_counts_decode_blocks_and_input_patches(horizon, blocks):
    from vtsfm.batching import BatchKey, BatchRequest

    spec = get_adapter('toto2')
    capabilities = spec.execution_capabilities
    key = BatchKey(spec.name, spec.model_id, spec.model_revision, 100, horizon, (0.5,),
                   execution_options=capabilities.execution_options)
    work = capabilities.workload(BatchRequest('workload', key, (1, 1, 1)))
    assert work.rows == 3
    assert work.forecast_blocks == blocks
    patch = capabilities.input_patch_size
    assert patch is not None
    assert work.input_patches == 3 * -(-100 // patch)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def toto():
    """Adapter runtime and an independent official model with identical weights."""
    import dataclasses

    import torch
    from model_integration_support import NativeForecaster

    from vtsfm.adapters.toto2.runtime import TotoRuntime

    native = NativeForecaster('toto2')
    config = native.reference.config
    runtime = TotoRuntime({field.name: getattr(config, field.name)
                           for field in dataclasses.fields(config) if field.init})
    runtime.model.load_state_dict(native.reference.state_dict())
    runtime.model.to(device=native.device, dtype=torch.float32).eval()
    return SimpleNamespace(runtime=runtime, reference=native.reference,
                           device=native.device, patch=config.patch_size)


def official(toto, histories, horizon):
    """Official Toto for rows padded together: [rows, 9, horizon].

    NaN entries in a history are missing observations: mask False, value zero.
    """
    import torch

    rows, width = len(histories), max(map(len, histories))
    width += -width % toto.patch
    target = torch.zeros((rows, 1, width), device=toto.device)
    mask = torch.zeros((rows, 1, width), dtype=torch.bool, device=toto.device)
    for index, history in enumerate(histories):
        values = torch.tensor(history, dtype=torch.float32, device=toto.device)
        target[index, 0, -len(history):] = torch.nan_to_num(values)
        mask[index, 0, -len(history):] = torch.isfinite(values)
    series_ids = torch.zeros((rows, 1), dtype=torch.long, device=toto.device)
    with torch.inference_mode():
        quantiles = toto.reference.forecast(
            {'target': target, 'target_mask': mask, 'series_ids': series_ids},
            horizon=horizon, decode_block_size=768)
    return quantiles[:, :, 0, :].permute(1, 0, 2)


def isolated(toto, histories, horizon):
    """Official Toto for each row alone; what an unbatched request would get."""
    import torch

    return torch.cat([official(toto, [history], horizon) for history in histories])


def nan_padded(toto, histories):
    """The live-serving forecast() input: [rows, longest] with left NaN padding."""
    import torch

    context = torch.full((len(histories), max(map(len, histories))), float('nan'), device=toto.device)
    for row, history in zip(context, histories):
        row[-len(history):] = torch.tensor(history, dtype=torch.float32, device=toto.device)
    return context


def prepare(toto, request_id, histories, horizon, levels=QUANTILES, *, width=None,
            fill=-1000.0, device=None, capabilities=None, group_sizes=None, **key_changes):
    """Build a PreparedRequest. Padding and missing slots hold ``fill``, not NaN,
    so a runtime that infers the mask from values instead of reading
    observed_mask produces wrong forecasts."""
    from dataclasses import replace

    import torch

    from vtsfm.adapters.toto2.spec import TOTO2
    from vtsfm.batching import BatchKey, BatchRequest
    from vtsfm.execution import PreparedRequest

    capabilities = capabilities or TOTO2.execution_capabilities
    device = device or toto.device
    rows, width = len(histories), width or max(map(len, histories))
    key = BatchKey(TOTO2.name, TOTO2.model_id, TOTO2.model_revision, width, horizon, tuple(levels),
                   execution_options=capabilities.execution_options)
    key = replace(key, **key_changes)
    context = torch.full((rows, width), fill, device=device)
    observed = torch.zeros((rows, width), dtype=torch.bool, device=device)
    for index, history in enumerate(histories):
        values = torch.tensor(history, dtype=torch.float32, device=device)
        seen = torch.isfinite(values)
        context[index, -len(history):] = torch.where(seen, values, torch.full_like(values, fill))
        observed[index, -len(history):] = seen
    return PreparedRequest(
        BatchRequest(request_id, key, group_sizes or (1,) * rows), context,
        tuple(map(len, histories)), observed,
        tuple(f'{request_id}-{row}' for row in range(rows)), capabilities)


def plan(*requests):
    from vtsfm.batching import plan_batches
    from vtsfm.execution import PreparedBatch

    rows = sum(len(request.item_ids) for request in requests)
    batch_plan, = plan_batches([request.metadata for request in requests],
                               max_rows=rows, max_requests=len(requests))
    by_id = {request.metadata.request_id: request for request in requests}
    return PreparedBatch(batch_plan, tuple(by_id[part.request_id] for part in batch_plan.request_slices))


def execute(toto, *requests):
    return toto.runtime.execute(plan(*requests))


def channels(levels):
    return [QUANTILES.index(level) for level in levels]


@contextmanager
def no_host_copies(monkeypatch):
    """Fail if a forecast moves tensors to the host (GPU→NumPy→GPU round trips)."""
    import torch

    def fail(*args, **kwargs):
        pytest.fail('Toto forecast moved a tensor to the host')

    original_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        target = kwargs.get('device', args[0] if args else None)
        if isinstance(target, (str, torch.device)) and torch.device(target).type == 'cpu':
            fail()
        return original_to(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'cpu', fail)
        patch.setattr(torch.Tensor, 'numpy', fail)
        patch.setattr(torch.Tensor, 'to', to)
        yield


def assert_forecast(actual, expected, toto):
    import torch

    assert actual.device == toto.device
    assert actual.dtype == torch.float32
    assert tuple(actual.shape) == tuple(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


# Lengths are (patches, offset) pairs because the patch size comes from the checkpoint.
FORECAST_CASES = {
    'single-short-h17': ([(0, 128)], 17),
    'patch-boundaries-h768': ([(1, -1), (1, 0), (1, 1), (2, 0)], 768),
    'equal-one-block-h767': ([(0, 200), (0, 200)], 767),
    'ragged-two-blocks-h769': ([(0, 97), (0, 163)], 769),
    'multi-block-h800': ([(0, 128), (0, 128), (0, 128)], 800),
}


def histories_for(toto, lengths):
    return [signal(patches * toto.patch + offset, phase=index / 3)
            for index, (patches, offset) in enumerate(lengths)]


# ---------------------------------------------------------------------------
# GPU: capabilities against the loaded checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_toto
def test_declared_capabilities_match_loaded_checkpoint(toto):
    capabilities = get_adapter('toto2').execution_capabilities
    assert capabilities.input_patch_size == toto.runtime.model.config.patch_size
    assert 768 % toto.runtime.model.config.patch_size == 0


# ---------------------------------------------------------------------------
# GPU: forecast(), the live-serving path (should already pass on main)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_toto
@pytest.mark.parametrize('case', FORECAST_CASES, ids=list(FORECAST_CASES))
def test_forecast_matches_official(toto, case):
    lengths, horizon = FORECAST_CASES[case]
    histories = histories_for(toto, lengths)
    actual = toto.runtime.forecast(nan_padded(toto, histories), horizon)
    assert_forecast(actual, official(toto, histories, horizon), toto)


@pytest.mark.gpu
@requires_toto
def test_forecast_matches_official_with_missing_observations(toto):
    """NaN gaps inside a history, including in the final context patch, which
    upstream forces to observed (with value zero)."""
    middle = signal(150)
    middle[40:55] = [float('nan')] * 15
    tail = signal(150, 1.0)
    tail[-3] = float('nan')
    histories = [middle, tail]
    actual = toto.runtime.forecast(nan_padded(toto, histories), 769)
    assert_forecast(actual, official(toto, histories, 769), toto)


@pytest.mark.gpu
@requires_toto
def test_forecast_a_b_a_has_no_stale_state(toto):
    """Upstream keeps a KV cache on the model and reuses it when shapes match.
    B has A's exact shape (cache reused); C changes rows, length and horizon."""
    a = [signal(200), signal(200, 0.5)]
    b = [signal(200, 2.0), signal(200, 3.0)]
    c = [signal(97), signal(163), signal(64)]
    first = toto.runtime.forecast(nan_padded(toto, a), 800)
    toto.runtime.forecast(nan_padded(toto, b), 800)
    toto.runtime.forecast(nan_padded(toto, c), 769)
    again = toto.runtime.forecast(nan_padded(toto, a), 800)
    assert_forecast(again, first, toto)
    assert_forecast(again, official(toto, a, 800), toto)


@pytest.mark.gpu
@requires_toto
def test_forecast_stays_on_device(toto, monkeypatch):
    histories = [signal(97), signal(163)]
    context = nan_padded(toto, histories)
    expected = official(toto, histories, 800)
    with no_host_copies(monkeypatch):
        actual = toto.runtime.forecast(context, 800)
    assert_forecast(actual, expected, toto)


# ---------------------------------------------------------------------------
# GPU: execute(), the prepared-batch contract (new)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_toto
def test_execute_matches_official_and_splits_requests(toto):
    levels = (0.2, 0.8)
    series = {'first': [signal(200), signal(200, 0.5)], 'second': [signal(200, 2.0)]}
    requests = [prepare(toto, request_id, histories, 800, levels)
                for request_id, histories in series.items()]
    result = execute(toto, *requests).split()

    assert tuple(result) == ('first', 'second')
    for request in requests:
        request_id = request.metadata.request_id
        forecast = result[request_id]
        assert forecast.item_ids == request.item_ids
        assert forecast.quantile_levels == levels
        expected = isolated(toto, series[request_id], 800)[:, channels(levels)]
        assert_forecast(forecast.values, expected, toto)


@pytest.mark.gpu
@requires_toto
@pytest.mark.parametrize('levels', [(0.5,), (0.1, 0.5, 0.9), QUANTILES], ids=['median', 'three', 'all'])
def test_execute_returns_requested_quantiles_in_key_order(toto, levels):
    histories = [signal(128), signal(128, 1.0)]
    result = execute(toto, prepare(toto, 'only', histories, 33, levels))
    assert tuple(result.values.shape) == (2, len(levels), 33)
    assert_forecast(result.values, isolated(toto, histories, 33)[:, channels(levels)], toto)


@pytest.mark.gpu
@requires_toto
@pytest.mark.parametrize('length, horizon', [
    ((1, -1), 768), ((1, 0), 767), ((1, 1), 769), ((2, 0), 800), ((0, 300), 17),
], ids=['patch-1', 'patch', 'patch+1', 'two-patches', 'long-short-horizon'])
def test_execute_matches_official_at_patch_and_block_boundaries(toto, length, horizon):
    patches, offset = length
    histories = [signal(patches * toto.patch + offset), signal(patches * toto.patch + offset, 1.0)]
    result = execute(toto, prepare(toto, 'boundary', histories, horizon))
    assert_forecast(result.values, isolated(toto, histories, horizon), toto)


@pytest.mark.gpu
@requires_toto
@pytest.mark.parametrize('fill', [-1000.0, 1e6, float('nan')], ids=['negative', 'huge', 'nan'])
def test_execute_reads_mask_and_lengths_not_padding_values(toto, fill):
    """Extra left padding (width > history) and garbage padding values must not
    change the forecast; only observed_mask and context_lengths define history."""
    histories = [signal(150), signal(150, 1.0)]
    request = prepare(toto, 'padded', histories, 769, width=150 + 37, fill=fill)
    assert_forecast(execute(toto, request).values, isolated(toto, histories, 769), toto)


@pytest.mark.gpu
@requires_toto
def test_execute_matches_official_with_missing_observations(toto):
    middle = signal(150)
    middle[40:55] = [float('nan')] * 15
    tail = signal(150, 1.0)
    tail[-3] = float('nan')  # Final patch: upstream forces observed, value must be zero.
    histories = [middle, tail]
    result = execute(toto, prepare(toto, 'gaps', histories, 769))
    assert_forecast(result.values, isolated(toto, histories, 769), toto)


@pytest.mark.gpu
@requires_toto
def test_execute_keeps_series_and_requests_independent(toto):
    """A row's forecast must not depend on which other rows share its batch."""
    x = [signal(200), signal(200, 0.5)]
    y = [signal(200, 2.0)]
    y_changed = [[value * -50 + 1e3 for value in signal(200, 4.0)]]
    alone = execute(toto, prepare(toto, 'x', x, 800)).values
    with_y = execute(toto, prepare(toto, 'x', x, 800), prepare(toto, 'y', y, 800)).split()
    with_changed = execute(toto, prepare(toto, 'x', x, 800), prepare(toto, 'y', y_changed, 800)).split()

    expected = isolated(toto, x, 800)
    for values in (alone, with_y['x'].values, with_changed['x'].values):
        assert_forecast(values, expected, toto)
    assert_forecast(with_y['y'].values, isolated(toto, y, 800), toto)


@pytest.mark.gpu
@requires_toto
def test_execute_a_b_a_has_no_stale_state(toto):
    a = prepare(toto, 'a', [signal(200), signal(200, 0.5)], 800, (0.1, 0.5, 0.9))
    same_shape = prepare(toto, 'b', [signal(200, 2.0), signal(200, 3.0)], 800, (0.1, 0.5, 0.9))
    different = prepare(toto, 'c', [signal(97), signal(97, 1.0), signal(97, 2.0)], 769, (0.5,))
    first = execute(toto, a).values
    execute(toto, same_shape)
    execute(toto, different)
    assert_forecast(execute(toto, a).values, first, toto)


@pytest.mark.gpu
@requires_toto
def test_execute_stays_on_device(toto, monkeypatch):
    # PreparedRequest validation reads the mask on the host; build it first.
    batch = plan(prepare(toto, 'device', [signal(200), signal(200, 0.5)], 800, (0.2, 0.8)))
    with no_host_copies(monkeypatch):
        result = toto.runtime.execute(batch)
    assert result.values.device == toto.device


@pytest.mark.gpu
@requires_toto
def test_execute_ragged_requests_follow_the_declared_capability(toto):
    """Unequal history lengths across requests need evidence before enabling.

    The printed difference shows whether extra fully masked left patches change
    official Toto output; it is the evidence for supports_ragged_batching.
    """
    import torch

    short, long = [signal(97)], [signal(163, 1.0)]
    alone = official(toto, short, 769)
    widened = official(toto, [[float('nan')] * (163 - 97) + short[0]], 769)
    print(f'extra masked left padding: max abs difference '
          f'{(alone - widened).abs().max().item():.3g}')

    capabilities = get_adapter('toto2').execution_capabilities
    # Same padded width (one BatchKey), different true history lengths.
    requests = prepare(toto, 'short', short, 769, width=163), prepare(toto, 'long', long, 769)
    if not capabilities.supports_ragged_batching:
        with pytest.raises(ValueError, match='ragged'):
            plan(*requests)
        return
    result = execute(toto, *requests).split()
    assert_forecast(result['short'].values, isolated(toto, short, 769), toto)
    assert_forecast(result['long'].values, isolated(toto, long, 769), toto)
    torch.testing.assert_close(alone, widened, rtol=RTOL, atol=ATOL)


@pytest.mark.gpu
@requires_toto
def test_execute_rejects_incompatible_batches(toto):
    from dataclasses import replace

    from vtsfm.adapters.toto2.spec import TOTO2

    histories = [signal(128), signal(128, 1.0)]
    # The shared contract pins Toto's decoding policy before execute() runs.
    with pytest.raises(ValueError):
        prepare(toto, 'policy', histories, 17, execution_options=(('decode_block_size', '512'),))

    grouped = replace(TOTO2.execution_capabilities, supports_grouped_rows=True)
    invalid = {
        'other adapter': prepare(toto, 'adapter', histories, 17, adapter='timesfm25'),
        'other revision': prepare(toto, 'revision', histories, 17, model_revision='0' * 40),
        'host inputs': prepare(toto, 'host', histories, 17, device='cpu'),
        # Two rows in one group would share Toto's variate attention.
        'grouped rows': prepare(toto, 'grouped', histories, 17, capabilities=grouped, group_sizes=(2,)),
    }
    for name, request in invalid.items():
        with pytest.raises(ValueError):
            execute(toto, request)
        print(f'rejected: {name}')
