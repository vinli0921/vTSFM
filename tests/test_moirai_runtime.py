"""Moirai 2.0 tensor-path tests (issue B3), written ahead of the implementation.

CPU tests check declared capabilities and run in the normal CPU environment.
GPU tests load the pinned checkpoint and compare with official uni2ts:

    VTSFM_TEST_ADAPTER=moirai2 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:tests \\
      .venv-moirai2/bin/python -m pytest -m gpu tests/test_moirai_runtime.py -q -s

Pinned upstream behavior (uni2ts cfd46d4, ``model/moirai2/forecast.py``):

- ``Moirai2Forecast.forward`` takes ``past_target`` [b, t, 1], ``past_observed_target``
  [b, t, 1] and ``past_is_pad`` [b, t]. It owns patching, time/sample IDs, the
  recursive quantile loop (9 trajectories fed back, then ``torch.quantile`` over
  9x9 values), and output cropping to ``prediction_length``.
- ``predict`` only converts NumPy lists: causal-mean imputation for NaN, then
  left padding to ``context_length`` by repeating each history's first value
  and first observed flag (True), with only ``past_is_pad`` marking the padding.
- ``forward`` uses ``past_is_pad`` at whole-patch granularity (``sample_id``).
  In a partially padded patch, the repeated values therefore count as observed:
  they enter ``PackedStdScaler`` loc/scale and the input tokens. A history
  forecast alone instead gets zero values with observed False from
  ``_patched_seq_pad``. That is the known ragged-batch discrepancy.

Reference semantics in these tests: a row's correct forecast is official
``predict`` on that history alone with ``context_length=len(history)``. For
equal-length batches that equals batched ``predict``. For ragged batches, these
tests compare against isolated forecasts only; one test documents that batched
``predict`` differs, and nothing requires reproducing it.

Assumed MoiraiRuntime behavior; update these tests if the design changes:

- ``forecast(context, horizon)`` is the live-serving path. ``context`` is FP32
  [rows, width] with left NaN padding; it returns [rows, 9, horizon] in the
  checkpoint quantile order on the model device, without host copies.
- ``execute(batch: PreparedBatch) -> ExecutionResult`` reads ``observed_mask``
  and ``context_lengths`` (never NaN detection), returns only the key's
  quantiles in key order, stays on the model device, and rejects another
  model/revision, grouped rows, and inputs that are not on the model device.
- ``spec.py`` declares ``input_patch_size=16`` and ``output_block_size=64``
  (patch_size * num_predict_token in config.json at MODEL_REVISION).
"""

from contextlib import contextmanager
import math
import os
from types import SimpleNamespace

import pytest

from vtsfm.adapters import get_adapter


ADAPTER = os.environ.get('VTSFM_TEST_ADAPTER')
requires_moirai = pytest.mark.skipif(
    ADAPTER != 'moirai2', reason='Set VTSFM_TEST_ADAPTER=moirai2 in the Moirai GPU environment')
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RTOL = ATOL = 1e-4  # Declared FP32 parity tolerance from the issue.
# config.json at MODEL_REVISION: patch_size 16, num_predict_token 4, max_seq_len 512.
PATCH, BLOCK, MAX_TOKENS = 16, 64, 512


def signal(length, phase=0.0):
    """Deterministic synthetic history; not an accuracy workload."""
    return [2.0 + phase + 0.01 * step + math.sin(step / 7 + phase) + 0.2 * math.cos(step / 13)
            for step in range(length)]


# ---------------------------------------------------------------------------
# CPU: declared capabilities (no Torch or model import)
# ---------------------------------------------------------------------------

def test_capabilities_declare_checkpoint_patch_and_block():
    capabilities = get_adapter('moirai2').execution_capabilities
    assert capabilities.input_patch_size == PATCH
    assert capabilities.output_block_size == BLOCK
    assert capabilities.quantiles == QUANTILES
    assert not capabilities.supports_point_forecast
    # Multi-row groups would be multivariate Moirai input, not independent series.
    assert not capabilities.supports_grouped_rows
    # Context and prediction tokens share the checkpoint's rotary range.
    tokens = -(-capabilities.max_context // PATCH) + -(-capabilities.max_horizon // PATCH)
    assert tokens <= MAX_TOKENS


@pytest.mark.parametrize('horizon, blocks', [
    (1, 1), (63, 1), (64, 1), (65, 2), (128, 2), (129, 3), (1024, 16),
])
def test_workload_counts_recursive_blocks_and_input_patches(horizon, blocks):
    from vtsfm.batching import BatchKey, BatchRequest

    spec = get_adapter('moirai2')
    capabilities = spec.execution_capabilities
    key = BatchKey(spec.name, spec.model_id, spec.model_revision, 100, horizon, (0.5,))
    work = capabilities.workload(BatchRequest('workload', key, (1, 1, 1)))
    assert work.rows == 3
    # Each block after the first is one recursive pass over 9x expanded rows.
    assert work.forecast_blocks == blocks
    assert work.input_patches == 3 * -(-100 // PATCH)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def moirai():
    """Adapter runtime and an independent official module with identical weights."""
    import json
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from model_integration_support import NativeForecaster

    from vtsfm.adapters.moirai2.runtime import MoiraiRuntime

    native = NativeForecaster('moirai2')
    spec = get_adapter('moirai2')
    directory = snapshot_download(spec.model_id, revision=spec.model_revision,
                                  allow_patterns=['config.json', 'model.safetensors'])
    # The serving wrapper passes the raw config.json as vtsfm_config.
    config = json.loads((Path(directory) / 'config.json').read_text())
    runtime = MoiraiRuntime(config)
    runtime.model.load_state_dict(native.reference.state_dict())
    runtime.model.to(device=native.device, dtype=torch.float32).eval()
    module = native.reference
    return SimpleNamespace(runtime=runtime, reference=module, forecast_class=native.forecast_class,
                           device=native.device, patch=module.patch_size,
                           block=module.patch_size * module.num_predict_token)


def official(moirai, histories, horizon):
    """Official ``predict`` with rows batched together: [rows, 9, horizon] on device.

    ``context_length`` is the longest history, as in the bring-up wrapper. NaN
    entries are missing observations (upstream imputes them and masks them).
    """
    import numpy as np
    import torch

    runner = moirai.forecast_class(
        module=moirai.reference, prediction_length=horizon,
        context_length=max(map(len, histories)), target_dim=1,
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(moirai.device).eval()
    values = runner.predict([np.asarray(history, dtype=np.float32) for history in histories])
    return torch.from_numpy(values).to(moirai.device)


def isolated(moirai, histories, horizon):
    """Official ``predict`` for each row alone; the correct ragged reference."""
    import torch

    return torch.cat([official(moirai, [history], horizon) for history in histories])


def nan_padded(moirai, histories):
    """The live-serving forecast() input: [rows, longest] with left NaN padding."""
    import torch

    context = torch.full((len(histories), max(map(len, histories))), float('nan'), device=moirai.device)
    for row, history in zip(context, histories):
        row[-len(history):] = torch.tensor(history, dtype=torch.float32, device=moirai.device)
    return context


def prepare(moirai, request_id, histories, horizon, levels=QUANTILES, *, width=None,
            fill=-1000.0, device=None, capabilities=None, group_sizes=None, **key_changes):
    """Build a PreparedRequest. Padding and missing slots hold ``fill``, not NaN
    or zero, so a runtime that infers the mask from values, or lets unobserved
    values reach the scaler or input tokens, produces wrong forecasts."""
    from dataclasses import replace

    import torch

    from vtsfm.adapters.moirai2.spec import MOIRAI2
    from vtsfm.batching import BatchKey, BatchRequest
    from vtsfm.execution import PreparedRequest

    capabilities = capabilities or MOIRAI2.execution_capabilities
    device = device or moirai.device
    rows, width = len(histories), width or max(map(len, histories))
    key = BatchKey(MOIRAI2.name, MOIRAI2.model_id, MOIRAI2.model_revision, width, horizon, tuple(levels),
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


def execute(moirai, *requests):
    return moirai.runtime.execute(plan(*requests))


def channels(levels):
    return [QUANTILES.index(level) for level in levels]


@contextmanager
def no_host_copies(monkeypatch):
    """Fail if a forecast moves tensors to the host (GPU→NumPy→GPU round trips).

    CPU index tensors that upstream builds with torch.arange are host→device and
    stay allowed; only device→host conversions are rejected.
    """
    import torch

    def fail(*args, **kwargs):
        pytest.fail('Moirai forecast moved a tensor to the host')

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


def assert_forecast(actual, expected, moirai):
    import torch

    assert actual.device == moirai.device
    assert actual.dtype == torch.float32
    assert tuple(actual.shape) == tuple(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# GPU: capabilities against the loaded checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_moirai
def test_declared_capabilities_match_loaded_checkpoint(moirai):
    capabilities = get_adapter('moirai2').execution_capabilities
    module = moirai.runtime.model
    assert capabilities.input_patch_size == module.patch_size == PATCH
    assert capabilities.output_block_size == module.patch_size * module.num_predict_token == BLOCK
    assert tuple(module.quantile_levels) == capabilities.quantiles
    tokens = -(-capabilities.max_context // module.patch_size) + -(-capabilities.max_horizon // module.patch_size)
    assert tokens <= module.max_seq_len


# ---------------------------------------------------------------------------
# GPU: forecast(), the live-serving path
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_moirai
@pytest.mark.parametrize('length', [1, 15, 16, 17, 31, 32, 33, 128])
def test_forecast_matches_official_at_patch_boundaries(moirai, length):
    """Equal-length rows around 16-point patch boundaries, one output block."""
    histories = [signal(length), signal(length, 1.0)]
    actual = moirai.runtime.forecast(nan_padded(moirai, histories), BLOCK)
    assert_forecast(actual, official(moirai, histories, BLOCK), moirai)


@pytest.mark.gpu
@requires_moirai
@pytest.mark.parametrize('horizon', [1, 16, 17, 63, 64, 65, 128, 129, 200])
def test_forecast_matches_official_at_block_boundaries_and_recursion(moirai, horizon):
    """Horizons around the 64-point output block. Above one block, upstream
    recurses on all 9 quantile trajectories; median-only feedback fails here.
    Non-multiples of 16 also check cropping to the requested horizon."""
    histories = [signal(128), signal(128, 1.0), signal(128, 2.0)]
    actual = moirai.runtime.forecast(nan_padded(moirai, histories), horizon)
    assert_forecast(actual, official(moirai, histories, horizon), moirai)


@pytest.mark.gpu
@requires_moirai
def test_forecast_matches_official_at_declared_limits(moirai):
    capabilities = get_adapter('moirai2').execution_capabilities
    histories = [signal(capabilities.max_context), signal(capabilities.max_context, 1.0)]
    actual = moirai.runtime.forecast(nan_padded(moirai, histories), capabilities.max_horizon)
    assert_forecast(actual, official(moirai, histories, capabilities.max_horizon), moirai)


RAGGED_CASES = {
    # The issue's case: one partially padded patch (15 artificial slots).
    '17-with-32-h96': ((17, 32), 96),
    '17-with-32-h64': ((17, 32), 64),
    # Fully padded patches plus a partially padded one, with recursion.
    'mixed-h129': ((1, 16, 33, 100), 129),
    '15-with-200-h65': ((15, 200), 65),
}


@pytest.mark.gpu
@requires_moirai
@pytest.mark.parametrize('case', RAGGED_CASES, ids=list(RAGGED_CASES))
def test_forecast_ragged_rows_match_isolated_official(moirai, case):
    """Ragged rows within one HTTP request must forecast as if each were alone.

    The bring-up wrapper (batched NumPy predict) fails this; see the next test.
    """
    lengths, horizon = RAGGED_CASES[case]
    histories = [signal(length, phase=index / 3) for index, length in enumerate(lengths)]
    actual = moirai.runtime.forecast(nan_padded(moirai, histories), horizon)
    assert_forecast(actual, isolated(moirai, histories, horizon), moirai)


@pytest.mark.gpu
@requires_moirai
def test_official_batched_predict_has_ragged_padding_discrepancy(moirai):
    """Documents the upstream discrepancy; the runtime is NOT compared with it.

    Batched predict pads the 17-point history to 32 by repeating its first value
    as observed, so the partially padded patch changes normalization. The
    unpadded 32-point row is unaffected. If a uni2ts pin bump fixes this, this
    test fails and the module docstring should be updated.
    """
    import torch

    histories = [signal(17), signal(32, 1.0)]
    batched = official(moirai, histories, 96)
    alone = isolated(moirai, histories, 96)
    difference = (batched[0] - alone[0]).abs().max().item()
    print(f'17-point row, batched vs isolated official: max abs difference {difference:.3g}')
    assert not torch.allclose(batched[0], alone[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(batched[1], alone[1], rtol=RTOL, atol=ATOL)


@pytest.mark.gpu
@requires_moirai
def test_forecast_a_b_a_has_no_stale_state(moirai):
    """A runtime that caches a Moirai2Forecast must not keep B's context or
    horizon hyperparameters for the repeated A."""
    a = [signal(128), signal(128, 0.5)]
    b = [signal(17), signal(32, 1.0)]
    c = [signal(33), signal(33, 1.0), signal(33, 2.0)]
    first = moirai.runtime.forecast(nan_padded(moirai, a), 160)
    moirai.runtime.forecast(nan_padded(moirai, b), 65)
    moirai.runtime.forecast(nan_padded(moirai, c), 1)
    again = moirai.runtime.forecast(nan_padded(moirai, a), 160)
    assert_forecast(again, first, moirai)
    assert_forecast(again, official(moirai, a, 160), moirai)


@pytest.mark.gpu
@requires_moirai
def test_forecast_stays_on_device_and_keeps_input(moirai, monkeypatch):
    """Ragged and recursive, so padding, masks and recursion all run guarded."""
    import torch

    histories = [signal(17), signal(32, 1.0)]
    context = nan_padded(moirai, histories)
    before = context.clone()
    expected = isolated(moirai, histories, 129)
    with no_host_copies(monkeypatch):
        actual = moirai.runtime.forecast(context, 129)
    assert_forecast(actual, expected, moirai)
    # The caller owns the transported context; it must still hold NaN padding.
    torch.testing.assert_close(context, before, rtol=0, atol=0, equal_nan=True)


# ---------------------------------------------------------------------------
# GPU: execute(), the prepared-batch contract
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_moirai
def test_execute_matches_official_and_splits_requests(moirai):
    levels = (0.2, 0.8)
    series = {'first': [signal(128), signal(128, 0.5)], 'second': [signal(128, 2.0)]}
    requests = [prepare(moirai, request_id, histories, 129, levels)
                for request_id, histories in series.items()]
    result = execute(moirai, *requests).split()

    assert tuple(result) == ('first', 'second')
    for request in requests:
        request_id = request.metadata.request_id
        forecast = result[request_id]
        assert forecast.item_ids == request.item_ids
        assert forecast.quantile_levels == levels
        expected = isolated(moirai, series[request_id], 129)[:, channels(levels)]
        assert_forecast(forecast.values, expected, moirai)


@pytest.mark.gpu
@requires_moirai
@pytest.mark.parametrize('levels', [(0.5,), (0.1, 0.5, 0.9), (0.3, 0.7), QUANTILES],
                         ids=['median', 'three', 'inner', 'all'])
def test_execute_returns_requested_quantiles_in_key_order(moirai, levels):
    histories = [signal(128), signal(128, 1.0)]
    result = execute(moirai, prepare(moirai, 'only', histories, 65, levels))
    assert tuple(result.values.shape) == (2, len(levels), 65)
    assert_forecast(result.values, isolated(moirai, histories, 65)[:, channels(levels)], moirai)


@pytest.mark.gpu
@requires_moirai
@pytest.mark.parametrize('length, horizon', [
    (15, 64), (16, 63), (17, 65), (32, 128), (33, 129), (300, 17),
])
def test_execute_matches_official_at_patch_and_block_boundaries(moirai, length, horizon):
    histories = [signal(length), signal(length, 1.0)]
    result = execute(moirai, prepare(moirai, 'boundary', histories, horizon))
    assert_forecast(result.values, isolated(moirai, histories, horizon), moirai)


@pytest.mark.gpu
@requires_moirai
@pytest.mark.parametrize('fill', [-1000.0, 1e6, float('nan')], ids=['negative', 'huge', 'nan'])
@pytest.mark.parametrize('length, extra', [(17, 15), (128, 37)], ids=['partial-patch', 'extra-patches'])
def test_execute_crops_padding_and_ignores_padding_values(moirai, fill, length, extra):
    """A prepared width wider than every history, and garbage padding values,
    must not change the forecast. NaN padding reaching the scaler would poison
    loc/scale because NaN * False is NaN."""
    histories = [signal(length), signal(length, 1.0)]
    request = prepare(moirai, 'padded', histories, 96, width=length + extra, fill=fill)
    assert_forecast(execute(moirai, request).values, isolated(moirai, histories, 96), moirai)


@pytest.mark.gpu
@requires_moirai
def test_execute_missing_observations_are_rejected_or_match_official(moirai):
    """Interior gaps are not reachable over HTTP. Upstream predict replaces them
    with a causal mean before masking, and that value reaches the input tokens.
    Either reproduce that or reject gaps explicitly; never silently differ."""
    gapped = signal(150)
    gapped[40:55] = [float('nan')] * 15
    tail = signal(150, 1.0)
    tail[-3] = float('nan')
    histories = [gapped, tail]
    request = prepare(moirai, 'gaps', histories, 96)
    try:
        result = execute(moirai, request)
    except ValueError as error:
        print(f'interior gaps rejected: {error}')
        return
    assert_forecast(result.values, isolated(moirai, histories, 96), moirai)


@pytest.mark.gpu
@requires_moirai
def test_execute_keeps_series_and_requests_independent(moirai):
    """A row's forecast must not depend on which other rows share its batch."""
    x = [signal(128), signal(128, 0.5)]
    y = [signal(128, 2.0)]
    y_changed = [[value * -50 + 1e3 for value in signal(128, 4.0)]]
    alone = execute(moirai, prepare(moirai, 'x', x, 129)).values
    with_y = execute(moirai, prepare(moirai, 'x', x, 129), prepare(moirai, 'y', y, 129)).split()
    with_changed = execute(moirai, prepare(moirai, 'x', x, 129),
                           prepare(moirai, 'y', y_changed, 129)).split()

    expected = isolated(moirai, x, 129)
    for values in (alone, with_y['x'].values, with_changed['x'].values):
        assert_forecast(values, expected, moirai)
    assert_forecast(with_y['y'].values, isolated(moirai, y, 129), moirai)


@pytest.mark.gpu
@requires_moirai
def test_execute_a_b_a_has_no_stale_state(moirai):
    a = prepare(moirai, 'a', [signal(128), signal(128, 0.5)], 160, (0.1, 0.5, 0.9))
    same_shape = prepare(moirai, 'b', [signal(128, 2.0), signal(128, 3.0)], 160, (0.1, 0.5, 0.9))
    different = prepare(moirai, 'c', [signal(33), signal(33, 1.0), signal(33, 2.0)], 17, (0.5,))
    first = execute(moirai, a).values
    execute(moirai, same_shape)
    execute(moirai, different)
    assert_forecast(execute(moirai, a).values, first, moirai)


@pytest.mark.gpu
@requires_moirai
def test_execute_stays_on_device(moirai, monkeypatch):
    # PreparedRequest validation reads the mask on the host; build it first.
    batch = plan(prepare(moirai, 'device', [signal(128), signal(128, 0.5)], 129, (0.2, 0.8),
                         width=150))
    with no_host_copies(monkeypatch):
        result = moirai.runtime.execute(batch)
    assert result.values.device == moirai.device


@pytest.mark.gpu
@requires_moirai
def test_execute_ragged_rows_follow_the_declared_capability(moirai):
    """Ragged prepared batches stay rejected until this regression passes.

    With supports_ragged_batching False, the shared contract must reject them.
    Set it True only once both branches below pass: ragged rows within one
    request and across requests must match isolated official forecasts.
    """
    capabilities = get_adapter('moirai2').execution_capabilities
    short, long = [signal(17)], [signal(32, 1.0)]
    if not capabilities.supports_ragged_batching:
        with pytest.raises(ValueError, match='ragged'):
            prepare(moirai, 'both', short + long, 96)
        with pytest.raises(ValueError, match='ragged'):
            plan(prepare(moirai, 'short', short, 96, width=32), prepare(moirai, 'long', long, 96))
        return

    together = execute(moirai, prepare(moirai, 'both', short + long, 96)).values
    assert_forecast(together, isolated(moirai, short + long, 96), moirai)
    split = execute(moirai, prepare(moirai, 'short', short, 96, width=32),
                    prepare(moirai, 'long', long, 96)).split()
    assert_forecast(split['short'].values, isolated(moirai, short, 96), moirai)
    assert_forecast(split['long'].values, isolated(moirai, long, 96), moirai)


@pytest.mark.gpu
@requires_moirai
def test_execute_rejects_incompatible_batches(moirai):
    from dataclasses import replace

    from vtsfm.adapters.moirai2.spec import MOIRAI2

    histories = [signal(128), signal(128, 1.0)]
    grouped = replace(MOIRAI2.execution_capabilities, supports_grouped_rows=True)
    invalid = {
        'other adapter': prepare(moirai, 'adapter', histories, 17, adapter='timesfm25'),
        'other revision': prepare(moirai, 'revision', histories, 17, model_revision='0' * 40),
        'host inputs': prepare(moirai, 'host', histories, 17, device='cpu'),
        # Two rows in one group would be one multivariate Moirai series.
        'grouped rows': prepare(moirai, 'grouped', histories, 17, capabilities=grouped, group_sizes=(2,)),
    }
    for name, request in invalid.items():
        with pytest.raises(ValueError):
            execute(moirai, request)
        print(f'rejected: {name}')
