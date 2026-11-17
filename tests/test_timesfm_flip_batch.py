"""One stacked normal+flipped decode matches the original two-call flip invariance."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timesfm')

from test_timesfm_block_decode import tiny_model  # noqa: E402
from timesfm.torch.util import revin  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime  # noqa: E402
from vtsfm.adapters.timesfm25.stepping import TimesFMStepper  # noqa: E402


@pytest.fixture(params=[True, False], ids=['fused-qkv', 'separate-qkv'])
def runtime(request):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    actual = object.__new__(TimesFMRuntime)
    actual.model = tiny_model(request.param)
    try:
        yield actual
    finally:
        torch.set_num_threads(threads)


def context(lengths, offset=0.0):
    values = torch.full((len(lengths), max(lengths)), float('nan'))
    for row, width in enumerate(lengths):
        time = torch.arange(width, dtype=torch.float32)
        values[row, -width:] = torch.sin(time / (5 + row)) * (row + 1) + offset
    return values


def two_call_forecast(model, context, horizon):
    """The pre-#52 runtime: one native decode for x and a second one for -x."""
    observed = torch.isfinite(context)
    left = -context.shape[-1] % model.p
    values = F.pad(context.masked_fill(~observed, 0), (left, 0))
    padding = F.pad(~observed, (left, 0), value=True)
    positive = (values >= 0).all(dim=-1, keepdim=True).unsqueeze(-1)
    mean, std = values.mean(dim=-1, keepdim=True), values.std(dim=-1, keepdim=True)
    normalized = revin(values, mean, std, reverse=False)

    def decode(inputs):
        prefill, spread, autoregressive = model.decode(horizon, inputs, padding)
        blocks = [prefill[:, -1]] + ([autoregressive.flatten(1, 2)] if autoregressive is not None else [])
        return torch.cat(blocks, dim=1)[:, :horizon, 1:], spread[:, :horizon, 1:]

    forecast, spread = decode(normalized)
    flipped_forecast, flipped_spread = decode(-normalized)
    forecast = (forecast - flipped_forecast.flip(-1)) / 2
    spread = (spread - flipped_spread.flip(-1)) / 2
    median = forecast[..., 4:5]
    continuous = spread - spread[..., 4:5] + median
    lower = torch.cat((median, continuous[..., :4].flip(-1)), dim=-1).cummin(dim=-1).values[..., 1:].flip(-1)
    upper = torch.cat((median, continuous[..., 5:]), dim=-1).cummax(dim=-1).values[..., 1:]
    quantiles = revin(torch.cat((lower, median, upper), dim=-1), mean, std, reverse=True)
    return torch.where(positive, quantiles.clamp_min(0), quantiles).transpose(1, 2)


CASES = [((40,), 1, 0.0), ((17, 32, 33), 128, -0.5), ((64, 97), 129, 3.0), ((160, 31), 513, 0.2)]


@pytest.mark.parametrize('lengths,horizon,offset', CASES)
def test_whole_forecast_decodes_both_flips_in_one_call(runtime, monkeypatch, lengths, horizon, offset):
    values = context(lengths, offset)
    original = values.clone()
    with torch.inference_mode():
        expected = two_call_forecast(runtime.model, values, horizon)
    decode, calls = runtime.model.decode, []

    def counted(horizon, inputs, masks):
        calls.append(inputs.shape[0])
        # Stream-major: the second half is exactly the negated first half.
        torch.testing.assert_close(inputs[len(lengths):], -inputs[:len(lengths)], rtol=0, atol=0)
        assert torch.equal(masks[len(lengths):], masks[:len(lengths)])
        return decode(horizon, inputs, masks)

    monkeypatch.setattr(runtime.model, 'decode', counted)
    actual = runtime.forecast(values, horizon)
    assert calls == [2 * len(lengths)]
    assert actual.shape == (len(lengths), 9, horizon)
    # Rows are independent, but GEMM blocking depends on the row count, so a
    # 2x-row call is not bitwise identical: CPU differences are a few FP32 ulps
    # (<= 4e-6 at magnitude ~10). The serving parity gate is 1e-4.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(values, original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('lengths,horizon,offset', CASES)
def test_block_mode_matches_the_two_call_forecast(runtime, lengths, horizon, offset):
    values = context(lengths, offset)
    with torch.inference_mode():
        expected = two_call_forecast(runtime.model, values, horizon)
    stepper = TimesFMStepper(runtime)
    state = stepper.initialize(values, horizon)
    stepper.prefill([state])
    while not stepper.complete(state):
        stepper.advance([state])
    torch.testing.assert_close(stepper.finalize(state), expected, rtol=1e-5, atol=1e-5)
    stepper.release(state)
    assert state.tensor_bytes == 0


def test_shared_prefill_cohort_gives_each_request_its_own_cache_copies(runtime):
    stepper = TimesFMStepper(runtime)
    values = [context((33, 64), -0.3), context((64,), 1.0)]
    states = [stepper.initialize(value, 257) for value in values]
    stepper.prefill(states)
    storages = [cache.key.untyped_storage().data_ptr()
                for state in states for stream in state.passes for cache in stream.caches]
    assert len(set(storages)) == len(storages)  # cohorts are cloned per request and stream
    for state, value in zip(states, values, strict=True):
        while not stepper.complete(state):
            stepper.advance([state])
        with torch.inference_mode():
            expected = two_call_forecast(runtime.model, value, 257)
        torch.testing.assert_close(stepper.finalize(state), expected, rtol=1e-5, atol=1e-5)
