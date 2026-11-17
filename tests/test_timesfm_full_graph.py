"""Complete-forecast graph ownership, validation and manual CUDA parity."""
# ruff: noqa: F811

from contextlib import contextmanager, nullcontext
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_timesfm_native_kv import device_monitor, runtimes  # noqa: F401


full_graph = pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_TIMESFM_FULL_GRAPH') != '1',
    reason='Set VTSFM_RUN_TIMESFM_FULL_GRAPH=1 on an idle GPU')


@pytest.mark.parametrize('end_fails', [False, True])
def test_capture_failure_preserves_original_error_and_releases_owner(monkeypatch, end_fails):
    torch = pytest.importorskip('torch')
    pytest.importorskip('timesfm')
    from vtsfm.adapters.timesfm25 import full_graph as graph_module

    class Example(torch.Tensor):
        @property
        def device(self):
            return torch.device('cuda:0')

    example = torch.ones(1, 32).as_subclass(Example)
    runtime = SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter([example])))
    stream = SimpleNamespace(wait_stream=Mock(), synchronize=Mock())
    graph = SimpleNamespace(reset=Mock())
    stepper = SimpleNamespace(arena=SimpleNamespace(cache_bytes=16), close=Mock())
    failure = RuntimeError('forecast failed during capture')
    owners = []

    @contextmanager
    def capture(graph, *, stream):
        try:
            yield
        finally:
            if end_fails:
                raise RuntimeError('capture end failed')

    def body(owner):
        owners.append(owner)
        if len(owners) == 2:
            raise failure
        return example

    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', lambda: graph)
    monkeypatch.setattr(torch.cuda, 'graph', capture)
    monkeypatch.setattr(graph_module, '_PagedStepper', lambda *args: stepper)
    monkeypatch.setattr(graph_module.FixedForecast, '_body', body)

    with pytest.raises(RuntimeError) as caught:
        graph_module.FixedForecast(runtime, example, 1, backend='paged', warmups=1)

    assert caught.value is failure
    assert len(owners) == 2
    assert owners[0].graph is owners[0].input is owners[0].static_output is None
    stream.synchronize.assert_called_once()
    assert graph.reset.call_count == (0 if end_fails else 1)
    stepper.close.assert_called_once()


@pytest.mark.parametrize('rows,field,bad_shape', [
    (0, None, None),
    (1, 'next_index', (1, 1)),
    (1, 'num_masked', (1, 1)),
    (1, 'key', (1, 1, 2, 4)),
    (1, 'value', (1, 1, 2, 4)),
])
def test_frozen_paged_prepare_rejects_invalid_counter_only_cache(rows, field, bad_shape):
    torch = pytest.importorskip('torch')
    pytest.importorskip('timesfm')
    from vtsfm.adapters.timesfm25.full_graph import _PagedStepper

    stepper = object.__new__(_PagedStepper)
    stepper.model = SimpleNamespace(p=32, x=2, h=2, hd=4)
    stepper._active = tuple(range(rows)), (0,) * rows, (0,) * rows
    inputs = torch.zeros(rows, 2, 32)
    masks = torch.zeros_like(inputs, dtype=torch.bool)
    caches = [SimpleNamespace(next_index=torch.zeros(rows, dtype=torch.int32),
                              num_masked=torch.zeros(rows, dtype=torch.int32),
                              key=torch.empty(rows, 0, 2, 4), value=torch.empty(rows, 0, 2, 4))
              for _ in range(stepper.model.x)]
    if field is not None:
        setattr(caches[-1], field, torch.zeros(bad_shape))

    with pytest.raises(ValueError, match='counters with empty dense K/V'):
        stepper.prepare(stepper.model, inputs, masks, [caches], [0])


@pytest.mark.gpu
@full_graph
@pytest.mark.parametrize('backend', ['dense', 'paged'])
@pytest.mark.parametrize('rows,length,horizon', [(1, 31, 1), (2, 32, 128), (8, 33, 129),
                                               (2, 512, 1024), (2, 8192, 1024)])
def test_full_forecast_graph_changes_inputs_and_resets_state(runtimes, backend, rows, length, horizon):
    import torch
    from vtsfm.adapters.timesfm25.full_graph import FixedForecast

    reference, candidate = runtimes
    t = torch.arange(length, dtype=torch.float32, device='cuda')
    series = (torch.sin(t / 7) + t / 1000 - 2, torch.cos(t / 11) + 3)
    a = torch.stack([series[row % 2] + row / 16 for row in range(rows)])
    b = a * -1.7 + .3
    expected_a, expected_b = reference.forecast(a, horizon), reference.forecast(b, horizon)
    with FixedForecast(candidate, a, horizon, backend=backend, capture=True, warmups=1) as runner:
        assert runner.graph is not None
        first = runner.run(a)
        saved = first.clone()
        second = runner.run(b)
        again = runner.run(a)
        assert len({output.data_ptr() for output in (first, second, again)}) == 3
        for actual, expected in ((first, expected_a), (second, expected_b), (again, expected_a)):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        assert runner.replay_count == 3
        if backend == 'paged':
            arena = runner.stepper.arena
            assert len(arena.allocated) == 2 * rows
        with pytest.raises(ValueError, match='shape'):
            runner.run(a[:, :-1])
        invalid = a.clone()
        invalid[0, 0] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            runner.run(invalid)
    if backend == 'paged':
        assert not arena.allocated and arena.cache_bytes == 0
    with pytest.raises(ValueError, match='closed'):
        runner.run(a)


@pytest.mark.gpu
@full_graph
@pytest.mark.parametrize('backend', ['dense', 'paged'])
def test_graph_safe_eager_and_graph_forecasts_preserve_row_isolation(runtimes, backend):
    import torch
    from vtsfm.adapters.timesfm25.full_graph import FixedForecast

    reference, candidate = runtimes
    a = torch.linspace(-2, 3, 66, device='cuda').reshape(2, 33)
    b = a.clone()
    b[0] = 4 - b[0] * 3
    expected = [reference.forecast(x, 257) for x in (a, b)]
    noncontiguous = a.T.contiguous().T
    with pytest.raises(ValueError, match='contiguous'):
        FixedForecast(candidate, noncontiguous, 257, backend=backend, warmups=1)
    for capture in (False, True):
        with FixedForecast(candidate, a, 257, backend=backend, capture=capture, warmups=1) as runner:
            for invalid, reason in ((a.to(torch.float64), 'dtype'), (a.cpu(), 'device'),
                                    (noncontiguous, 'contiguous')):
                with pytest.raises(ValueError, match=reason):
                    runner.run(invalid)
            actual = [runner.run(x) for x in (a, b)]
            for output, baseline in zip(actual, expected, strict=True):
                torch.testing.assert_close(output, baseline, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(actual[0][1], actual[1][1], rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@full_graph
def test_paged_capture_body_failure_releases_real_cuda_arena(runtimes, monkeypatch):
    import torch
    from vtsfm.adapters.timesfm25.full_graph import FixedForecast

    _, candidate = runtimes
    example = torch.arange(33, device='cuda', dtype=torch.float32)[None]
    body = FixedForecast._body
    failure = RuntimeError('injected CUDA capture body failure')
    owners = []

    def inject_failure(owner):
        result = body(owner)
        if torch.cuda.is_current_stream_capturing():
            owners.append(owner)
            raise failure
        return result

    with monkeypatch.context() as patch:
        patch.setattr(FixedForecast, '_body', inject_failure)
        with pytest.raises(RuntimeError) as caught:
            FixedForecast(candidate, example, 129, backend='paged', warmups=1)

    assert caught.value is failure
    owner, = owners
    assert owner.graph is owner.input is owner.static_output is None
    assert not owner.stepper.arena.allocated and owner.stepper.arena.cache_bytes == 0
    assert not owner.stepper.arena.plans
    with FixedForecast(candidate, example, 129, backend='paged', warmups=1) as retry:
        assert retry.run(example).shape == (1, 9, 129)
