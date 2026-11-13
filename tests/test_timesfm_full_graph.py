"""Manual complete-forecast graph parity and independent replay state."""
# ruff: noqa: F811

import os

import pytest

from test_timesfm_native_kv import device_monitor, runtimes  # noqa: F401


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_TIMESFM_FULL_GRAPH') != '1',
    reason='Set VTSFM_RUN_TIMESFM_FULL_GRAPH=1 on an idle GPU')]


@pytest.mark.parametrize('backend', ['dense', 'paged'])
@pytest.mark.parametrize('length,horizon', [(31, 1), (32, 128), (33, 129),
                                          (512, 1024), (8192, 1024)])
def test_full_forecast_graph_changes_inputs_and_resets_state(runtimes, backend, length, horizon):
    import torch
    from vtsfm.adapters.timesfm25.full_graph import FixedForecast

    reference, candidate = runtimes
    t = torch.arange(length, dtype=torch.float32, device='cuda')
    a = torch.stack((torch.sin(t / 7) + t / 1000 - 2, torch.cos(t / 11) + 3))
    b = a * -1.7 + .3
    expected_a, expected_b = reference.forecast(a, horizon), reference.forecast(b, horizon)
    with FixedForecast(candidate, a, horizon, backend=backend, capture=True, warmups=1) as runner:
        assert runner.graph is not None
        first = runner.run(a)
        saved = first.clone()
        second = runner.run(b)
        again = runner.run(a)
        assert first.data_ptr() != second.data_ptr() != again.data_ptr()
        for actual, expected in ((first, expected_a), (second, expected_b), (again, expected_a)):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        assert runner.replay_count == 3
        if backend == 'paged':
            arena = runner.stepper.arena
            assert len(arena.allocated) == 4
        with pytest.raises(ValueError, match='shape'):
            runner.run(a[:1])
        invalid = a.clone()
        invalid[0, 0] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            runner.run(invalid)
    if backend == 'paged':
        assert not arena.allocated and arena.cache_bytes == 0
    with pytest.raises(ValueError, match='closed'):
        runner.run(a)


@pytest.mark.parametrize('backend', ['dense', 'paged'])
def test_graph_safe_eager_and_graph_forecasts_preserve_row_isolation(runtimes, backend):
    import torch
    from vtsfm.adapters.timesfm25.full_graph import FixedForecast

    reference, candidate = runtimes
    a = torch.linspace(-2, 3, 66, device='cuda').reshape(2, 33)
    b = a.clone()
    b[0] = 4 - b[0] * 3
    expected = [reference.forecast(x, 257) for x in (a, b)]
    for capture in (False, True):
        with FixedForecast(candidate, a, 257, backend=backend, capture=capture, warmups=1) as runner:
            actual = [runner.run(x) for x in (a, b)]
            for output, baseline in zip(actual, expected, strict=True):
                torch.testing.assert_close(output, baseline, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(actual[0][1], actual[1][1], rtol=1e-4, atol=1e-4)
