"""Regression for replaying changing values and masks without aliasing old results."""

from dataclasses import asdict
import os
from types import SimpleNamespace

import pytest

import test_routing_engine


gpu = test_routing_engine.gpu


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('VTSFM_RUN_TOTO_GRAPH_HTTP') != '1',
                    reason='Run on an idle A100 with VTSFM_RUN_TOTO_GRAPH_HTTP=1')
def test_toto_graph_survives_multiblock_eager_fallback_and_fused_requests(gpu, tmp_path, monkeypatch):
    from model_integration_support import compare_response
    from model_serving_support import save_json
    from test_model_admission import _bounded_calls, _case, _empty, _measure, _native, _post, _rpc, _server, _wave
    from vtsfm.adapters.toto2.admission import DEFAULT_LIMITS, forecast_cost
    from vtsfm.adapters.toto2.graph_config import GRAPH_SHAPE_ENV

    monkeypatch.setenv(GRAPH_SHAPE_ENV, '2,64,800')
    matched = _case('matched', (64, 64), -1., horizon=800)
    max_shape = (16, 8192, 1024)
    maximum = _case('maximum', (max_shape[1],) * max_shape[0], 0.25, horizon=max_shape[2], quantiles=(.5,))
    parts = {name: _case(name, (64,), offset, horizon=800) for name, offset in (('a', -2.), ('b', 3.))}
    cases = {'matched': matched, 'maximum': maximum, **parts}
    references = _native(tmp_path / 'native', gpu, 'toto2', cases)
    record = {'graph_shape': [2, 64, 800], 'fallback_shape': max_shape, 'parity': []}

    def compare(label, predictions):
        parity = compare_response({'data': {'predictions': predictions}},
                                  {'name': label, 'data': cases[label]}, references[label],
                                  atol=1e-4, rtol=1e-4)
        record['parity'].append(parity)

    with _server(tmp_path / 'graph', gpu, 'toto2') as server:
        url, model, control = server['url'], server['model'], server['control']
        compare('matched', _post(url, model, 'match-before-fallback', matched))
        maximum_predictions, record['fallback_memory'] = _measure(server, gpu,
            lambda: _post(url, model, 'max-eager-fallback', maximum))
        compare('maximum', maximum_predictions)
        compare('matched', _post(url, model, 'match-after-fallback', matched))
        completed = _wave(server, 'graph-fused', ('a', 'b'), parts)
        _, calls = _bounded_calls(server, completed, asdict(DEFAULT_LIMITS), forecast_cost)
        assert [(call['kind'], call['rows']) for call in calls] == [('batch', 2)]
        for _, label, predictions in completed:
            compare(label, predictions)
        record['state'] = _empty(server)
        record['route'] = _rpc(control, url, 'bench_toto_forecast')
        assert record['route']['capture_attempts'] == 1
        assert record['route']['graph_replays'] == 3
        assert record['route']['fallbacks'] == 1
    save_json(tmp_path / 'toto-graph-evidence.json', record)


@pytest.mark.gpu
def test_toto_graph_stages_both_inputs_and_stays_stable_across_streams():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Requires CUDA')
    from vtsfm.adapters.toto2.graph_serving import TotoForecastGraph

    # Use a small operator graph to isolate stream, mask and output ownership.
    # The model parity check is the separate real-model HTTP benchmark.
    runtime = SimpleNamespace(model=torch.nn.Linear(1, 1, device='cuda', dtype=torch.float32),
                              _forecast=lambda values, observed, horizon:
                                  values.masked_fill(~observed, 0)[:, None, :horizon])
    owner = TotoForecastGraph((2, 4, 3))
    owner.warm(runtime)
    try:
        # TotoRuntime.forecast and execute both run under inference mode.
        with torch.inference_mode():
            first = torch.arange(8, device='cuda', dtype=torch.float32).reshape(2, 4)
            mask = torch.ones_like(first, dtype=torch.bool)
            a = owner.try_forecast(first, mask, 3)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                second = (torch.arange(8, device='cuda', dtype=torch.float32) + 10).reshape(4, 2).T
                mask_b = second.T.remainder(3).bool().T
                assert not second.is_contiguous() and not mask_b.is_contiguous()
                b = owner.try_forecast(second, mask_b, 3)
                expected_b = runtime._forecast(second, mask_b, 3)
            torch.cuda.current_stream().wait_stream(stream)
            assert owner.try_forecast(first, mask, 2) is None
            a_again = owner.try_forecast(first, mask, 3)
            torch.testing.assert_close(a, first[:, None, :3], atol=0, rtol=0)
            torch.testing.assert_close(b, expected_b, atol=0, rtol=0)
            torch.testing.assert_close(a_again, a, atol=0, rtol=0)
            assert a.data_ptr() != a_again.data_ptr()
            assert owner.snapshot()['capture_attempts'] == 1
            assert owner.snapshot()['graph_replays'] == 3
            assert owner.snapshot()['fallbacks'] == 1
    finally:
        owner.close()
