"""Route complete forecasts by worker identity without importing GPU runtimes."""

from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vtsfm.adapters.routing import order_requests, validate_forecast
from vtsfm.adapters.transport import TransportRequest


def request(name='a', rows=2, horizon=5):
    return TransportRequest(
        request_id=name, item_ids=tuple(f'{name}-{i}' for i in range(rows)),
        context_lengths=(3,) * rows, horizon=horizon,
        quantile_levels=(0.1, 0.5, 0.9),
        context=np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
    )


def test_worker_slot_order_can_differ_from_transport_order():
    a, b, c = request('a', 1, 2), request('b', 2, 5), request('c', 3, 7)
    ordered = order_requests([a, b, c], ['c', 'a', 'b'])
    assert isinstance(ordered, tuple)
    assert ordered == (c, a, b)
    assert ordered[0] is c and ordered[1] is a and ordered[2] is b
    assert [(r.item_ids, r.horizon) for r in ordered] == [
        (('c-0', 'c-1', 'c-2'), 7), (('a-0',), 2), (('b-0', 'b-1'), 5),
    ]


@pytest.mark.parametrize('slot_ids', [
    ['a'], ['a', 'b', 'c'], ['a', 'a'], ['a', 'unknown'], [],
])
def test_missing_duplicate_or_unknown_slot_ids_fail(slot_ids):
    with pytest.raises(ValueError):
        order_requests([request('a'), request('b')], slot_ids)


def test_duplicate_transport_identity_fails_before_outputs_can_be_overwritten():
    with pytest.raises(ValueError):
        order_requests([request('a'), request('a')], ['a', 'b'])


def test_matching_output_preserves_all_native_quantile_channels():
    # Three requested quantiles are selected by IO later; the runtime returns
    # all nine native channels here, retaining their original axis ordering.
    output = np.zeros((2, 9, 5), dtype=np.float32)
    assert validate_forecast(request(), output, 9) is None


@pytest.mark.parametrize('shape', [
    (1, 9, 5), (3, 9, 5), (2, 8, 5), (2, 9, 4), (2, 9, 6),
    (9, 2, 5), (2, 5, 9), (1, 2, 9, 5), (18, 5),
])
def test_wrong_rows_quantiles_horizon_and_axes_fail(shape):
    with pytest.raises(ValueError):
        validate_forecast(request(), np.zeros(shape, dtype=np.float32), 9)


def test_forecast_dtype_must_match_its_context():
    with pytest.raises(ValueError):
        validate_forecast(request(), np.zeros((2, 9, 5), dtype=np.float64), 9)


def test_forecast_device_must_match_its_context_without_reading_values():
    prepared = replace(request(), context=SimpleNamespace(
        shape=(2, 3), dtype='float32', device='cuda:0'))
    same_device = SimpleNamespace(shape=(2, 9, 5), dtype='float32', device='cuda:0')
    assert validate_forecast(prepared, same_device, 9) is None
    for device in ('cpu', 'cuda:1'):
        output = SimpleNamespace(shape=(2, 9, 5), dtype='float32', device=device)
        with pytest.raises(ValueError):
            validate_forecast(prepared, output, 9)


def test_import_and_identity_routing_stay_cpu_safe():
    subprocess.run([sys.executable, '-c',
                    'import sys; import vtsfm.adapters.routing; '
                    'assert not {"torch", "vllm", "timesfm", "toto2", "uni2ts"} & sys.modules.keys()'],
                   check=True)
