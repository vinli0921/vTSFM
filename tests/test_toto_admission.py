"""CPU-safe Toto admission follows its pinned 32/768-point decoder."""

from dataclasses import replace
import subprocess
import sys

import pytest


@pytest.mark.parametrize('context,horizon,work,input_bytes,output_bytes', [
    (31, 31, 4, 128, 1116),
    (32, 32, 4, 128, 1152),
    (33, 33, 16, 256, 1188),
    (32, 767, 625, 128, 27612),
    (32, 768, 625, 128, 27648),
    # The second call embeds 24 median patches plus one prediction patch.
    (32, 769, 1275, 128, 27684),
    (32, 1024, 1681, 128, 36864),
    (8192, 1024, 87616, 32768, 36864),
])
def test_cost_accounts_for_padding_and_median_feedback(context, horizon, work, input_bytes, output_bytes):
    from vtsfm.adapters.toto2.admission import forecast_cost

    cost = forecast_cost(1, context, horizon)
    assert cost.work_units == work
    assert cost.input_bytes == input_bytes
    # Transport accounting is the cropped public output, not native scratch.
    assert cost.output_bytes == output_bytes


def test_ragged_rows_are_charged_at_the_longest_padded_context():
    from vtsfm.adapters.toto2.admission import forecast_cost

    cost = forecast_cost(2, max((1, 8192)), 769)
    assert cost.input_bytes == 65536
    assert cost.work_units == 170850
    assert cost.work_units > (forecast_cost(1, 1, 769).work_units
                              + forecast_cost(1, 8192, 769).work_units)


def test_default_corner_is_admitted_and_an_extra_row_is_rejected():
    from vtsfm.adapters.toto2.admission import DEFAULT_LIMITS, forecast_cost

    DEFAULT_LIMITS.validate_request(forecast_cost(16, 8192, 1024))
    DEFAULT_LIMITS.validate_launch(4, forecast_cost(16, 8192, 1024))
    with pytest.raises(ValueError, match='rows'):
        DEFAULT_LIMITS.validate_request(forecast_cost(17, 8192, 1024))


@pytest.mark.parametrize('context,horizon,reason', [(8193, 96, 'context'), (32, 1025, 'horizon')])
def test_shape_overload_fails_before_execution(context, horizon, reason):
    from vtsfm.adapters.toto2.admission import DEFAULT_LIMITS, forecast_cost

    with pytest.raises(ValueError, match=reason):
        DEFAULT_LIMITS.validate_request(forecast_cost(1, context, horizon))


def test_work_limit_includes_the_second_block_and_splits_at_whole_rows():
    from vtsfm.adapters.toto2.admission import DEFAULT_LIMITS, forecast_cost

    limits = replace(DEFAULT_LIMITS, max_execution_work=1875)
    limits.validate_request(forecast_cost(3, 32, 768))
    assert limits.execution_row_cap(forecast_cost(1, 32, 769).work_units) == 1
    with pytest.raises(ValueError, match='execution work'):
        limits.validate_request(forecast_cost(2, 32, 769))


@pytest.mark.parametrize('shape', [(0, 32, 96), (1, 0, 96), (1, 32, 0),
                                  (True, 32, 96), (1, 32.0, 96), (1, 32, '96')])
def test_cost_rejects_invalid_shape_metadata(shape):
    from vtsfm.adapters.toto2.admission import forecast_cost

    with pytest.raises(ValueError, match='positive integer'):
        forecast_cost(*shape)


def test_descriptor_exposes_admission_without_loading_model_packages():
    subprocess.run([sys.executable, '-c', '''
import sys
from vtsfm.adapters.toto2.admission import DEFAULT_LIMITS
from vtsfm.adapters.toto2.spec import TOTO2
assert TOTO2.admission_limits is DEFAULT_LIMITS
assert TOTO2.admission_cost == 'vtsfm.adapters.toto2.admission:forecast_cost'
assert not {'torch', 'vllm', 'toto2', 'timesfm', 'uni2ts'} & sys.modules.keys()
'''], check=True)
