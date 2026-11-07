"""Numerical admission bounds are independent of Torch and serving imports."""

from dataclasses import FrozenInstanceError, replace
import json
import subprocess
import sys

import pytest

from vtsfm.admission import forward_resident_bytes, resolve_limits
from vtsfm.adapters.timesfm25.admission import DEFAULT_LIMITS, forecast_cost


@pytest.mark.parametrize('context,patches', [(31, 1), (32, 1), (33, 2)])
@pytest.mark.parametrize('horizon,decode_steps', [(127, 0), (128, 0), (129, 1)])
def test_timesfm_cost_preserves_patch_and_output_block_boundaries(context, patches, horizon, decode_steps):
    cost = forecast_cost(3, context, horizon)
    assert (cost.rows, cost.context, cost.horizon) == (3, context, horizon)
    assert cost.work_units == 2 * 3 * (patches + 4 * decode_steps) ** 2
    assert cost.input_bytes == 3 * patches * 32 * 4
    assert cost.output_bytes == 3 * 9 * horizon * 4


def test_ragged_request_is_charged_for_the_full_padded_width():
    # The caller passes its padded width: one 31-point and one 8192-point row
    # occupy two full-width rows in the actual TimesFM execution.
    cost = forecast_cost(2, max((31, 8192)), 129)
    assert cost.input_bytes == 2 * 8192 * 4
    assert cost.work_units == 2 * forecast_cost(1, 8192, 129).work_units
    assert cost.work_units > forecast_cost(1, 31, 129).work_units + forecast_cost(1, 8192, 129).work_units


def test_validated_corner_accepts_and_next_row_is_rejected():
    corner = forecast_cost(16, 8192, 1024)
    assert corner.work_units == DEFAULT_LIMITS.max_execution_work == 2580992
    DEFAULT_LIMITS.validate_request(corner)
    DEFAULT_LIMITS.validate_launch(4, corner)
    with pytest.raises(ValueError, match='rows'):
        DEFAULT_LIMITS.validate_request(forecast_cost(17, 8192, 1024))


@pytest.mark.parametrize('cost,reason', [
    (forecast_cost(1, 8193, 1), 'context'),
    (forecast_cost(1, 32, 1025), 'horizon'),
])
def test_request_shape_limits_are_enforced_before_execution(cost, reason):
    with pytest.raises(ValueError, match=reason):
        DEFAULT_LIMITS.validate_request(cost)


def test_individually_valid_requests_must_not_exceed_a_combined_execution_budget():
    limits = replace(DEFAULT_LIMITS, max_execution_work=forecast_cost(3, 33, 129).work_units)
    request = forecast_cost(2, 33, 129)
    limits.validate_request(request)
    # Two individually admitted two-row requests cannot share one call. The
    # live planner uses this cap to split them rather than raising in the worker.
    assert limits.execution_row_cap(request.work_units // request.rows) == 3
    assert 2 + 2 > limits.execution_row_cap(request.work_units // request.rows)
    with pytest.raises(ValueError, match='work'):
        limits.validate_request(forecast_cost(4, 33, 129))


def test_request_cannot_bypass_a_lower_execution_row_limit():
    limits = replace(DEFAULT_LIMITS, max_execution_rows=2)
    with pytest.raises(ValueError, match='execution rows'):
        limits.validate_request(forecast_cost(3, 32, 128))
    assert limits.execution_row_cap(1) == 2
    assert limits.execution_row_cap(limits.max_execution_work + 1) == 0


def test_resident_bound_includes_padding_when_row_and_horizon_maxima_differ():
    costs = [forecast_cost(16, 32, 1), forecast_cost(1, 8192, 1024)]
    actual_outputs = sum(cost.output_bytes for cost in costs)
    padded_outputs = 2 * 16 * 9 * 1024 * 4
    assert padded_outputs > actual_outputs
    assert forward_resident_bytes(costs) == (
        2 * sum(cost.input_bytes for cost in costs) + 3 * actual_outputs
        + padded_outputs + 2 * 2 * (1024 * 1024 + 32))
    assert forward_resident_bytes(()) == 0


def test_launch_rejects_excess_slots_and_an_insufficient_resident_budget():
    corner = forecast_cost(16, 8192, 1024)
    with pytest.raises(ValueError, match='scheduled requests'):
        DEFAULT_LIMITS.validate_launch(5, corner)
    required = forward_resident_bytes([corner] * 4)
    assert required < DEFAULT_LIMITS.max_forward_bytes
    replace(DEFAULT_LIMITS, max_forward_bytes=required).validate_launch(4, corner)
    with pytest.raises(ValueError, match='resident'):
        replace(DEFAULT_LIMITS, max_forward_bytes=required - 1).validate_launch(4, corner)


def test_partial_limit_overrides_are_tighter_and_immutable():
    limits = resolve_limits(DEFAULT_LIMITS, json.dumps({'max_request_rows': 4, 'max_context': 512}))
    assert limits.max_request_rows == 4 and limits.max_context == 512
    assert limits.max_horizon == DEFAULT_LIMITS.max_horizon
    assert DEFAULT_LIMITS.max_request_rows == 16
    assert resolve_limits(DEFAULT_LIMITS, None) is DEFAULT_LIMITS
    with pytest.raises(FrozenInstanceError):
        limits.max_request_rows = 8


@pytest.mark.parametrize('encoded', [
    'not-json', '[]', 'null', '{"unknown":1}', '{"max_context":8193}',
    '{"max_request_rows":0}', '{"max_request_rows":true}',
    '{"max_request_rows":1.5}', '{"max_request_rows":"2"}',
    '{"max_request_rows":2,"max_request_rows":3}',
])
def test_invalid_or_unvalidated_configuration_is_rejected(encoded):
    with pytest.raises(ValueError):
        resolve_limits(DEFAULT_LIMITS, encoded)


@pytest.mark.parametrize('field', ['rows', 'context', 'horizon', 'work_units', 'input_bytes', 'output_bytes'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_cost_values_are_positive_integers(field, value):
    with pytest.raises(ValueError):
        replace(forecast_cost(1, 32, 128), **{field: value})


def test_resident_calculation_rejects_inconsistent_fp32_output_shape():
    cost = replace(forecast_cost(2, 32, 128), output_bytes=19)
    with pytest.raises(ValueError, match='FP32'):
        forward_resident_bytes([cost])


def test_admission_imports_stay_cpu_safe():
    subprocess.run([sys.executable, '-c',
                    'import sys; import vtsfm.admission; '
                    'import vtsfm.adapters.timesfm25.admission; '
                    'assert not {"torch", "vllm", "timesfm", "toto2", "uni2ts"} '
                    '& sys.modules.keys()'], check=True)
