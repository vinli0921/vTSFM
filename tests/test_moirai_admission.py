"""Moirai numerical admission accounts for padded recursive quantile work."""

from dataclasses import replace
import subprocess
import sys

import pytest

from vtsfm.adapters.moirai2.admission import DEFAULT_LIMITS, forecast_cost
from vtsfm.admission import forward_resident_bytes


@pytest.mark.parametrize('context,patches', [(15, 1), (16, 1), (17, 2), (7168, 448)])
def test_context_cost_uses_patch_padding(context, patches):
    cost = forecast_cost(3, context, 64)
    # One initial module call sees both history and all four future patches.
    assert cost.work_units == 3 * (patches + 4) ** 2
    assert cost.input_bytes == 3 * patches * 16 * 4
    assert cost.output_bytes == 3 * 9 * 64 * 4


@pytest.mark.parametrize('horizon,future_patches,blocks', [
    (1, 1, 1), (63, 4, 1), (64, 4, 1), (65, 5, 2),
    (128, 8, 2), (129, 9, 3), (1024, 64, 16),
])
def test_recursive_cost_uses_all_quantile_trajectories_and_full_padded_sequence(horizon, future_patches, blocks):
    rows, context_patches = 2, 3
    tokens = context_patches + future_patches
    module_rows = rows * (1 + 9 * (blocks - 1))
    reduction_candidates = rows * 81 * max(0, future_patches - 4) * 16
    cost = forecast_cost(rows, 33, horizon)
    assert cost.work_units == module_rows * tokens ** 2 + reduction_candidates
    # Public output is cropped, not rounded to a patch/block or expanded9x.
    assert cost.output_bytes == rows * 9 * horizon * 4


def test_recursive_boundary_is_not_charged_as_median_only_feedback():
    before = forecast_cost(1, 7168, 64)
    after = forecast_cost(1, 7168, 65)
    assert after.work_units > 10 * before.work_units
    assert after.work_units == 10 * 453 ** 2 + 81 * 16


def test_ragged_request_is_charged_at_the_longest_padded_history():
    lengths = (17, 7168)
    cost = forecast_cost(len(lengths), max(lengths), 129)
    assert cost.input_bytes == 2 * 7168 * 4
    assert cost.work_units == 2 * forecast_cost(1, 7168, 129).work_units
    assert cost.work_units > sum(forecast_cost(1, length, 129).work_units for length in lengths)


def test_default_corner_preserves_checkpoint_range_and_accepts_seven_sensor_rows():
    assert DEFAULT_LIMITS.max_context == 7168 and DEFAULT_LIMITS.max_horizon == 1024
    assert DEFAULT_LIMITS.max_request_rows >= 7
    assert (DEFAULT_LIMITS.max_context + 15) // 16 + (DEFAULT_LIMITS.max_horizon + 15) // 16 == 512
    corner = forecast_cost(DEFAULT_LIMITS.max_request_rows, 7168, 1024)
    assert corner.work_units == DEFAULT_LIMITS.max_execution_work
    DEFAULT_LIMITS.validate_request(corner)
    DEFAULT_LIMITS.validate_launch(4, corner)
    assert forward_resident_bytes([corner] * 4) <= DEFAULT_LIMITS.max_forward_bytes


@pytest.mark.parametrize('args,reason', [
    ((9, 7168, 1024), 'rows'),
    ((1, 7169, 1024), 'context'),
    ((1, 7168, 1025), 'horizon'),
])
def test_outside_declared_envelope_is_rejected(args, reason):
    with pytest.raises(ValueError, match=reason):
        DEFAULT_LIMITS.validate_request(forecast_cost(*args))


def test_individually_valid_recursive_requests_split_under_a_tighter_work_limit():
    one_row = forecast_cost(1, 17, 65)
    limits = replace(DEFAULT_LIMITS, max_execution_work=3 * one_row.work_units)
    pair = forecast_cost(2, 17, 65)
    limits.validate_request(pair)
    assert limits.execution_row_cap(one_row.work_units) == 3
    # Two independently valid two-row requests cannot be one model call.
    assert 2 + 2 > limits.execution_row_cap(one_row.work_units)
    with pytest.raises(ValueError, match='work'):
        limits.validate_request(forecast_cost(4, 17, 65))


@pytest.mark.parametrize('args', [
    (True, 32, 64), (1, False, 64), (1, 32, True),
    (0, 32, 64), (1, 0, 64), (1, 32, 0),
    (1.5, 32, 64), (1, 32.5, 64), (1, 32, 64.5),
])
def test_invalid_cost_dimensions_fail_without_runtime_imports(args):
    with pytest.raises(ValueError):
        forecast_cost(*args)


def test_cost_is_linear_in_independent_rows_and_monotone_in_shape():
    assert forecast_cost(4, 33, 129).work_units == 4 * forecast_cost(1, 33, 129).work_units
    costs = [forecast_cost(1, 33, horizon).work_units for horizon in (1, 16, 17, 64, 65, 128, 129, 1024)]
    assert costs == sorted(costs)


def test_moirai_cost_metadata_stays_cpu_safe():
    subprocess.run([sys.executable, '-c',
                    'import sys; import vtsfm.adapters.moirai2.admission; '
                    'assert not {"torch", "vllm", "uni2ts"} & sys.modules.keys()'], check=True)
