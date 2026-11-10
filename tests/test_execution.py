"""The prepared tensor boundary must preserve identities and reject unsafe work."""

from dataclasses import replace
import os
import subprocess
import sys

import numpy as np
import pytest

from vtsfm import execution
from vtsfm.adapters import get_adapter
from vtsfm.batching import BatchKey, BatchPlan, BatchRequest, plan_batches


def prepared(request_id='first', targets=((1., 2., 3., 4.),), **changes):
    fields = dict(
        metadata=BatchRequest(request_id, BatchKey(
            'fake', 'fake/checkpoint', 'revision', 4, 2, (0.1, 0.5, 0.9)),
            (1,) * len(targets)),
        context=np.array(targets, dtype=np.float32),
        context_lengths=(4,) * len(targets),
        observed_mask=np.ones((len(targets), 4), dtype=bool),
        item_ids=tuple(f'item-{i}' for i in range(len(targets))),
        capabilities=execution.ExecutionCapabilities(
            max_context=8, max_horizon=8, quantiles=(0.1, 0.5, 0.9),
            input_patch_size=2, output_block_size=2),
    )
    fields.update(changes)
    return execution.PreparedRequest(**fields)


def batch(*requests):
    plan, = plan_batches([r.metadata for r in requests], max_rows=10, max_requests=10)
    return execution.PreparedBatch(plan, requests)


def test_fake_adapter_round_trip_routes_unequal_requests_and_independent_groups():
    # Reversing or broadcasting output rows would associate the wrong forecast.
    from examples.fake_adapter import LastValueAdapter

    first = prepared('first', ((1., 2., 3., 4.), (5., 6., 7., 8.)),
                     item_ids=('repeated', 'repeated'))
    second = prepared('second', ((9., 10., 11., 12.),), item_ids=(None,))
    work = batch(first, second)
    assert work.plan.group_ids == (0, 1, 2)
    result = LastValueAdapter().execute(work)
    routed = result.split()

    assert list(routed) == ['first', 'second']
    assert routed['first'].item_ids == ('repeated', 'repeated')
    assert routed['second'].item_ids == (None,)
    np.testing.assert_array_equal(routed['first'].values[:, 1, :], [[4., 4.], [8., 8.]])
    np.testing.assert_array_equal(routed['second'].values[:, 1, :], [[12., 12.]])
    assert routed['first'].quantile_levels == (0.1, 0.5, 0.9)
    assert routed['second'].request_id == 'second'


@pytest.mark.parametrize('change', [
    {'context': np.zeros((1, 3), dtype=np.float32)},
    {'context': np.zeros((1, 4), dtype=np.float64)},
    {'observed_mask': np.ones((2, 4), dtype=bool)},
    {'observed_mask': np.ones((1, 4), dtype=np.int32)},
    {'context_lengths': ()}, {'context_lengths': (0,)},
    {'context_lengths': (5,)}, {'context_lengths': (True,)},
    {'context_lengths': (3,)},  # Mask incorrectly marks left padding as observed.
    {'observed_mask': np.zeros((1, 4), dtype=bool)},
    {'item_ids': ()}, {'item_ids': (42,)},
])
def test_rejects_misaligned_tensors_lengths_and_items(change):
    with pytest.raises((TypeError, ValueError)):
        prepared(**change)


def test_explicit_padding_and_missing_observations_are_distinct():
    request = prepared(context_lengths=(3,),
        observed_mask=np.array([[False, True, False, True]]),
        capabilities=execution.ExecutionCapabilities(8, 8, (0.1, 0.5, 0.9),
                                                       supports_ragged_batching=True))
    assert request.context_lengths == (3,)


def test_batch_rejects_duplicate_or_permuted_request_mapping():
    first, second = prepared(), prepared('second')
    plan, = plan_batches([first.metadata, second.metadata], max_rows=2, max_requests=2)
    with pytest.raises(ValueError):
        execution.PreparedBatch(plan, (first, first))
    with pytest.raises(ValueError):
        execution.PreparedBatch(plan, (second, first))


def test_batch_rejects_a_plan_that_merges_independent_groups():
    request = prepared(targets=((1., 2., 3., 4.), (5., 6., 7., 8.)))
    original = batch(request).plan
    merged = BatchPlan(original.key, (0, 0), original.request_slices)
    with pytest.raises(ValueError, match='groups'):
        execution.PreparedBatch(merged, (request,))


def test_batch_rejects_incompatible_options_even_when_shapes_match():
    first = prepared()
    key = replace(first.metadata.key, execution_options=(('flip', 'true'),))
    second = prepared(metadata=BatchRequest('first', key, (1,)),
        capabilities=replace(first.capabilities, execution_options=(('flip', 'true'),)))
    with pytest.raises(ValueError, match='key'):
        execution.PreparedBatch(batch(first).plan, (second,))


def test_ragged_combination_requires_explicit_adapter_capability():
    first = prepared()
    short = prepared('second', context_lengths=(2,),
        observed_mask=np.array([[False, False, True, True]]))
    with pytest.raises(ValueError, match='ragged'):
        batch(first, short)
    capable = replace(first.capabilities, supports_ragged_batching=True)
    short = prepared('second', context_lengths=(2,),
        observed_mask=np.array([[False, False, True, True]]), capabilities=capable)
    assert batch(replace(first, capabilities=capable), short).plan.group_ids == (0, 1)


@pytest.mark.parametrize('shape', [(3, 3, 2), (1, 2, 3), (1, 3, 3), (1, 3), (3, 1, 2)])
def test_output_layout_rejects_wrong_rows_quantiles_horizon_and_axes(shape):
    with pytest.raises(ValueError):
        execution.ExecutionResult(batch(prepared()), np.zeros(shape, dtype=np.float32))


def test_result_does_not_silently_cast_precision():
    with pytest.raises(ValueError, match='dtype'):
        execution.ExecutionResult(batch(prepared()), np.zeros((1, 3, 2), dtype=np.float64))


def test_point_only_contract_has_one_output_channel():
    first = prepared()
    point = replace(first, metadata=replace(first.metadata,
        key=replace(first.metadata.key, quantile_levels=())))
    result = execution.ExecutionResult(batch(point), np.ones((1, 1, 2), dtype=np.float32))
    assert result.split()['first'].quantile_levels == ()


def test_capabilities_reject_unsupported_dtype_quantile_limit_and_options():
    first = prepared()
    for key in [
        replace(first.metadata.key, dtype='bfloat16'),
        replace(first.metadata.key, quantile_levels=(0.2,)),
        replace(first.metadata.key, prediction_length=9),
        replace(first.metadata.key, execution_options=(('unknown', 'true'),)),
    ]:
        with pytest.raises(ValueError):
            first.capabilities.validate_key(key)


def test_workload_describes_dimensions_without_guessing_memory():
    work = prepared(targets=((1., 2., 3., 4.), (5., 6., 7., 8.))).workload
    assert (work.rows, work.context_points, work.forecast_values) == (2, 8, 12)
    assert (work.input_patches, work.forecast_blocks) == (4, 1)


def test_execution_options_canonicalize_order_and_prevent_unsafe_planner_merges():
    first = prepared().metadata.key
    configured = replace(first, execution_options=[['flip', 'true'], ['decode', '128']])
    same = replace(first, execution_options=(('decode', '128'), ('flip', 'true')))
    assert configured == same
    plans = plan_batches([BatchRequest('default', first, (1,)),
                          BatchRequest('configured', configured, (1,))],
                         max_rows=2, max_requests=2)
    assert len(plans) == 2


@pytest.mark.parametrize('options', [
    (('flip', 'true'), ('flip', 'false')), (('flip', True),),
    (('', 'value'),), (('flip',),), ('flip',),
])
def test_execution_options_reject_ambiguous_or_mutable_values(options):
    with pytest.raises((TypeError, ValueError)):
        replace(prepared().metadata.key, execution_options=options)


def test_torch_cpu_tensors_keep_device_dtype_and_output_views():
    torch = pytest.importorskip('torch')
    first = prepared(context=torch.tensor([[1., 2., 3., 4.]]),
                     observed_mask=torch.ones(1, 4, dtype=torch.bool))
    values = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    result = execution.ExecutionResult(batch(first), values)
    routed = result.split()['first'].values
    assert routed.device == torch.device('cpu')
    assert routed.dtype == torch.float32
    assert routed.data_ptr() == values.data_ptr()
    assert routed.tolist() == [[[0., 1.], [2., 3.], [4., 5.]]]
    with pytest.raises(ValueError, match='device'):
        execution.ExecutionResult(batch(first), torch.empty(1, 3, 2, device='meta'))


@pytest.mark.parametrize('name', ['timesfm25', 'toto2', 'moirai2'])
def test_model_capabilities_match_existing_supported_io_and_gate_future_batching(name):
    import importlib
    spec = get_adapter(name)
    settings = importlib.import_module(f'vtsfm.adapters.{name}.spec').FORECAST_SETTINGS
    capabilities = spec.execution_capabilities
    key = BatchKey(name, spec.model_id, spec.model_revision, settings.max_context,
        settings.max_horizon, settings.default_quantiles,
        execution_options=capabilities.execution_options)
    capabilities.validate_key(key)
    with pytest.raises(ValueError):
        capabilities.validate_key(replace(key, context_length=settings.max_context + 1))
    with pytest.raises(ValueError):
        capabilities.validate_key(replace(key, dtype='bfloat16'))
    # Only Moirai has GPU ragged parity evidence (tests/test_moirai_runtime.py).
    assert capabilities.supports_ragged_batching == (name == 'moirai2')
    assert capabilities.supports_block_stepping is (name == 'timesfm25')


def test_import_and_capability_discovery_do_not_load_gpu_or_model_packages():
    code = '''
import sys
from vtsfm.execution import ExecutionCapabilities, PreparedRequest, PreparedBatch, ExecutionResult
from vtsfm.adapters import list_adapters
assert all(a.execution_capabilities is not None for a in list_adapters() if a.name != 'chronos2')
heavy = {'torch', 'vllm', 'numpy', 'timesfm', 'toto2', 'uni2ts', 'transformers', 'chronos'}
assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], env=os.environ.copy(),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
