"""CPU-only contracts for planning prepared, row-aligned TSFM work."""

from dataclasses import FrozenInstanceError
import os
import subprocess
import sys

import pytest

from vtsfm.batching import BatchKey, BatchPlan, BatchRequest, RequestSlice, plan_batches


def make_key(**changes):
    fields = dict(
        adapter="chronos2",
        model_id="checkpoint",
        model_revision="revision",
        context_length=256,
        prediction_length=96,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    fields.update(changes)
    return BatchKey(**fields)


def test_compatible_requests_preserve_rows_and_independent_groups():
    key = BatchKey("chronos2", "checkpoint", "revision", 256, 96, (0.1, 0.5, 0.9))
    requests = (
        BatchRequest("first", key, (2, 1)),
        BatchRequest("second", key, (1, 2)),
    )

    plans = plan_batches(requests, max_rows=6, max_requests=2)

    assert len(plans) == 1
    assert plans[0].key == key
    assert plans[0].group_ids == (0, 0, 1, 2, 3, 3)
    assert [(part.request_id, part.start, part.stop) for part in plans[0].request_slices] == [
        ("first", 0, 3),
        ("second", 3, 6),
    ]


@pytest.mark.parametrize("change", [
    {"adapter": "other"},
    {"model_id": "other"},
    {"model_revision": "other"},
    {"context_length": 128},
    {"prediction_length": 48},
    {"quantile_levels": (0.2, 0.5, 0.9)},
    {"dtype": "float16"},
])
def test_each_execution_setting_separates_batches(change):
    first = BatchRequest("first", make_key(), (1,))
    second = BatchRequest("second", make_key(**change), (1,))

    plans = plan_batches([first, second], max_rows=10, max_requests=10)

    assert len(plans) == 2
    assert [plan.key for plan in plans] == [first.key, second.key]
    assert [plan.group_ids for plan in plans] == [(0,), (0,)]


def test_fifo_planning_does_not_reorder_requests_to_merge_matching_keys():
    requests = [
        BatchRequest("first", make_key(), (1,)),
        BatchRequest("middle", make_key(model_id="other"), (1,)),
        BatchRequest("last", make_key(), (1,)),
    ]

    plans = plan_batches(iter(requests), max_rows=10, max_requests=10)

    assert [[part.request_id for part in plan.request_slices] for plan in plans] == [
        ["first"], ["middle"], ["last"],
    ]


@pytest.mark.parametrize(("max_rows", "max_requests", "expected"), [
    (4, 10, [["first", "second"], ["third"]]),
    (10, 2, [["first", "second"], ["third"]]),
    (10, 1, [["first"], ["second"], ["third"]]),
])
def test_budgets_keep_requests_atomic_and_accept_exact_boundaries(
    max_rows, max_requests, expected
):
    requests = [
        BatchRequest("first", make_key(), (1, 1)),
        BatchRequest("second", make_key(), (2,)),
        BatchRequest("third", make_key(), (1,)),
    ]

    plans = plan_batches(requests, max_rows=max_rows, max_requests=max_requests)

    assert [[part.request_id for part in plan.request_slices] for plan in plans] == expected


def test_empty_input_produces_no_plans():
    assert plan_batches([], max_rows=1, max_requests=1) == ()


def test_oversized_request_is_rejected_without_splitting_its_groups():
    request = BatchRequest("too-large", make_key(), (2, 2))

    with pytest.raises(ValueError, match="too-large"):
        plan_batches([request], max_rows=3, max_requests=10)


def test_duplicate_ids_are_rejected_across_separate_plans():
    requests = [
        BatchRequest("duplicate", make_key(), (1,)),
        BatchRequest("middle", make_key(model_id="other"), (1,)),
        BatchRequest("duplicate", make_key(), (1,)),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        plan_batches(requests, max_rows=1, max_requests=1)


@pytest.mark.parametrize("name", ["max_rows", "max_requests"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", None])
def test_budgets_require_positive_integers_even_for_empty_input(name, value):
    budgets = dict(max_rows=10, max_requests=10)
    budgets[name] = value

    with pytest.raises((TypeError, ValueError)):
        plan_batches([], **budgets)


@pytest.mark.parametrize("field", ["adapter", "model_id", "model_revision", "dtype"])
@pytest.mark.parametrize("value", ["", "  ", None, 3])
def test_key_identifiers_require_nonempty_strings(field, value):
    with pytest.raises((TypeError, ValueError)):
        make_key(**{field: value})


@pytest.mark.parametrize("field", ["context_length", "prediction_length"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", None])
def test_key_lengths_require_positive_integers(field, value):
    with pytest.raises((TypeError, ValueError)):
        make_key(**{field: value})


@pytest.mark.parametrize("levels", [
    (True,), ("0.5",), (float("nan"),), (float("inf"),),
    (-0.1,), (0.0,), (1.0,), (1.1,), (0.5, 0.5), (0.9, 0.1), None,
])
def test_quantiles_require_finite_increasing_probabilities(levels):
    with pytest.raises((TypeError, ValueError)):
        make_key(quantile_levels=levels)


def test_point_forecast_metadata_does_not_require_quantile_support():
    point = BatchRequest('point', make_key(quantile_levels=()), (2,))
    probabilistic = BatchRequest('quantiles', make_key(), (2,))
    plans = plan_batches([point, probabilistic], max_rows=4, max_requests=2)
    assert len(plans) == 2
    assert plans[0].key.quantile_levels == ()
    assert plans[0].split_outputs(['p0', 'p1']) == {'point': ('p0', 'p1')}


@pytest.mark.parametrize("request_id", ["", "  ", None, 1])
def test_request_ids_require_nonempty_strings(request_id):
    with pytest.raises((TypeError, ValueError)):
        BatchRequest(request_id, make_key(), (1,))


@pytest.mark.parametrize("groups", [(), (True,), (0,), (-1,), (1.5,), ("2",), None])
def test_group_sizes_require_nonempty_positive_integer_counts(groups):
    with pytest.raises((TypeError, ValueError)):
        BatchRequest("request", make_key(), groups)


def test_requests_require_validated_keys():
    with pytest.raises(TypeError):
        BatchRequest("request", "chronos2", (1,))


def test_planner_rejects_non_request_metadata():
    with pytest.raises(TypeError):
        plan_batches(["request"], max_rows=10, max_requests=10)


def test_metadata_copies_mutable_inputs_and_remains_hashable():
    quantiles = [0.1, 0.5, 0.9]
    groups = [2, 1]
    key = make_key(quantile_levels=quantiles)
    request = BatchRequest("request", key, groups)
    quantiles[0] = 0.2
    groups[0] = 100

    plans = plan_batches([request], max_rows=3, max_requests=1)

    assert key.quantile_levels == (0.1, 0.5, 0.9)
    assert request.group_sizes == (2, 1)
    assert plans[0].group_ids == (0, 0, 1)
    assert {key: "key", request: "request", plans[0]: "plan"}[plans[0]] == "plan"
    for metadata, field, value in [
        (key, "dtype", "float16"),
        (request, "request_id", "changed"),
        (plans[0], "group_ids", (0,)),
        (plans[0].request_slices[0], "start", 1),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(metadata, field, value)


def test_plan_copies_mutable_metadata_inputs():
    group_ids = [0, 0, 1]
    request_slices = [RequestSlice("request", 0, 3)]
    plan = BatchPlan(make_key(), group_ids, request_slices)
    group_ids.append(2)
    request_slices.clear()

    assert plan.group_ids == (0, 0, 1)
    assert plan.request_slices == (RequestSlice("request", 0, 3),)


@pytest.mark.parametrize(("request_id", "start", "stop"), [
    ("", 0, 1), ("request", -1, 1), ("request", True, 2),
    ("request", 0, False), ("request", 0, 0), ("request", 2, 1),
])
def test_request_slices_require_valid_ids_and_nonempty_integer_ranges(request_id, start, stop):
    with pytest.raises((TypeError, ValueError)):
        RequestSlice(request_id, start, stop)


@pytest.mark.parametrize(("group_ids", "slices"), [
    ((), ()),
    ((False,), (RequestSlice("request", 0, 1),)),
    ((-1,), (RequestSlice("request", 0, 1),)),
    ((0, 2), (RequestSlice("request", 0, 2),)),
    ((0, 1, 0), (RequestSlice("request", 0, 3),)),
    ((0, 1), (RequestSlice("request", 1, 2),)),
    ((0, 1), (RequestSlice("request", 0, 1),)),
    ((0,), (RequestSlice("request", 0, 2),)),
    ((0, 1), (RequestSlice("request", 0, 1), RequestSlice("request", 1, 2))),
    ((0, 0), (RequestSlice("first", 0, 1), RequestSlice("second", 1, 2))),
    ((0,), ("request",)),
])
def test_plans_reject_metadata_that_could_misroute_rows_or_share_attention(group_ids, slices):
    with pytest.raises((TypeError, ValueError)):
        BatchPlan(make_key(), group_ids, slices)


def test_plans_require_validated_keys():
    with pytest.raises(TypeError):
        BatchPlan("chronos2", (0,), (RequestSlice("request", 0, 1),))


def test_split_outputs_routes_all_prepared_rows_in_request_order():
    requests = [
        BatchRequest("first", make_key(), (2, 1)),
        BatchRequest("second", make_key(), (1,)),
    ]
    plan, = plan_batches(requests, max_rows=4, max_requests=2)
    outputs = ["target-a", "covariate-a", "target-b", "target-c"]

    assert plan.split_outputs(outputs) == {
        "first": ("target-a", "covariate-a", "target-b"),
        "second": ("target-c",),
    }
    assert list(plan.split_outputs(outputs)) == ["first", "second"]


@pytest.mark.parametrize("outputs", [[], ["a", "b"], ["a", "b", "c", "d"]])
def test_split_outputs_rejects_missing_or_extra_rows(outputs):
    plan, = plan_batches(
        [BatchRequest("request", make_key(), (2, 1))], max_rows=3, max_requests=1
    )

    with pytest.raises(ValueError, match="row"):
        plan.split_outputs(outputs)


def test_importing_planner_does_not_load_model_or_gpu_packages():
    code = """
import sys
from vtsfm.batching import BatchKey, BatchRequest, plan_batches
key = BatchKey('chronos2', 'checkpoint', 'revision', 256, 96, (0.1, 0.5, 0.9))
plan_batches([BatchRequest('request', key, (2, 1))], max_rows=3, max_requests=1)
for name in ('torch', 'vllm', 'chronos', 'numpy'):
    if name in sys.modules:
        raise SystemExit(f'Planner imported {name}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=os.environ.copy(), capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
