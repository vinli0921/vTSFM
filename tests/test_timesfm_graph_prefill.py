"""CPU numerical coverage for the graph-safe shared prefill neural path.

These checks use tiny official modules and do not establish CUDA graph support.
"""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timesfm')

from test_timesfm_block_decode import _clone_caches, _new_caches, tiny_model  # noqa: E402
from vtsfm.adapters.timesfm25.block_decode import block_decode_forward  # noqa: E402


@pytest.fixture(params=[True, False], ids=['fused-qkv', 'separate-qkv'])
def model(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield tiny_model(request.param)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('patches', [1, 2, 16])
@torch.inference_mode()
def test_shared_prefill_matches_all_native_outputs_and_counters(model, patches):
    generator = torch.Generator().manual_seed(1700 + patches)
    inputs = torch.randn(2, patches, model.p, generator=generator)
    masks = torch.zeros_like(inputs, dtype=torch.bool)
    masks[0, 0, :7] = True  # A partially observed patch must keep its position.
    if patches > 1:
        masks[1, 0] = True  # A whole masked patch still affects native counters.
    reference_caches = _new_caches(model, 2, patches + 8)
    shared_caches = _clone_caches(reference_caches)
    expected, expected_caches = model(inputs, masks, reference_caches)

    from torch.utils._python_dispatch import TorchDispatchMode

    class NoHostScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError('Shared prefill extracted a tensor scalar to the host')
            return func(*args, **(kwargs or {}))

    with NoHostScalar():
        actual, returned = block_decode_forward(
            model, inputs, masks, [shared_caches], [0], prefill=True)
    assert returned[0] is shared_caches
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    for cache, reference in zip(shared_caches, expected_caches, strict=True):
        for name in ('key', 'value', 'next_index', 'num_masked'):
            torch.testing.assert_close(getattr(cache, name), getattr(reference, name),
                                       rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match='Block decode requires'):
        block_decode_forward(model, inputs, masks, [_new_caches(model, 2, patches + 8)], [0])


@torch.inference_mode()
def test_prefill_checks_full_query_capacity_before_cache_mutation(model):
    inputs = torch.randn(1, 16, model.p)
    masks = torch.zeros_like(inputs, dtype=torch.bool)
    caches = _new_caches(model, 1, 15)
    original = _clone_caches(caches)
    with pytest.raises(ValueError, match='write capacity'):
        block_decode_forward(model, inputs, masks, [caches], [0], prefill=True)
    for cache, before in zip(caches, original, strict=True):
        for name in ('key', 'value', 'next_index', 'num_masked'):
            torch.testing.assert_close(getattr(cache, name), getattr(before, name), rtol=0, atol=0)


def test_paged_graph_freezes_one_prefill_plan_for_both_signs(monkeypatch):
    from types import SimpleNamespace

    from vtsfm.adapters.timesfm25 import full_graph

    class PlanArena:
        def __init__(self, *args, **kwargs):
            self.allocations = {}
            self.plans = []

        def reserve(self, identity, capacity):
            self.allocations[identity] = capacity

        def add_plan(self, identities, start, queries):
            assert all(name in self.allocations for name in identities)
            self.plans.append((identities, start, queries))

        def close(self):
            self.allocations.clear()

    monkeypatch.setattr(full_graph._PagedStepper, '_arena_type', PlanArena)
    model = tiny_model()
    with full_graph._PagedStepper(SimpleNamespace(model=model), rows=2, patches=3, horizon=257) as stepper:
        routes = stepper.identities[0] + stepper.identities[1]
        assert len(set(routes)) == 4
        assert stepper.arena.plans == [(routes, 0, 3), (routes, 3, 4), (routes, 7, 4)]
        assert set(stepper.arena.allocations.values()) == {11}
        assert all('forward' not in vars(layer.attn) for layer in model.stacked_xf)
    assert not stepper.arena.allocations
    assert not hasattr(model, '_vtsfm_paged_stepper_active')
