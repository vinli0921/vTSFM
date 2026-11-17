"""Persistent block caches preserve independent native forecasts on tiny models."""

import copy
import gc
import weakref

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timesfm')

from test_timesfm_block_decode import tiny_model  # noqa: E402
from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime  # noqa: E402
from vtsfm.adapters.timesfm25.stepping import TimesFMStepper  # noqa: E402


@pytest.fixture(params=[True, False], ids=['fused-qkv', 'separate-qkv'])
def runtimes(request):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = tiny_model(request.param)
    actual = object.__new__(TimesFMRuntime)
    actual.model = model
    reference = object.__new__(TimesFMRuntime)
    reference.model = copy.deepcopy(model)
    try:
        yield actual, reference
    finally:
        torch.set_num_threads(threads)


def context(lengths, offset=0.0):
    values = torch.full((len(lengths), max(lengths)), float('nan'))
    for row, width in enumerate(lengths):
        time = torch.arange(width, dtype=torch.float32)
        values[row, -width:] = torch.sin(time / (7 + row)) + offset + row / 3
    return values


def snapshot(state):
    return [[(cache.key.data_ptr(), cache.value.data_ptr(), cache.key.clone(), cache.value.clone())
             for cache in stream.caches] for stream in state.passes]


def assert_append_only(state, before, position):
    for stream, saved in zip(state.passes, before, strict=True):
        for cache, (key_ptr, value_ptr, key, value) in zip(stream.caches, saved, strict=True):
            assert (cache.key.data_ptr(), cache.value.data_ptr()) == (key_ptr, value_ptr)
            torch.testing.assert_close(cache.key[:, :position], key[:, :position], rtol=0, atol=0)
            torch.testing.assert_close(cache.value[:, :position], value[:, :position], rtol=0, atol=0)
            torch.testing.assert_close(cache.key[:, position + 4:], key[:, position + 4:], rtol=0, atol=0)
            torch.testing.assert_close(cache.value[:, position + 4:], value[:, position + 4:], rtol=0, atol=0)
            assert cache.next_index.tolist() == [position + 4] * state.rows
            torch.testing.assert_close(cache.num_masked, stream.caches[0].num_masked, rtol=0, atol=0)


def test_singleton_adopts_prefill_and_advances_owned_caches_in_place(runtimes, monkeypatch):
    runtime, reference = runtimes
    values = context((17, 32, 33), -0.5)
    expected = reference.forecast(values, 513)
    stepper = TimesFMStepper(runtime)
    allocated = []
    new_caches = stepper._new_caches

    def allocate(*args):
        caches = new_caches(*args)
        allocated.append(caches)
        return caches

    monkeypatch.setattr(stepper, '_new_caches', allocate)
    state = stepper.initialize(values, 513)
    stepper.prefill([state])
    # One stacked [x; -x] prefill allocation; each flip stream adopts its own
    # row half in place rather than copying it.
    assert len(allocated) == 1 and allocated[0][0].key.shape[0] == 2 * state.rows
    for stream, passes in enumerate(state.passes):
        for cache, owner in zip(passes.caches, allocated[0], strict=True):
            for actual, full in ((cache.key, owner.key), (cache.value, owner.value)):
                assert actual.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
                assert actual.data_ptr() == full[stream * state.rows].data_ptr()
                assert actual.shape[0] == state.rows
    while not stepper.complete(state):
        before, position = snapshot(state), state.cache_position
        stepper.advance([state])
        if not stepper.complete(state):
            assert_append_only(state, before, position)
    torch.testing.assert_close(stepper.finalize(state), expected, rtol=1e-4, atol=1e-4)
    assert state.cache_bytes == 0
    stepper.release(state)
    stepper.release(state)
    assert state.tensor_bytes == 0 and state.passes == []


def test_staggered_positions_reordering_and_retirement_preserve_native_outputs(runtimes):
    runtime, reference = runtimes
    cases = {'a': (context((97, 160), -1.5), 1024),
             'b': (context((32,), 1), 513),
             'short': (context((33,), 0.3), 257)}
    expected = {name: reference.forecast(values, horizon)
                for name, (values, horizon) in cases.items()}
    stepper = TimesFMStepper(runtime)
    states = {name: stepper.initialize(values, horizon)
              for name, (values, horizon) in cases.items()}
    stepper.prefill([states['a']])
    stepper.advance([states['a']])
    stepper.prefill([states['b']])
    stepper.prefill([states['short']])
    assert [states[name].cache_position for name in ('short', 'b', 'a')] == [2, 1, 9]
    assert {state.bucket for state in states.values()} == {('decode',)}
    completed = []
    order = ('short', 'b', 'a')
    while states:
        group = [states[name] for name in order if name in states]
        saved = {name: (snapshot(state), state.cache_position) for name, state in states.items()}
        stepper.advance(group)
        for name, state in list(states.items()):
            if stepper.complete(state):
                torch.testing.assert_close(stepper.finalize(state), expected[name], rtol=1e-4, atol=1e-4)
                stepper.release(state)
                del states[name]
                completed.append(name)
            else:
                assert_append_only(state, *saved[name])
        order = tuple(reversed(order))
    assert completed[0] == 'short' and completed[-1] == 'a'
    # The private mixed-position path never changes the whole-forecast model.
    values, horizon = cases['a']
    torch.testing.assert_close(runtime.forecast(values, horizon), expected['a'], rtol=0, atol=0)


def test_shared_uniform_decode_can_transition_to_mixed_positions(runtimes):
    runtime, reference = runtimes
    cases = [(context((32,), -0.5), 513), (context((32, 32), 0.5), 1024)]
    expected = [reference.forecast(values, horizon) for values, horizon in cases]
    stepper = TimesFMStepper(runtime)
    states = [stepper.initialize(values, horizon) for values, horizon in cases]
    stepper.prefill(states)
    before = [snapshot(state) for state in states]
    assert [state.cache_position for state in states] == [1, 1]
    stepper.advance(states)
    for state, saved in zip(states, before, strict=True):
        assert_append_only(state, saved, 1)
    stepper.advance([states[0]])
    assert [state.cache_position for state in states] == [9, 5]
    before = [snapshot(state) for state in states]
    stepper.advance(list(reversed(states)))
    for state, saved, position in zip(states, before, [9, 5], strict=True):
        assert_append_only(state, saved, position)
    for state, forecast in zip(states, expected, strict=True):
        while not stepper.complete(state):
            stepper.advance([state])
        torch.testing.assert_close(stepper.finalize(state), forecast, rtol=1e-4, atol=1e-4)
        stepper.release(state)


def test_cancelled_owner_does_not_retain_peers_or_mutate_their_state(runtimes):
    runtime, reference = runtimes
    values = context((64,), -1)
    expected = reference.forecast(values, 513)
    stepper = TimesFMStepper(runtime)
    cancelled = stepper.initialize(values, 513)
    peer = stepper.initialize(values, 513)
    stepper.prefill([cancelled, peer])
    stepper.advance([cancelled])
    stepper.advance([peer, cancelled])
    old_keys = [weakref.ref(cache.key) for stream in cancelled.passes for cache in stream.caches]
    peer_before = snapshot(peer)
    stepper.release(cancelled)
    stepper.release(cancelled)
    gc.collect()
    assert all(key() is None for key in old_keys)
    assert cancelled.tensor_bytes == 0
    assert [[(cache.key.data_ptr(), cache.value.data_ptr()) for cache in stream.caches]
            for stream in peer.passes] == [[(row[0], row[1]) for row in stream] for stream in peer_before]
    while not stepper.complete(peer):
        stepper.advance([peer])
    torch.testing.assert_close(stepper.finalize(peer), expected, rtol=1e-4, atol=1e-4)
    stepper.release(peer)
    assert peer.tensor_bytes == 0


def test_equal_capacity_ragged_owners_reorder_cancel_and_finish_independently(runtimes):
    runtime, reference = runtimes
    values = [context((33, 64), -1), context((64,), 0.5)]
    expected = [reference.forecast(value, 513) for value in values]
    stepper = TimesFMStepper(runtime)
    a, b = [stepper.initialize(value, 513) for value in values]
    cancelled = stepper.initialize(context((47, 64), 1), 513)
    stepper.prefill([a])
    stepper.advance([a])
    stepper.prefill([b, cancelled])
    assert [state.cache_position for state in (a, b, cancelled)] == [6, 2, 2]
    assert len({state.passes[0].caches[0].key.shape[1] for state in (a, b, cancelled)}) == 1
    saved = [(snapshot(state), state.cache_position) for state in (a, b, cancelled)]
    stepper.advance([b, a, cancelled])
    for state, before in zip((a, b, cancelled), saved, strict=True):
        assert_append_only(state, *before)
    old_keys = [weakref.ref(cache.key) for stream in cancelled.passes for cache in stream.caches]
    stepper.release(cancelled)
    gc.collect()
    assert all(key() is None for key in old_keys)
    states = [a, b]
    while states:
        stepper.advance(list(reversed(states)))
        for state in states[:]:
            if stepper.complete(state):
                index = 0 if state is a else 1
                torch.testing.assert_close(stepper.finalize(state), expected[index], rtol=1e-4, atol=1e-4)
                stepper.release(state)
                states = [active for active in states if active is not state]
        states.reverse()
    assert a.tensor_bytes == b.tensor_bytes == cancelled.tensor_bytes == 0


@pytest.mark.parametrize('cohort', [False, True], ids=['singleton', 'staggered-cohort'])
def test_ar_batches_both_flip_passes_once_with_independent_statistics(runtimes, monkeypatch, cohort):
    from timesfm.torch.util import revin, update_running_stats
    from vtsfm.adapters.timesfm25 import block_decode

    runtime, reference = runtimes
    values = [context((33, 64), -0.3)] + ([context((64,), 1)] if cohort else [])
    originals = [value.clone() for value in values]
    expected = [reference.forecast(value, 513) for value in values]
    stepper = TimesFMStepper(runtime)
    states = [stepper.initialize(value, 513) for value in values]
    rows = sum(state.rows for state in states)
    native_forward, prefill_calls = runtime.model.forward, []

    def prefill_forward(inputs, masks, caches):
        prefill_calls.append(inputs.shape[0])
        return native_forward(inputs, masks, caches)

    monkeypatch.setattr(runtime.model, 'forward', prefill_forward)
    stepper.prefill(states)
    assert prefill_calls == [2 * rows]  # Prefill batches both flips in one call.
    if cohort:
        stepper.advance([states[0]])
        assert [state.cache_position for state in states] == [6, 2]
    streams = [state.passes[index] for index in range(2) for state in states]
    expected_inputs, expected_stats = [], []
    for stream in streams:
        inputs = stream.feedback.reshape(-1, 4, 32)
        masks = torch.zeros_like(inputs, dtype=torch.bool)
        n, mu, sigma = stream.last_n, stream.last_mu, stream.last_sigma
        means, sigmas = [], []
        for patch in range(4):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, inputs[:, patch], masks[:, patch])
            means.append(mu)
            sigmas.append(sigma)
        expected_inputs.append(revin(inputs, torch.stack(means, 1), torch.stack(sigmas, 1), reverse=False))
        expected_stats.append((n.clone(), mu.clone(), sigma.clone()))
    before = [(snapshot(state), state.cache_position) for state in states]
    cache_bytes = sum(state.cache_bytes for state in states)
    work = 2 * rows * max(state.cache_capacity for state in states) ** 2
    assert stepper.model_call_work(states) == work
    shared_forward, calls = block_decode.block_decode_forward, []

    def shared(model, inputs, masks, cache_groups, positions):
        assert model is runtime.model
        assert inputs.shape == (2 * rows, 4, 32)
        assert all(actual is stream.caches for actual, stream in zip(cache_groups, streams, strict=True))
        assert positions == [state.cache_position for _ in range(2) for state in states]
        assert not masks.any()
        torch.testing.assert_close(inputs, torch.cat(expected_inputs), rtol=1e-6, atol=1e-6)
        calls.append(inputs.shape[0])
        return shared_forward(model, inputs, masks, cache_groups, positions)

    def forbidden(*args, **kwargs):
        raise AssertionError('AR bypassed the shared normal+flipped core')

    monkeypatch.setattr(block_decode, 'block_decode_forward', shared)
    monkeypatch.setattr(runtime.model, 'forward', forbidden)
    stepper.advance(states)
    assert calls == [2 * rows]
    assert sum(state.cache_bytes for state in states) == cache_bytes
    assert stepper.model_call_work(states) == work
    for state, saved in zip(states, before, strict=True):
        assert_append_only(state, *saved)
    for stream, stats in zip(streams, expected_stats, strict=True):
        for actual, target in zip((stream.last_n, stream.last_mu, stream.last_sigma), stats, strict=True):
            torch.testing.assert_close(actual, target, rtol=1e-6, atol=1e-6)
    monkeypatch.setattr(block_decode, 'block_decode_forward', shared_forward)
    for state, forecast in zip(states, expected, strict=True):
        while not stepper.complete(state):
            stepper.advance([state])
        torch.testing.assert_close(stepper.finalize(state), forecast, rtol=1e-4, atol=1e-4)
        stepper.release(state)
        assert state.tensor_bytes == 0
    for value, original in zip(values, originals, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0, equal_nan=True)


def test_singleton_ar_never_reads_a_tensor_counter_to_the_host(runtimes):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoHostScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError('AR read a tensor scalar back to the host')
            return func(*args, **(kwargs or {}))

    runtime, _ = runtimes
    stepper = TimesFMStepper(runtime)
    state = stepper.initialize(context((33, 64), -0.3), 257)
    stepper.prefill([state])
    with NoHostScalar():
        stepper.advance([state])
    stepper.release(state)


def test_invalid_owners_fail_before_a_shared_decode_can_mutate_state(runtimes, monkeypatch):
    from vtsfm.adapters.timesfm25 import block_decode

    runtime, _ = runtimes
    stepper = TimesFMStepper(runtime)
    values = context((64,), -1)
    state = stepper.initialize(values, 513)
    cancelled = stepper.initialize(values, 513)
    stepper.prefill([state, cancelled])
    before = snapshot(state)

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid owners reached shared decode')

    monkeypatch.setattr(block_decode, 'block_decode_forward', forbidden)
    with pytest.raises(ValueError, match='unique'):
        stepper.advance([state, state])
    stepper.release(cancelled)
    with pytest.raises(ValueError, match='released'):
        stepper.advance([state, cancelled])
    assert state.blocks_done == 1
    for stream, saved in zip(state.passes, before, strict=True):
        for cache, (key_ptr, value_ptr, key, value) in zip(stream.caches, saved, strict=True):
            assert (cache.key.data_ptr(), cache.value.data_ptr()) == (key_ptr, value_ptr)
            torch.testing.assert_close(cache.key, key, rtol=0, atol=0)
            torch.testing.assert_close(cache.value, value, rtol=0, atol=0)
            assert cache.next_index.tolist() == [2]
    stepper.release(state)
