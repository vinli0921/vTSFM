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
        allocated.append([(cache.key.data_ptr(), cache.value.data_ptr()) for cache in caches])
        return caches

    monkeypatch.setattr(stepper, '_new_caches', allocate)
    state = stepper.initialize(values, 513)
    stepper.prefill([state])
    assert [[(cache.key.data_ptr(), cache.value.data_ptr()) for cache in stream.caches]
            for stream in state.passes] == allocated
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
