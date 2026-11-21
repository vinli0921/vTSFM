"""CPU fault-injection checks for the experimental paged stepper lifecycle."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from benchmarks.timesfm_native_kv import _forecast


class _FakeArena:
    def __init__(self, layers, heads, head_size, max_sequences, max_tokens, *, streams, device):
        self.device = device
        self.max_sequences = max_sequences
        self.allocated = {}
        self.reserve_error = None
        self.release_errors = {}
        self.close_error = None

    def reserve(self, identity, capacity):
        self.allocated[identity] = capacity
        if self.reserve_error is not None and len(self.allocated) == 3:
            raise self.reserve_error

    def release(self, identity):
        if identity in self.release_errors:
            raise self.release_errors.pop(identity)
        self.allocated.pop(identity, None)

    def close(self):
        if self.close_error is not None:
            raise self.close_error
        for identity in tuple(self.allocated):
            self.release(identity)


class _Attention:
    fuse_qkv = use_rotary_position_embeddings = use_per_dim_scale = True
    use_bias = False

    def forward(self):
        return 'original forward'


@pytest.fixture
def native_lifecycle(monkeypatch):
    # The standard CPU environment omits Torch and TimesFM. Stub their import
    # surface and tensor preparation, while running the real lifecycle methods.
    import numpy as np

    class Tensor(np.ndarray):
        device = 'cpu'

    stubs = {
        'torch': SimpleNamespace(Tensor=Tensor, float32=object(),
                                 inference_mode=lambda: lambda fn: fn,
                                 arange=lambda n, *, device: np.arange(n),
                                 equal=np.array_equal),
        'torch.nn': SimpleNamespace(functional=None),
        'timesfm': SimpleNamespace(),
        'timesfm.torch.util': SimpleNamespace(DecodeCache=None, revin=None, update_running_stats=None),
        'timesfm.torch.transformer': SimpleNamespace(make_attn_mask=None),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    package = importlib.import_module('vtsfm.adapters.timesfm25')
    modules = ('block_decode', 'paged_cache', 'stepping', 'native_stepping', 'runtime')
    for name in modules:
        # Save and restore both import cache and package attributes, including
        # when another test has previously loaded the real modules.
        monkeypatch.setitem(sys.modules, f'{package.__name__}.{name}', None)
        sys.modules.pop(f'{package.__name__}.{name}')
        monkeypatch.setattr(package, name, None, raising=False)

    native = importlib.import_module(f'{package.__name__}.native_stepping')
    runtime_module = importlib.import_module(f'{package.__name__}.runtime')
    block_decode = importlib.import_module(f'{package.__name__}.block_decode')
    monkeypatch.setattr(block_decode, 'validate_block_decode_source', lambda model: None)
    monkeypatch.setattr(native, 'PagedCacheArena', _FakeArena)

    created = []

    def initialize(stepper, context, horizon):
        rows, width = context.shape
        state = package.stepping.TimesFMForecastState(
            rows, width, horizon, width, 3, stepper.owner,
            padding=np.zeros_like(context, dtype=bool), output=object(),
        )
        created.append(state)
        return state

    monkeypatch.setattr(native.TimesFMStepper, 'initialize', initialize)
    runtime = object.__new__(runtime_module.TimesFMRuntime)
    runtime.model = SimpleNamespace(
        p=32, o=128, os=1024, q=10, aridx=5, h=16, hd=80, x=2,
        parameters=lambda: iter((SimpleNamespace(device='cpu'),)),
        stacked_xf=[SimpleNamespace(attn=_Attention()) for _ in range(2)],
    )
    return native.PagedTimesFMStepper, runtime, np.ones((2, 32)).view(Tensor), created


@pytest.mark.parametrize('name,value', [
    ('VTSFM_FORECAST_EXECUTION', 'compile'),
    ('VTSFM_PRECISION', 'tf32'),
])
def test_fixed_forecast_runtime_rejects_compile_or_tf32_before_loading(native_lifecycle, monkeypatch, name, value):
    _, runtime, _, _ = native_lifecycle
    monkeypatch.setenv('VTSFM_TIMESFM_FORECAST_MODE', 'paged-graph')
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match='eager FP32'):
        type(runtime)({})


def test_failed_reservation_preserves_primary_error_and_releases_other_leases(native_lifecycle):
    paged, runtime, context, created = native_lifecycle
    with paged(runtime) as stepper:
        reserve_error = RuntimeError('reservation failed after acquiring a lease')
        stepper.arena.reserve_error = reserve_error
        stuck = 'proof-1-sign-0-row-0'
        stepper.arena.release_errors[stuck] = RuntimeError('one release failed')

        with pytest.raises(RuntimeError) as caught:
            stepper.initialize(context, 128)

        assert caught.value is reserve_error
        assert any('one release failed' in note for note in caught.value.__notes__)
        assert created[0].released and created[0].output is None
        assert set(stepper.arena.allocated) == {stuck}
    assert not stepper.arena.allocated


def test_release_after_error_clears_state_and_other_sign_leases(native_lifecycle):
    paged, runtime, context, _ = native_lifecycle
    with paged(runtime) as stepper:
        state = stepper.initialize(context, 128)
        stuck, *_ = stepper.sequence_ids(state)
        release_error = RuntimeError('one release failed')
        stepper.arena.release_errors[stuck] = release_error

        with pytest.raises(RuntimeError) as caught:
            stepper.release(state)

        assert caught.value is release_error
        assert state.released and state.output is None
        assert set(stepper.arena.allocated) == {stuck}
        stepper.release(state)
    assert not stepper.arena.allocated


def test_close_releases_all_states_and_restores_original_forward_descriptors(native_lifecycle):
    paged, runtime, context, _ = native_lifecycle
    plain, overridden = [layer.attn for layer in runtime.model.stacked_xf]

    def original_override():
        return 'previous override'

    overridden.forward = original_override
    stepper = paged(runtime)
    states = [stepper.initialize(context[:1], 128) for _ in range(2)]
    stuck, *_ = stepper.sequence_ids(states[0])
    failure = RuntimeError('one release failed')
    stepper.arena.release_errors[stuck] = failure

    with pytest.raises(RuntimeError) as caught:
        stepper.close()

    assert caught.value is failure
    assert all(state.released and state.output is None for state in states)
    assert not stepper.arena.allocated
    assert 'forward' not in vars(plain)
    assert plain.forward() == 'original forward'
    assert overridden.forward is original_override
    with pytest.raises(ValueError, match='closed'):
        stepper.initialize(context, 128)
    stepper.close()


def test_live_paged_stepper_guards_shared_model_until_close(native_lifecycle):
    paged, runtime, _, _ = native_lifecycle
    alias = SimpleNamespace(model=runtime.model)
    with paged(runtime):
        with pytest.raises(ValueError, match='owned'):
            paged(alias)
        with pytest.raises(ValueError, match='owned'):
            runtime._forecast(None, None, 0)

    with paged(alias):
        pass
    with pytest.raises(ValueError, match='horizon'):
        runtime._forecast(None, None, 0)


def test_constructor_failure_rolls_back_partial_attention_patch(native_lifecycle, monkeypatch):
    paged, runtime, _, _ = native_lifecycle
    original = paged._forward_for_layer
    failure = RuntimeError('failed to install another attention patch')

    def patch_forward(stepper, index):
        if index == 1:
            raise failure
        return original(stepper, index)

    with monkeypatch.context() as scoped:
        scoped.setattr(paged, '_forward_for_layer', patch_forward)
        with pytest.raises(RuntimeError) as caught:
            paged(runtime)
    assert caught.value is failure
    assert all('forward' not in vars(layer.attn) for layer in runtime.model.stacked_xf)
    with paged(runtime):
        pass


def test_context_cleanup_preserves_body_failure_and_allows_arena_close_retry(native_lifecycle):
    paged, runtime, context, _ = native_lifecycle
    failure = RuntimeError('forecast failed')
    with pytest.raises(RuntimeError) as caught:
        with paged(runtime) as stepper:
            state = stepper.initialize(context, 128)
            stepper.arena.close_error = RuntimeError('arena close failed')
            raise failure

    assert caught.value is failure
    assert any('arena close failed' in note for note in caught.value.__notes__)
    assert state.released
    with paged(runtime):
        pass
    stepper.arena.close_error = None
    stepper.close()


def test_benchmark_preserves_forecast_failure_if_state_release_also_fails():
    failure = RuntimeError('prefill failed')

    def prefill(states):
        raise failure

    def release(state):
        raise RuntimeError('state release failed')

    stepper = SimpleNamespace(initialize=lambda context, horizon: object(), prefill=prefill,
                              release=release)
    with pytest.raises(RuntimeError) as caught:
        _forecast(None, stepper, None, 128)

    assert caught.value is failure
    assert any('state release failed' in note for note in caught.value.__notes__)
