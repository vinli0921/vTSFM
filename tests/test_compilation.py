"""CPU-only guards for opt-in neural compilation and native-method ownership."""

from contextlib import contextmanager
import hashlib
import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vtsfm import compilation
from vtsfm.cli import build_vllm_command


def test_compilation_import_and_cli_discovery_remain_torch_free():
    code = '''
import sys
from vtsfm.cli import build_vllm_command
from vtsfm.compilation import EXECUTION_MODES
build_vllm_command(adapter='moirai2', forecast_execution='compile')
assert not {'torch', 'vllm', 'triton', 'timesfm', 'toto2', 'uni2ts'}.intersection(sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())


def test_eager_default_clears_an_inherited_compile_request(monkeypatch):
    monkeypatch.setenv(compilation.EXECUTION_ENV, 'compile')
    _, environment = build_vllm_command(adapter='toto2')
    assert compilation.EXECUTION_ENV not in environment
    monkeypatch.delenv(compilation.EXECUTION_ENV)
    # The no-op path accepts an uninitialized runtime and imports no backend.
    assert compilation.configure_compilation(SimpleNamespace(), 'toto2') is None


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_compiled_command_sets_import_time_flags_only_in_child(adapter, monkeypatch):
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '1')
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '1')
    _, environment = build_vllm_command(adapter=adapter, forecast_execution='compile')
    assert environment[compilation.EXECUTION_ENV] == 'compile'
    assert environment['TRITON_DEFAULT_FP_FUSION'] == '0'
    assert environment['NVIDIA_TF32_OVERRIDE'] == '0'
    assert os.environ['TRITON_DEFAULT_FP_FUSION'] == '1'
    assert os.environ['NVIDIA_TF32_OVERRIDE'] == '1'


@pytest.mark.parametrize('kwargs, message', [
    ({'adapter': 'chronos2'}, 'TimesFM'),
    ({'adapter': 'timesfm25', 'continuous_batching': True}, 'continuous'),
    ({'adapter': 'toto2', 'forecast_execution': 'cudagraph'}, 'Unsupported'),
])
def test_unsupported_modes_fail_before_launch(kwargs, message):
    options = {'adapter': 'toto2', 'forecast_execution': 'compile', **kwargs}
    with pytest.raises(ValueError, match=message):
        build_vllm_command(**options)


@pytest.mark.parametrize('precision', ['tf32', 'bf16'])
def test_reduced_precision_is_not_silently_combined_with_compilation(precision, monkeypatch):
    monkeypatch.setenv('VTSFM_PRECISION', precision)
    with pytest.raises(ValueError, match='fp32'):
        build_vllm_command(adapter='toto2', forecast_execution='compile')


def test_dry_run_describes_compile_settings_without_loading_models(capsys):
    from vtsfm.cli import main

    assert main(['serve', '--adapter', 'toto2', '--forecast-execution', 'compile', '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['environment'][compilation.EXECUTION_ENV] == 'compile'
    assert '--enforce-eager' in result['command']  # vLLM itself remains eager.


def test_native_backend_rejects_compile_selection(capsys):
    from vtsfm.cli import main

    with pytest.raises(SystemExit):
        main(['serve', '--adapter', 'chronos2', '--backend', 'native',
              '--forecast-execution', 'compile', '--dry-run'])
    assert '--forecast-execution' in capsys.readouterr().err


class RMSNorm:
    def forward(self, value):
        return value + 2


class SelfAttention:
    def forward(self, value):
        return value * 3


class GatedLinearUnitFeedForwardNetwork:
    def forward(self, value):
        return value - 1


class FeedForward(GatedLinearUnitFeedForwardNetwork):
    pass


class Activation:
    def forward(self, value):
        return value


class Core:
    def __init__(self, adapter):
        self.norm, self.attention, self.activation = RMSNorm(), SelfAttention(), Activation()
        self.ffn = FeedForward() if adapter == 'moirai2' else GatedLinearUnitFeedForwardNetwork()

    def named_modules(self):
        return [('', self), ('norm', self.norm), ('attention', self.attention),
                ('activation', self.activation), ('ffn', self.ffn)]

    def parameters(self):
        return [SimpleNamespace(dtype='float32')]

    def forward(self, value):
        return self.ffn.forward(self.activation.forward(self.norm.forward(self.attention.forward(value))))


@pytest.fixture
def fake_backend(monkeypatch):
    """Exercise installation, rollback and config lifetime without importing Torch."""
    original = {**compilation._BUDGET, 'recompile_limit': 8, 'accumulated_recompile_limit': 256,
                'fail_on_recompile_limit_hit': False, 'suppress_errors': False}
    active = original.copy()
    events = []

    @contextmanager
    def patch(values):
        before = active.copy()
        active.update(values)
        try:
            yield
        finally:
            active.clear()
            active.update(before)

    def compile_core(function, **options):
        events.append(('compile', options))

        def run(*args, **kwargs):
            events.append(('forward', active.copy()))
            return function(*args, **kwargs)
        return run

    def disable(function, *, recursive):
        assert recursive is True

        def native(*args, **kwargs):
            return function(*args, **kwargs)
        return native

    config = SimpleNamespace(**original, patch=patch)
    torch = SimpleNamespace(__version__=compilation.TORCH_VERSION + '+cu130', float32='float32',
        nn=SimpleNamespace(GELU=Activation, ReLU=type('ReLU', (), {}), SiLU=type('SiLU', (), {})),
        compiler=SimpleNamespace(disable=disable), compile=compile_core,
        _dynamo=SimpleNamespace(config=config), is_grad_enabled=lambda: False,
        get_float32_matmul_precision=lambda: 'highest',
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setenv(compilation.EXECUTION_ENV, 'compile')
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '0')
    monkeypatch.setenv('VTSFM_PRECISION', 'fp32')
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE', raising=False)
    monkeypatch.delenv('VTSFM_CONTINUOUS_BATCHING', raising=False)
    sources = {}
    for cls in [RMSNorm, SelfAttention, GatedLinearUnitFeedForwardNetwork, FeedForward]:
        sources[cls.__module__ + '.' + cls.__name__] = hashlib.sha256(
            inspect.getsource(cls.forward).encode()).hexdigest()
    monkeypatch.setattr(compilation, '_NATIVE_SOURCES', {name: sources for name in compilation._TARGETS})

    def runtime(adapter):
        core = Core(adapter)
        model = core if adapter == 'timesfm25' else SimpleNamespace(
            transformer=core, encoder=core, parameters=core.parameters)
        return SimpleNamespace(model=model), core
    return SimpleNamespace(torch=torch, runtime=runtime, events=events, active=active, original=original)


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_all_neural_targets_preserve_calls_and_restore_methods(adapter, fake_backend):
    runtime, core = fake_backend.runtime(adapter)
    expected = core.forward(4)
    original = core.forward
    state = compilation.configure_compilation(runtime, adapter)
    assert core.forward(4) == expected
    assert fake_backend.events[0][1]['dynamic'] is False
    assert fake_backend.events[0][1]['fullgraph'] is False
    assert fake_backend.events[-1] == ('forward', compilation._BUDGET)
    assert fake_backend.active == fake_backend.original
    assert state.describe()['successful_core_calls'] == 1
    state.close()
    state.close()
    assert core.forward == original
    assert all('forward' not in module.__dict__ for _, module in core.named_modules())
    with pytest.raises(RuntimeError, match='closed'):
        state(4)


def test_setup_failure_restores_every_native_method(fake_backend):
    runtime, core = fake_backend.runtime('toto2')

    def failure(*args, **kwargs):
        raise RuntimeError('compiler unavailable')
    fake_backend.torch.compile = failure
    with pytest.raises(RuntimeError, match='compiler unavailable'):
        compilation.configure_compilation(runtime, 'toto2')
    assert all('forward' not in module.__dict__ for _, module in core.named_modules())
    assert not hasattr(runtime, compilation._STATE)


def test_partial_native_setup_failure_rolls_back_without_masking_the_error(fake_backend):
    runtime, core = fake_backend.runtime('toto2')
    original = fake_backend.torch.compiler.disable
    count = 0

    def failure(function, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('native boundary failed')
        return original(function, **kwargs)
    fake_backend.torch.compiler.disable = failure
    with pytest.raises(RuntimeError, match='native boundary failed'):
        compilation.configure_compilation(runtime, 'toto2')
    assert all('forward' not in module.__dict__ for _, module in core.named_modules())


def test_changed_native_source_is_rejected_before_any_mutation(fake_backend, monkeypatch):
    runtime, core = fake_backend.runtime('moirai2')
    monkeypatch.setattr(compilation.inspect, 'getsource', lambda _: 'changed implementation')
    with pytest.raises(RuntimeError, match='boundary changed'):
        compilation.configure_compilation(runtime, 'moirai2')
    assert not fake_backend.events
    assert all('forward' not in module.__dict__ for _, module in core.named_modules())


def test_compiler_failure_propagates_and_restores_config(fake_backend):
    runtime, core = fake_backend.runtime('timesfm25')
    state = compilation.configure_compilation(runtime, 'timesfm25')

    def failure(*args, **kwargs):
        assert fake_backend.active['fail_on_recompile_limit_hit'] is True
        raise RuntimeError('specialization limit reached')
    state.compiled = failure
    with pytest.raises(RuntimeError, match='specialization limit'):
        core.forward(4)
    assert fake_backend.active == fake_backend.original
    assert state.calls == 0
    state.close()


@pytest.mark.parametrize('change, message', [
    ('version', 'pinned Torch'), ('tf32', 'highest FP32'), ('dtype', 'FP32 model'),
])
def test_unsupported_runtime_is_rejected_before_native_mutation(change, message, fake_backend):
    runtime, core = fake_backend.runtime('toto2')
    if change == 'version':
        fake_backend.torch.__version__ = '0.0'
    elif change == 'tf32':
        fake_backend.torch.backends.cuda.matmul.allow_tf32 = True
    else:
        runtime.model.parameters = lambda: [SimpleNamespace(dtype='bfloat16')]
    with pytest.raises((RuntimeError, ValueError), match=message):
        compilation.configure_compilation(runtime, 'toto2')
    assert not fake_backend.events
    assert all('forward' not in module.__dict__ for _, module in core.named_modules())
