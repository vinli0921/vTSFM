"""CPU-safe selection, CLI composition and compiler installation regressions."""

import json
import os
import subprocess
import sys

import pytest

from vtsfm.cli import build_vllm_command, main
from vtsfm.compilation import EXECUTION_ENV, compile_core, execution_mode


def test_discovery_and_compile_dry_run_do_not_import_model_packages():
    script = '''
import sys
from vtsfm.cli import main
main(['serve', '--adapter', 'moirai2', '--forecast-execution', 'compile', '--dry-run'])
assert not {'torch', 'triton', 'vllm', 'timesfm', 'toto2', 'uni2ts'} & set(sys.modules)
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)


def test_default_and_explicit_selection_respect_the_environment(monkeypatch):
    monkeypatch.delenv(EXECUTION_ENV, raising=False)
    assert execution_mode() == 'eager'
    monkeypatch.setenv(EXECUTION_ENV, 'compile')
    _, inherited = build_vllm_command(adapter='toto2')
    _, overridden = build_vllm_command(adapter='toto2', forecast_execution='eager')
    assert inherited[EXECUTION_ENV] == 'compile'
    assert overridden[EXECUTION_ENV] == 'eager'
    assert os.environ[EXECUTION_ENV] == 'compile'


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_compile_sets_contraction_policy_only_in_child(adapter, monkeypatch):
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '1')
    command, environment = build_vllm_command(adapter=adapter, forecast_execution='compile')
    assert environment['TRITON_DEFAULT_FP_FUSION'] == '0'
    assert environment[EXECUTION_ENV] == 'compile'
    assert os.environ['TRITON_DEFAULT_FP_FUSION'] == '1'
    assert '--enforce-eager' in command


@pytest.mark.parametrize('precision', ['fp32', 'tf32'])
def test_compilation_composes_with_precision_and_continuous_selection(precision, monkeypatch):
    monkeypatch.setenv('VTSFM_PRECISION', precision)
    command, environment = build_vllm_command(
        adapter='timesfm25', continuous_batching=True, forecast_execution='compile')
    assert environment['VTSFM_PRECISION'] == precision
    assert environment['VTSFM_CONTINUOUS_BATCHING'] == '1'
    assert '--scheduler-cls' in command


def test_invalid_mode_and_missing_adapter_hook_fail_before_launch():
    with pytest.raises(ValueError, match='execution mode'):
        build_vllm_command(adapter='toto2', forecast_execution='unknown')
    with pytest.raises(ValueError, match='TimesFM, Toto and Moirai'):
        build_vllm_command(adapter='chronos2', forecast_execution='compile')


def test_toto_whole_compile_requires_fp32_without_tf32(monkeypatch):
    monkeypatch.setenv('VTSFM_TOTO_COMPILE_SCOPE', 'whole')
    for mode, precision in [('eager', 'fp32'), ('compile', 'tf32'), ('compile', 'bf16')]:
        with pytest.raises(ValueError, match='whole.*FP32'):
            build_vllm_command(adapter='toto2', forecast_execution=mode, forecast_precision=precision)
    _, child = build_vllm_command(adapter='toto2', forecast_execution='compile')
    assert child['VTSFM_TOTO_COMPILE_SCOPE'] == 'whole'
    assert child['VTSFM_PRECISION'] == 'fp32'
    assert child['NVIDIA_TF32_OVERRIDE'] == '0'
    monkeypatch.setenv('VTSFM_TOTO_COMPILE_SCOPE', 'typo')
    with pytest.raises(ValueError, match='VTSFM_TOTO_COMPILE_SCOPE'):
        build_vllm_command(adapter='toto2', forecast_execution='compile')


def test_dry_run_records_selection_and_native_rejects_unimplemented_compile(capsys):
    assert main(['serve', '--adapter', 'toto2', '--forecast-execution', 'compile', '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['environment'][EXECUTION_ENV] == 'compile'
    with pytest.raises(SystemExit):
        main(['serve', '--backend', 'native', '--forecast-execution', 'compile', '--dry-run'])
    assert '--forecast-execution compile' in capsys.readouterr().err


def test_setup_failure_restores_native_methods_and_checkpoint_names(monkeypatch):
    torch = pytest.importorskip('torch')
    from torch._dynamo import config

    core = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU()).eval()
    keys = tuple(core.state_dict())
    original = core[1].forward
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '0')

    def fail(*args, **kwargs):
        raise RuntimeError('compiler setup failed')

    monkeypatch.setattr(torch, 'compile', fail)
    previous = {name: getattr(config, name) for name in
                ('recompile_limit', 'accumulated_recompile_limit',
                 'fail_on_recompile_limit_hit', 'suppress_errors')}
    with config.patch(previous):
        with pytest.raises(RuntimeError, match='compiler setup failed'):
            compile_core(core, [(core[1], 'forward')])
    assert core[1].forward == original
    assert 'forward' not in vars(core[1])
    assert tuple(core.state_dict()) == keys


def test_direct_runtime_requires_import_time_arithmetic_setting(monkeypatch):
    monkeypatch.delenv('TRITON_DEFAULT_FP_FUSION', raising=False)
    with pytest.raises(ValueError, match='before importing Torch'):
        compile_core(None, [])


def test_cache_limit_falls_back_and_retained_shapes_keep_working(monkeypatch):
    torch = pytest.importorskip('torch')
    from torch._dynamo import config

    class Core(torch.nn.Module):
        def forward(self, inputs):
            return inputs.sin() + 1

    core = Core().eval()
    eager = core.forward
    compiled_graphs = []
    original_compile = torch.compile

    def counting_backend(graph, inputs):
        compiled_graphs.append(graph)
        return graph.forward

    def trace_only(function, **options):
        options.pop('options')
        return original_compile(function, **{**options, 'backend': counting_backend})

    monkeypatch.setattr(torch, 'compile', trace_only)
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '0')
    previous = {name: getattr(config, name) for name in
                ('recompile_limit', 'accumulated_recompile_limit',
                 'fail_on_recompile_limit_hit', 'suppress_errors')}
    with config.patch(previous), torch.inference_mode():
        compile_core(core, [])
        assert config.fail_on_recompile_limit_hit is False
        assert config.suppress_errors is False
        # Exercise the real Dynamo cache behavior without CPU/GPU codegen.
        config.recompile_limit = 2
        for length in (2, 3, 4, 2, 5):
            inputs = torch.arange(length, dtype=torch.float32).unsqueeze(0)
            torch.testing.assert_close(core(inputs), eager(inputs), rtol=0, atol=0)
        assert len(compiled_graphs) == 2


def test_lazy_backend_failure_is_not_suppressed(monkeypatch):
    torch = pytest.importorskip('torch')
    from torch._dynamo import config
    from torch._dynamo.exc import BackendCompilerFailed

    original_compile = torch.compile

    def failed_backend(graph, inputs):
        raise RuntimeError('backend execution setup failed')

    def trace_only(function, **options):
        options.pop('options')
        return original_compile(function, **{**options, 'backend': failed_backend})

    monkeypatch.setattr(torch, 'compile', trace_only)
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '0')
    previous = {name: getattr(config, name) for name in
                ('recompile_limit', 'accumulated_recompile_limit',
                 'fail_on_recompile_limit_hit', 'suppress_errors')}
    with config.patch(previous), torch.inference_mode():
        core = torch.nn.Linear(2, 2).eval()
        compile_core(core, [])
        with pytest.raises(BackendCompilerFailed, match='backend execution setup failed'):
            core(torch.ones(1, 2))
