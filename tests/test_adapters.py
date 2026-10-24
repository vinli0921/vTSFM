"""Adapter selection must drive discovery and launch without loading model code."""
import json
import subprocess
import sys
from dataclasses import replace

import pytest


def test_registry_describes_first_adapter_without_gpu_imports():
    process = subprocess.run([sys.executable, '-c', '''
import json,sys
from vtsfm.adapters import get_adapter,list_adapters
adapter = get_adapter('chronos2')
print(json.dumps({'names':[item.name for item in list_adapters()],
 'model':adapter.model_id,'architecture':adapter.architectures,
 'heavy':[name for name in ('torch','vllm','chronos') if name in sys.modules]}))
'''], check=True, capture_output=True, text=True)
    assert json.loads(process.stdout) == {
        'names': ['chronos2'], 'model': 'amazon/chronos-2',
        'architecture': ['Chronos2Model'], 'heavy': [],
    }


def test_unknown_adapter_fails_before_loading_any_runtime():
    from vtsfm.adapters import get_adapter
    with pytest.raises(ValueError, match='Unknown adapter.*timesfm'):
        get_adapter('timesfm')


def test_plugin_registration_and_io_selection_use_adapter_metadata(monkeypatch):
    from types import SimpleNamespace
    from vtsfm.adapters import get_adapter
    import vtsfm.plugin as plugin
    adapter = replace(get_adapter('chronos2'), name='example',
        architectures=('ExampleForecastModel',),
        model_class='example.model:ForecastModel',
        io_processor_class='example.io.ForecastIO')
    registrations = []
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(ModelRegistry=
        SimpleNamespace(register_model=lambda *args: registrations.append(args))))
    monkeypatch.setattr(plugin, 'list_adapters', lambda: (adapter,))
    monkeypatch.setattr(plugin, 'get_adapter', lambda name: adapter if name == 'example' else None)
    monkeypatch.setenv('VTSFM_ADAPTER', 'example')
    plugin.register_models()
    assert registrations == [('ExampleForecastModel', 'example.model:ForecastModel')]
    assert plugin.get_io_processor() == 'example.io.ForecastIO'


def test_launch_uses_selected_adapter_model_and_runtime_arguments(monkeypatch):
    from vtsfm.adapters import get_adapter
    import vtsfm.cli as cli
    adapter = replace(get_adapter('chronos2'), name='example', model_id='example/forecast',
        model_revision='revision-example', upstream_io_plugin='example-upstream',
        native_server_module='example.native', vllm_args=('--runner', 'pooling'))
    monkeypatch.setattr(cli, 'get_adapter', lambda name: adapter)
    command, env = cli.build_vllm_command(adapter='example')
    assert command[command.index('serve') + 1] == 'example/forecast'
    assert command[command.index('--revision') + 1] == 'revision-example'
    assert command[-2:] == ['--runner', 'pooling']
    assert '--max-model-len' not in command
    assert env['VTSFM_ADAPTER'] == 'example'
    assert env['VLLM_PLUGINS'] == 'vtsfm'
    upstream, env = cli.build_vllm_command(adapter='example', backend='upstream')
    assert upstream[upstream.index('--io-processor-plugin') + 1] == 'example-upstream'
    assert env['VLLM_PLUGINS'] == 'example-upstream'
    native, _ = cli.build_native_command(adapter='example')
    assert 'example.native:app' in native


def test_cli_lists_adapters_on_cpu_and_rejects_unknown_adapter():
    process = subprocess.run([sys.executable, '-m', 'vtsfm.cli', 'adapters'],
                             check=True, capture_output=True, text=True)
    assert [item['name'] for item in json.loads(process.stdout)['adapters']] == ['chronos2']
    bad = subprocess.run([sys.executable, '-m', 'vtsfm.cli', 'serve',
                          '--adapter', 'timesfm', '--dry-run'], capture_output=True, text=True)
    assert bad.returncode == 2
    assert 'Unknown adapter' in bad.stderr


def test_explicit_adapter_preserves_default_chronos_baseline():
    from vtsfm.cli import build_vllm_command
    command, env = build_vllm_command(adapter='chronos2')
    assert command[command.index('--dtype') + 1] == 'float32'
    assert command[command.index('--max-num-seqs') + 1] == '1'
    assert '--enforce-eager' in command
    assert '--no-enable-prefix-caching' in command
    assert env['VTSFM_ADAPTER'] == 'chronos2'
