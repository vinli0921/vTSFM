import json
import subprocess
import sys

import pytest


def test_benchmark_command_is_explicit_and_does_not_leak_env(monkeypatch):
    from vtsfm.cli import build_vllm_command
    monkeypatch.setenv('VLLM_PLUGINS', 'chronos2')
    command, environment = build_vllm_command(port=18200, benchmark=True)
    assert environment['VLLM_PLUGINS'] == 'vtsfm'
    assert environment['VLLM_SERVER_DEV_MODE'] == '1'
    assert command[command.index('--io-processor-plugin') + 1] == 'vtsfm'
    assert command[command.index('--port') + 1] == '18200'
    assert command[command.index('--worker-extension-cls') + 1] == 'vtsfm.benchmark_memory.MemoryMetrics'
    assert '--enforce-eager' in command


def test_regular_serving_disables_benchmark_rpc(monkeypatch):
    from vtsfm.cli import build_vllm_command
    monkeypatch.setenv('VLLM_SERVER_DEV_MODE', '1')
    command, environment = build_vllm_command(backend='chronos2')
    assert environment.get('VLLM_SERVER_DEV_MODE') != '1'
    assert environment['VLLM_PLUGINS'] == 'chronos2'
    assert '--worker-extension-cls' not in command


def test_dry_run_never_starts_a_server_or_prints_secrets(monkeypatch):
    monkeypatch.setenv('VTSFM_TEST_SECRET', 'do-not-print-me')
    process = subprocess.run([sys.executable, '-m', 'vtsfm.cli', 'serve', '--dry-run'],
                             check=True, capture_output=True, text=True)
    output = json.loads(process.stdout)
    assert output['environment']['VLLM_PLUGINS'] == 'vtsfm'
    assert 'do-not-print-me' not in process.stdout


@pytest.mark.parametrize('port', [0, 65536, True])
def test_command_rejects_invalid_ports(port):
    from vtsfm.cli import build_vllm_command
    with pytest.raises(ValueError):
        build_vllm_command(port=port)


def test_cpu_doctor_has_no_cuda_dependency():
    process = subprocess.run([sys.executable, '-m', 'vtsfm.cli', 'doctor', '--cpu'],
                             check=True, capture_output=True, text=True)
    assert json.loads(process.stdout)['mode'] == 'cpu'
