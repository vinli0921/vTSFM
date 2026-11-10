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


def test_new_adapters_use_routing_worker_and_concurrency_is_opt_in(monkeypatch):
    from vtsfm.cli import build_vllm_command
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1')
    for adapter in ('timesfm25', 'toto2', 'moirai2'):
        default, _ = build_vllm_command(adapter=adapter)
        assert default[default.index('--max-num-seqs') + 1] == '1'
        command, environment = build_vllm_command(adapter=adapter, max_num_seqs=3)
        assert environment['VLLM_USE_V2_MODEL_RUNNER'] == '0'
        assert command[command.index('--worker-cls') + 1] == 'vtsfm.worker.ForecastWorker'
        assert command.count('--max-num-seqs') == 1
        assert command[command.index('--max-num-seqs') + 1] == '3'


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_concurrency_limit_rejects_invalid_values(limit):
    from vtsfm.cli import build_vllm_command
    with pytest.raises(ValueError, match='positive integer'):
        build_vllm_command(adapter='toto2', max_num_seqs=limit)


def test_chronos_keeps_single_request_baseline():
    from vtsfm.cli import build_vllm_command
    with pytest.raises(ValueError, match='only for'):
        build_vllm_command(adapter='chronos2', max_num_seqs=3)


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_forecast_batching_is_opt_in_and_requires_worker_slots(monkeypatch, adapter):
    from vtsfm.cli import build_vllm_command

    monkeypatch.setenv('VTSFM_FORECAST_BATCH_ROWS', '8')
    _, normal = build_vllm_command(adapter=adapter, max_num_seqs=4)
    assert 'VTSFM_FORECAST_BATCH_ROWS' not in normal
    _, enabled = build_vllm_command(adapter=adapter, max_num_seqs=4, forecast_batch_rows=8)
    assert enabled['VTSFM_FORECAST_BATCH_ROWS'] == '8'
    with pytest.raises(ValueError, match='--max-num-seqs'):
        build_vllm_command(adapter=adapter, forecast_batch_rows=8)
    with pytest.raises(ValueError, match='2 and 16'):
        build_vllm_command(adapter=adapter, max_num_seqs=4, forecast_batch_rows=17)


def test_timesfm_limits_are_resolved_once_and_do_not_inherit_another_server(monkeypatch, tmp_path):
    from vtsfm.cli import build_vllm_command
    from vtsfm.admission import FORECAST_LIMITS_ENV

    monkeypatch.setenv(FORECAST_LIMITS_ENV, '{"max_request_rows":1}')
    _, normal = build_vllm_command(adapter='timesfm25')
    assert json.loads(normal[FORECAST_LIMITS_ENV])['max_request_rows'] == 16
    config = tmp_path / 'limits.json'
    config.write_text('{"max_request_rows":4,"max_execution_work":200}')
    _, enabled = build_vllm_command(adapter='timesfm25', max_num_seqs=4,
                                    forecast_batch_rows=16, forecast_limits=str(config))
    resolved = json.loads(enabled[FORECAST_LIMITS_ENV])
    assert resolved['max_request_rows'] == 4 and resolved['max_execution_work'] == 200
    config.unlink()  # Workers consume resolved JSON, not a potentially changed file.
    assert json.loads(enabled[FORECAST_LIMITS_ENV]) == resolved
    _, other = build_vllm_command(adapter='chronos2')
    assert FORECAST_LIMITS_ENV not in other


def test_timesfm_rejects_unsafe_launch_before_starting_a_worker(tmp_path):
    from vtsfm.cli import build_vllm_command

    with pytest.raises(ValueError, match='scheduled requests'):
        build_vllm_command(adapter='timesfm25', max_num_seqs=5)
    config = tmp_path / 'limits.json'
    config.write_text('{"max_forward_bytes":1}')
    with pytest.raises(ValueError, match='resident bytes'):
        build_vllm_command(adapter='timesfm25', forecast_limits=str(config))
    config.write_text('{"max_request_rows":17}')
    with pytest.raises(ValueError, match='validated ceiling'):
        build_vllm_command(adapter='timesfm25', forecast_limits=str(config))
    with pytest.raises(ValueError, match='admission policy'):
        build_vllm_command(adapter='chronos2', forecast_limits=str(config))


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_each_model_admits_only_its_own_resolved_envelope(adapter, monkeypatch, tmp_path):
    from vtsfm.adapters import get_adapter
    from vtsfm.admission import FORECAST_LIMITS_ENV
    from vtsfm.cli import build_vllm_command

    # A preceding server's policy must not change the new adapter's defaults.
    monkeypatch.setenv(FORECAST_LIMITS_ENV, '{"max_request_rows":1}')
    spec = get_adapter(adapter)
    assert spec.admission_limits is not None
    _, default = build_vllm_command(adapter=adapter)
    assert json.loads(default[FORECAST_LIMITS_ENV])['max_request_rows'] == spec.admission_limits.max_request_rows
    policy = tmp_path / 'policy.json'
    policy.write_text('{"max_request_rows":2}')
    command, env = build_vllm_command(adapter=adapter, max_num_seqs=4,
        forecast_batch_rows=4, forecast_limits=str(policy), forecast_queue_limit=8)
    assert json.loads(env[FORECAST_LIMITS_ENV])['max_request_rows'] == 2
    assert command[command.index('--max-num-queued-reqs') + 1] == '8'
    policy.unlink()
    assert json.loads(env[FORECAST_LIMITS_ENV])['max_request_rows'] == 2
    with pytest.raises(ValueError, match='scheduled requests'):
        build_vllm_command(adapter=adapter, max_num_seqs=spec.admission_limits.max_scheduled_requests + 1)


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_model_limits_dry_run_imports_no_runtime_packages(adapter):
    process = subprocess.run([sys.executable, '-c',
        'import sys; from vtsfm.cli import main; '
        f'main(["serve", "--adapter", "{adapter}", "--max-num-seqs", "4", '
        '"--forecast-batch-rows", "4", "--dry-run"]); '
        'assert not {"torch", "vllm", "timesfm", "toto2", "uni2ts"} & sys.modules.keys()'],
        check=True, capture_output=True, text=True)
    assert 'VTSFM_FORECAST_LIMITS' in json.loads(process.stdout)['environment']


def test_missing_limits_file_is_a_cli_error_without_runtime_imports(tmp_path):
    process = subprocess.run([sys.executable, '-m', 'vtsfm.cli', 'serve', '--adapter', 'timesfm25',
                              '--forecast-limits', str(tmp_path / 'missing.json'), '--dry-run'],
                             capture_output=True, text=True)
    assert process.returncode == 2
    assert 'No such file' in process.stderr and 'Traceback' not in process.stderr


def test_limits_are_visible_in_a_cpu_safe_dry_run():
    process = subprocess.run([sys.executable, '-m', 'vtsfm.cli', 'serve', '--adapter', 'timesfm25',
                              '--dry-run'], check=True, capture_output=True, text=True)
    limits = json.loads(json.loads(process.stdout)['environment']['VTSFM_FORECAST_LIMITS'])
    assert limits['max_scheduled_requests'] == 4 and limits['max_context'] == 8192


def test_continuous_mode_selects_its_pinned_bridge_and_bounded_queue(monkeypatch):
    from vtsfm.cli import build_vllm_command

    command, environment = build_vllm_command(adapter='timesfm25', continuous_batching=True)
    assert environment['VTSFM_CONTINUOUS_BATCHING'] == '1'
    assert command[command.index('--worker-cls') + 1] == 'vtsfm.block_worker.ForecastBlockWorker'
    assert command[command.index('--scheduler-cls') + 1] == 'vtsfm.continuous_scheduler.ForecastBlockScheduler'
    assert command[command.index('--max-num-queued-reqs') + 1] == '16'
    assert command[command.index('--max-num-seqs') + 1] == '4'
    assert '--no-async-scheduling' in command and '--enable-chunked-prefill' in command
    monkeypatch.setenv('VTSFM_CONTINUOUS_BATCHING', '1')
    _, normal = build_vllm_command(adapter='timesfm25')
    assert 'VTSFM_CONTINUOUS_BATCHING' not in normal
    with pytest.raises(ValueError, match='only TimesFM'):
        build_vllm_command(adapter='toto2', continuous_batching=True)
    with pytest.raises(ValueError, match='positive integer'):
        build_vllm_command(adapter='timesfm25', forecast_queue_limit=0)
