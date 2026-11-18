"""Shared lifecycle for opt-in integration, accuracy, and performance profiles."""

from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

import pytest

from model_integration_support import ADAPTERS, integration_cases


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_CASES = 6


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def positive_list(value):
    values = [int(part) for part in value.split(',')]
    if not values or any(number <= 0 for number in values) or len(values) != len(set(values)):
        raise ValueError('Expected unique positive comma-separated integers')
    return values


def source_metadata():
    def git(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True,
                                timeout=10, check=True)
        return result.stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'working_tree_changes': git('status', '--porcelain'),
            'python': platform.python_version()}


def child_environment(gpu_index):
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu_index),
                       NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    environment.pop('VTSFM_PRECISION', None)
    if environment.get('VTSFM_TEST_PRECISION') == 'tf32':
        environment['NVIDIA_TF32_OVERRIDE'] = '1'
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT / 'tests'), str(ROOT / 'src'), str(ROOT), environment.get('PYTHONPATH', '')])
    return environment


def profile_cases(adapter, profile, data_dir, manifest=None):
    from benchmarks.model_evaluation import assetops_cases, load_cases
    from benchmarks.prepare_data import prepare

    if manifest:
        if profile != 'accuracy':
            raise ValueError('--cases is supported only by the accuracy profile')
        cases = load_cases(manifest)
        if not cases or any('evaluation' not in case for case in cases):
            raise ValueError('Accuracy cases require observed ground-truth evaluation fields')
        return cases
    if not (data_dir / 'data_manifest.json').exists():
        print(f'Preparing the pinned AssetOpsBench fixture in {data_dir}', flush=True)
        prepare(data_dir)
    if profile == 'integration':
        return integration_cases(adapter) + assetops_cases(data_dir)
    if profile == 'accuracy':
        return assetops_cases(data_dir, count=5, batch_size=7)
    cases = []
    horizon = int(os.environ.get('VTSFM_PERF_HORIZON', '96'))
    for context in positive_list(os.environ.get('VTSFM_PERF_CONTEXTS', '128,512')):
        for batch in positive_list(os.environ.get('VTSFM_PERF_BATCHES', '1,4')):
            case, = assetops_cases(data_dir, count=1, context=context, horizon=horizon, batch_size=batch)
            case['name'] = f'assetops-b{batch}-c{context}-h{horizon}'
            cases.append(case)
    return cases


def record_check(run, check):
    run['metadata']['checks'].append(check)
    save_json(run['output'] / 'run.json', run['metadata'])


@pytest.fixture(scope='module')
def model_run():
    if platform.system() != 'Linux':
        pytest.fail('Model profiles require a Linux CUDA host; run scripts/test-models.sh there')
    adapter = os.environ.get('VTSFM_TEST_ADAPTER')
    profile = os.environ.get('VTSFM_TEST_PROFILE', 'integration')
    if adapter not in ADAPTERS or profile not in ('integration', 'accuracy', 'performance'):
        pytest.fail('Select an adapter and profile with scripts/test-models.sh')

    import requests
    from benchmarks.run import GpuMonitor, OwnedProcesses, ensure_port_free, ensure_worker_gpu, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command
    from vtsfm.compilation import EXECUTION_ENV, execution_mode
    from vtsfm.config import TORCH_VERSION, TRANSFORMERS_VERSION, VLLM_VERSION

    gpu_index = int(os.environ.get('VTSFM_TEST_GPU_INDEX', '0'))
    port = int(os.environ.get('VTSFM_TEST_PORT', '18001'))
    if gpu_index < 0 or not 1 <= port <= 65535:
        pytest.fail('GPU index must be nonnegative and port must be in [1,65535]')
    output = Path(os.environ.get('VTSFM_TEST_OUTPUT', ROOT / 'results' / f'model-{profile}' / adapter)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists():
        pytest.fail(f'Refusing to overwrite an earlier run: {output}')
    spec = get_adapter(adapter)
    from vtsfm.precision import validate_precision

    precision = validate_precision(adapter, os.environ.get('VTSFM_TEST_PRECISION', 'fp32'))
    execution = execution_mode(os.environ.get('VTSFM_TEST_EXECUTION', 'eager'))
    metadata = {**source_metadata(), 'adapter': adapter, 'profile': profile,
                'model_id': spec.model_id, 'model_revision': spec.model_revision,
                'precision': precision or 'fp32', 'tf32_override': '1' if precision == 'tf32' else '0',
                'status': 'running', 'checks': []}
    metadata['forecast_execution'] = {'native': 'eager', 'vtsfm': execution}
    monitor = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        metadata['packages'] = {name: importlib.metadata.version(name)
                                for name in ('torch', 'vllm', 'transformers', 'pytest')}
        for name, expected in [('torch', TORCH_VERSION), ('vllm', VLLM_VERSION),
                               ('transformers', TRANSFORMERS_VERSION)]:
            if metadata['packages'][name].split('+')[0] != expected:
                raise RuntimeError(f'{name} pin mismatch; rerun scripts/setup.sh --adapter {adapter}')
        ensure_port_free('127.0.0.1', port)
        monitor = GpuMonitor(gpu_index)
        monitor.assert_idle()
        metadata['gpu'] = monitor.metadata()
        save_json(output / 'run.json', metadata)
        cases = profile_cases(adapter, profile, Path(os.environ.get('VTSFM_TEST_DATA', ROOT / 'data')),
                              os.environ.get('VTSFM_TEST_CASES'))
        if not cases:
            raise ValueError('The profile has no cases')
        clients = positive_list(os.environ.get('VTSFM_PERF_CLIENTS', '1,4'))
        expected_checks = len(cases) * (2 * len(clients) if profile == 'performance' else 1)
        metadata.update(input_kind='synthetic and observed real data' if profile == 'integration' else 'observed real data',
                        expected_checks=expected_checks, case_count=len(cases))
        manifest = output / 'cases.json'
        save_json(manifest, {'cases': cases})
        environment = child_environment(gpu_index)
        environment['VTSFM_TEST_ADAPTER'] = adapter
        # Keep the independent reference and native HTTP comparator pristine.
        environment[EXECUTION_ENV] = 'eager'
        with OwnedProcesses(output) as owner:
            reference_file = output / 'native.json'
            reference = owner.spawn([sys.executable, str(ROOT / 'tests' / 'model_integration_support.py'),
                                     '--adapter', adapter, '--cases', str(manifest),
                                     '--output', str(reference_file)], environment, 'native')
            reference.wait(timeout=900)
            if reference.returncode:
                raise RuntimeError(f'Native reference failed ({reference.returncode}); see {output / "native.log"}')
            owner.stop(reference)
            monitor.wait_idle()
            native = json.loads(reference_file.read_text())
            if (native['adapter'], native['model_id'], native['model_revision']) != (
                    adapter, spec.model_id, spec.model_revision):
                raise RuntimeError('Native checkpoint does not match the selected adapter')
            ensure_worker_gpu({'gpu': {'uuid': native['runtime']['uuid']}}, monitor.uuid)
            if [entry['case'] for entry in native['cases']] != cases:
                raise RuntimeError('Native and served workloads differ')
            metadata['native_runtime'] = native['runtime']

            @contextmanager
            def backend(kind):
                monitor.assert_idle()
                ensure_port_free('127.0.0.1', port)
                if kind == 'native':
                    command = [sys.executable, '-m', 'uvicorn', 'native_forecast_server:create_app',
                               '--factory', '--host', '127.0.0.1', '--port', str(port), '--workers', '1']
                    serve_environment = dict(environment)
                    serve_environment['VLLM_PLUGINS'] = ''
                elif kind == 'vtsfm':
                    command, serve_environment = build_vllm_command(
                        adapter=adapter, port=port, forecast_execution=execution,
                        forecast_precision=precision)
                    for key in ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE',
                                'OMP_NUM_THREADS', 'PYTHONUNBUFFERED', 'PYTHONPATH'):
                        serve_environment[key] = environment[key]
                    command += ['--gpu-memory-utilization', '0.35']
                else:
                    raise ValueError(f'Unknown backend: {kind}')
                metadata.setdefault('server_commands', {})[kind] = command
                metadata.setdefault('server_execution', {})[kind] = {
                    'forecast_execution': serve_environment.get(EXECUTION_ENV, 'eager'),
                    'triton_default_fp_fusion': serve_environment.get('TRITON_DEFAULT_FP_FUSION')}
                # First use of each static shape may compile lazily. This bound
                # applies to untimed correctness calls, not measured requests.
                parity_timeout = 900 if kind == 'vtsfm' and execution == 'compile' else 180
                metadata['server_execution'][kind]['parity_timeout_s'] = parity_timeout
                server = owner.spawn(command, serve_environment, 'native-http' if kind == 'native' else 'vllm')
                stopped = False

                def stop_server():
                    nonlocal stopped
                    if not stopped:
                        owner.stop(server)
                        stopped = True

                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        url = f'http://127.0.0.1:{port}'
                        wait_ready(server, url, session, timeout=600)
                        if kind == 'native':
                            info = session.get(url + '/metadata', timeout=10)
                            info.raise_for_status()
                            ensure_worker_gpu({'gpu': {'uuid': info.json()['runtime']['uuid']}}, monitor.uuid)
                        yield {'session': session, 'url': url, 'backend': kind,
                               'parity_timeout_s': parity_timeout, 'stop': stop_server}
                finally:
                    stop_server()
                    monitor.wait_idle()

            yield {'adapter': adapter, 'spec': spec, 'cases': native['cases'], 'output': output,
                   'metadata': metadata, 'backend': backend, 'monitor': monitor, 'clients': clients}
        metadata['status'] = ('passed' if len(metadata['checks']) == expected_checks
                              and all(check['status'] == 'passed' for check in metadata['checks']) else 'failed')
        if len(metadata['checks']) != expected_checks:
            raise RuntimeError('The selected profile did not complete all expected checks')
    except BaseException as error:
        metadata.update(status='failed', error=repr(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        try:
            if monitor is not None:
                try:
                    monitor.wait_idle()
                    metadata['gpu_idle_after_cleanup'] = True
                except Exception as error:
                    metadata.update(status='failed', gpu_idle_after_cleanup=False, cleanup_error=repr(error))
                    raise
                finally:
                    monitor.close()
        finally:
            save_json(output / 'run.json', metadata)


@pytest.fixture(scope='module')
def served_model(model_run):
    with model_run['backend']('vtsfm') as server:
        yield {**model_run, **server, 'request_ids': set()}
