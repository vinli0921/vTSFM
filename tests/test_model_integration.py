"""Manual real-checkpoint CUDA/HTTP parity checks; run scripts/test-models.sh.

References run in a separate process before vLLM starts, on the same GPU with
the same FP32 settings. Inputs are synthetic correctness fixtures, not a
forecasting-accuracy dataset or a performance benchmark.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

import pytest

from model_integration_support import ADAPTERS, compare_response, integration_cases


pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_metadata():
    def git(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True,
                                timeout=10, check=True)
        return result.stdout.strip()

    return {'commit': git('rev-parse', 'HEAD'),
            'working_tree_changes': git('status', '--porcelain'),
            'python': platform.python_version()}


def child_environment(gpu_index):
    # Numeric visibility is required by pinned vLLM. Both processes see the
    # selected physical device as cuda:0; disable TF32 consistently for parity.
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu_index),
                       NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT / 'src'), str(ROOT), environment.get('PYTHONPATH', '')])
    return environment


@pytest.fixture(scope='module')
def served_model():
    if platform.system() != 'Linux':
        pytest.fail('Real-model integration requires a Linux CUDA host; run scripts/test-models.sh there')
    adapter = os.environ.get('VTSFM_TEST_ADAPTER')
    if adapter not in ADAPTERS:
        pytest.fail('Select an adapter with scripts/test-models.sh --adapter NAME')

    # Keep CPU collection independent of requests, NVML, Torch, and vLLM.
    import requests
    from benchmarks.run import GpuMonitor, OwnedProcesses, ensure_port_free, ensure_worker_gpu, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command
    from vtsfm.config import TORCH_VERSION, TRANSFORMERS_VERSION, VLLM_VERSION

    gpu_index = int(os.environ.get('VTSFM_TEST_GPU_INDEX', '0'))
    port = int(os.environ.get('VTSFM_TEST_PORT', '18001'))
    if gpu_index < 0 or not 1 <= port <= 65535:
        pytest.fail('GPU index must be nonnegative and port must be in [1,65535]')
    output = Path(os.environ.get('VTSFM_TEST_OUTPUT', ROOT / 'results' / 'model-integration' / adapter)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists():
        pytest.fail(f'Refusing to overwrite an earlier run: {output}; choose a new output directory')

    spec = get_adapter(adapter)
    metadata = {**source_metadata(), 'adapter': adapter, 'model_id': spec.model_id,
                'model_revision': spec.model_revision, 'input_kind': 'synthetic',
                'precision': 'float32', 'tf32_override': '0',
                'status': 'running', 'checks': []}
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
            actual = metadata['packages'][name].split('+')[0]
            if actual != expected:
                raise RuntimeError(f'{name} is {actual}, expected {expected}; rerun scripts/setup.sh --adapter {adapter}')
        ensure_port_free('127.0.0.1', port)
        monitor = GpuMonitor(gpu_index)
        monitor.assert_idle()
        metadata['gpu'] = monitor.metadata()
        save_json(output / 'run.json', metadata)
        environment = child_environment(gpu_index)
        with OwnedProcesses(output) as owner:
            reference_file = output / 'native.json'
            reference = owner.spawn([
                sys.executable, str(ROOT / 'tests' / 'model_integration_support.py'),
                '--adapter', adapter, '--output', str(reference_file),
            ], environment, 'native')
            reference.wait(timeout=900)
            if reference.returncode:
                raise RuntimeError(f'Native reference failed ({reference.returncode}); see {output / "native.log"}')
            owner.stop(reference)
            monitor.wait_idle()
            native = json.loads(reference_file.read_text())
            if (native['adapter'], native['model_id'], native['model_revision']) != (
                    adapter, spec.model_id, spec.model_revision):
                raise RuntimeError('Native reference does not match the selected adapter and checkpoint')
            ensure_worker_gpu({'gpu': {'uuid': native['runtime']['uuid']}}, monitor.uuid)
            cases = integration_cases(adapter)
            if [entry['case'] for entry in native['cases']] != cases:
                raise RuntimeError('Native and served workloads differ')
            metadata['native_runtime'] = native['runtime']

            # Recheck after reference generation: never attach to another service.
            ensure_port_free('127.0.0.1', port)
            command, serve_environment = build_vllm_command(adapter=adapter, port=port)
            for key in ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE',
                        'OMP_NUM_THREADS', 'PYTHONUNBUFFERED', 'PYTHONPATH'):
                serve_environment[key] = environment[key]
            command += ['--gpu-memory-utilization', '0.35']
            metadata['server_command'] = command
            server = owner.spawn(command, serve_environment, 'vllm')
            with requests.Session() as session:
                session.trust_env = False
                url = f'http://127.0.0.1:{port}'
                wait_ready(server, url, session, timeout=600)
                yield {'adapter': adapter, 'spec': spec, 'cases': native['cases'],
                       'session': session, 'url': url, 'output': output,
                       'metadata': metadata, 'request_ids': set()}
        metadata['status'] = ('passed' if len(metadata['checks']) == len(cases)
                              and all(check['status'] == 'passed' for check in metadata['checks'])
                              else 'failed')
        if len(metadata['checks']) != len(cases):
            raise RuntimeError('The pre-merge suite must run all four model cases')
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
                    metadata.update(status='failed', gpu_idle_after_cleanup=False,
                                    cleanup_error=repr(error))
                    raise
                finally:
                    monitor.close()
        finally:
            save_json(output / 'run.json', metadata)


@pytest.mark.parametrize('index', range(4), ids=['single', 'ragged', 'long-horizon', 'repeat'])
def test_http_forecast_matches_native(served_model, index):
    run = served_model
    entry = run['cases'][index]
    case = entry['case']
    payload = {'model': run['spec'].model_id, 'task': 'plugin', 'data': case['data']}
    record = {'case': case['name'], 'status': 'failed'}
    try:
        save_json(run['output'] / f'{index}-request.json', payload)
        response = run['session'].post(run['url'] + '/pooling', json=payload, timeout=180)
        (run['output'] / f'{index}-response.txt').write_text(response.text)
        response.raise_for_status()
        body = response.json()
        request_id = body.get('request_id')
        assert isinstance(request_id, str) and request_id, 'Missing HTTP request ID'
        assert request_id not in run['request_ids'], 'Server reused an earlier HTTP request ID'
        run['request_ids'].add(request_id)
        record.update(compare_response(body, case, entry['expected']))
        record['status'] = 'passed'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        run['metadata']['checks'].append(record)
        save_json(run['output'] / 'run.json', run['metadata'])
