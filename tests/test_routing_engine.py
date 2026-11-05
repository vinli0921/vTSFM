"""Manual Linux/CUDA routing checks with a real engine and HTTP endpoint.

    CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_ROUTING_ENGINE=1 PYTHONPATH=src:tests \\
      .venv-timesfm25/bin/python -m pytest tests/test_routing_engine.py -q -s

These deterministic synthetic checks load no checkpoint. They do not establish
model accuracy, cross-request collated execution, or a performance improvement.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = 'VTSFMRoutingProbe'
SERVED_MODEL = 'vtsfm-routing-probe'
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get('VTSFM_RUN_ROUTING_ENGINE') != '1',
                       reason='Set VTSFM_RUN_ROUTING_ENGINE=1 on an idle Linux GPU'),
]


def _configuration(directory):
    directory.mkdir()
    (directory / 'config.json').write_text(json.dumps({
        'architectures': [ARCHITECTURE], 'hidden_size': 32, 'num_hidden_layers': 1,
        'num_attention_heads': 1, 'vocab_size': 2, 'max_position_embeddings': 32,
    }))


def _environment(trace):
    environment = dict(os.environ)
    environment.update(VLLM_PLUGINS='vtsfm', VTSFM_ADAPTER='timesfm25',
                       VLLM_ENABLE_V1_MULTIPROCESSING='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                       VLLM_USE_V2_MODEL_RUNNER='0',
                       VTSFM_ROUTING_TRACE=str(trace), OMP_NUM_THREADS='4',
                       PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]))
    return environment


@pytest.fixture
def gpu():
    assert sys.platform == 'linux', 'The routing checks target Linux/CUDA'
    from benchmarks.run import GpuMonitor

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    assert visible.isdecimal(), 'Select one physical GPU with CUDA_VISIBLE_DEVICES=<index>'
    monitor = GpuMonitor(int(visible))
    try:
        monitor.assert_idle()
        yield monitor
    finally:
        monitor.wait_idle()
        monitor.nvml.nvmlShutdown()


@pytest.fixture
def engine(gpu, tmp_path, monkeypatch):
    trace = tmp_path / 'engine-trace.jsonl'
    for key, value in _environment(trace).items():
        if key.startswith(('VLLM_', 'VTSFM_')):
            monkeypatch.setenv(key, value)
    import vllm
    from vllm import LLM
    import routing_probe  # noqa: F401 - register in the driver and import hook
    from vtsfm.config import VLLM_VERSION

    assert vllm.__version__ == VLLM_VERSION
    directory = tmp_path / 'model'
    _configuration(directory)
    llm = LLM(
        model=str(directory), config_format='vtsfm', runner='pooling',
        hf_overrides={'architectures': [ARCHITECTURE]}, load_format='dummy',
        dtype='float32', skip_tokenizer_init=True, enforce_eager=True,
        enable_prefix_caching=False, enable_mm_embeds=True, max_model_len=32,
        max_num_batched_tokens=32, max_num_seqs=4, gpu_memory_utilization=0.1,
        worker_cls='vtsfm.worker.ForecastWorker',
        worker_extension_cls='routing_probe.RoutingProbeExtension',
    )
    try:
        yield llm, trace
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()


@pytest.fixture
def http_server(gpu, tmp_path):
    import requests
    from benchmarks.run import OwnedProcesses, ensure_port_free, wait_ready

    directory = tmp_path / 'model'
    _configuration(directory)
    trace = tmp_path / 'http-trace.jsonl'
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ensure_port_free('127.0.0.1', port)
    url = f'http://127.0.0.1:{port}'
    command = [
        # Register the fake architecture in the API process before config
        # resolution; the worker extension registers it separately after spawn.
        sys.executable, '-c', 'import routing_probe; from vllm.entrypoints.cli.main import main; main()',
        'serve', str(directory),
        '--served-model-name', SERVED_MODEL, '--host', '127.0.0.1', '--port', str(port),
        '--runner', 'pooling', '--config-format', 'vtsfm', '--load-format', 'dummy',
        '--hf-overrides', json.dumps({'architectures': [ARCHITECTURE]}),
        '--io-processor-plugin', 'vtsfm', '--dtype', 'float32', '--skip-tokenizer-init',
        '--enforce-eager', '--no-enable-prefix-caching', '--enable-mm-embeds',
        '--max-model-len', '32', '--max-num-batched-tokens', '32', '--max-num-seqs', '4',
        '--gpu-memory-utilization', '0.1', '--worker-cls', 'vtsfm.worker.ForecastWorker',
        '--worker-extension-cls', 'routing_probe.RoutingProbeExtension',
    ]
    with OwnedProcesses(tmp_path) as owner, requests.Session() as session:
        session.trust_env = False
        child = owner.spawn(command, _environment(trace), 'routing-server')
        wait_ready(child, url, session, timeout=180)
        yield url, trace


def _cases():
    rows = [
        ('A', [[1, 2, 3]], 2, [0.1, 0.5, 0.9]),
        ('B', [[11, 12], [21, 22, 23, 24]], 5, [0.2, 0.8]),
        ('C', [[30, 31], [41, 42, 43], [51, 52, 53, 54, 55]], 3, [0.3, 0.5]),
    ]
    return [{
        'inputs': [{'target': target, 'item_id': f'{name}-series-{index}'}
                   for index, target in enumerate(targets)],
        'parameters': {'prediction_length': horizon, 'quantile_levels': quantiles},
    } for name, targets, horizon, quantiles in rows]


def _expected(data):
    horizon = data['parameters']['prediction_length']
    levels = data['parameters']['quantile_levels']
    return [dict(item_id=item['item_id'], **{
        str(q): [float(item['target'][-1] + (round(q * 10) - 1) * 100 + step)
                 for step in range(horizon)] for q in levels}) for item in data['inputs']]


def _check_predictions(predictions, data):
    assert predictions == _expected(data)
    expected_keys = [str(q) for q in data['parameters']['quantile_levels']]
    for prediction in predictions:
        assert [key for key in prediction if key != 'item_id'] == expected_keys


def _encode(engine, data, prefix):
    from vllm.pooling_params import PoolingParams
    from vtsfm.adapters.timesfm25.io_processor import TimesFMIOProcessor

    processor = TimesFMIOProcessor(None, None)
    ids = [f'{prefix}-{index}' for index in range(len(data))]
    prompts = [processor.pre_process(processor.parse_data(value), request_id=identity)
               for identity, value in zip(ids, data)]
    engine = engine.llm_engine
    core = engine.engine_core
    core.call_utility('pause_scheduler', 'keep', False)
    try:
        for identity, prompt in zip(ids, prompts):
            engine.add_request(identity, prompt, PoolingParams(task='plugin'))
    finally:
        core.call_utility('resume_scheduler')
    completed = {}
    while engine.has_unfinished_requests():
        completed.update((output.request_id, output) for output in engine.step())
    assert completed.keys() == set(ids)
    responses = []
    for identity, value in zip(ids, data):
        response = processor.post_process([completed[identity]], request_id=identity)
        _check_predictions(response['predictions'], value)
        responses.append(response['predictions'])
    assert not processor.pending
    return responses


def test_real_engine_routes_three_shapes_and_reuses_slots(engine):
    llm, trace = engine
    cases = _cases()
    isolated = [_encode(llm, [data], f'isolated-{index}')[0] for index, data in enumerate(cases)]
    first = _encode(llm, cases, 'first')
    assert first == isolated
    reversed_outputs = _encode(llm, list(reversed(cases)), 'reversed')
    assert reversed_outputs == list(reversed(isolated))
    assert _encode(llm, cases, 'repeat') == isolated
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    shared = [event for event in events if len(event['request_ids']) == 3]
    assert len(shared) == 3, 'The three concurrent cycles did not share actual worker forwards'
    assert sorted(shared[0]['rows']) == [1, 2, 3]
    assert all(len(set(event['request_ids'])) == 3 for event in shared)
    assert all(event['request_ids'] != event['slot_ids'] for event in shared)
    print(json.dumps({'routing_engine': 'passed', 'shared_forwards': shared, 'trace': str(trace)}))


def _post(url, data, barrier=None, model=SERVED_MODEL, check_expected=True):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        if barrier is not None:
            barrier.wait(timeout=10)
        response = session.post(url + '/pooling', json={
            'model': model, 'task': 'plugin', 'data': data}, timeout=60)
        response.raise_for_status()
        result = response.json()
    assert isinstance(result['request_id'], str) and result['request_id']
    assert isinstance(result['created_at'], int) and result['created_at'] > 0
    if check_expected:
        _check_predictions(result['data']['predictions'], data)
    return result


def test_http_concurrent_requests_match_isolated_and_survive_reordering(http_server):
    url, trace = http_server
    cases = _cases()
    isolated = [_post(url, data) for data in cases]
    request_ids = [response['request_id'] for response in isolated]
    completions = []
    for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
        barrier = threading.Barrier(3)
        with ThreadPoolExecutor(max_workers=3) as executor:
            pending = {executor.submit(_post, url, cases[index], barrier): index for index in order}
            completion = []
            for future in as_completed(pending):
                index = pending[future]
                response = future.result()
                assert response['data']['predictions'] == isolated[index]['data']['predictions']
                request_ids.append(response['request_id'])
                completion.append(index)
            completions.append(completion)
    repeated = _post(url, cases[0])
    assert repeated['data']['predictions'] == isolated[0]['data']['predictions']
    request_ids.append(repeated['request_id'])
    assert len(set(request_ids)) == 13
    print(json.dumps({'routing_http': 'passed', 'requests': len(request_ids),
                      'completion_orders': completions, 'trace': str(trace)}))


def test_scrambled_worker_slots_route_by_feature_identity_and_recover_after_bad_shape():
    """Force transport A/B/C versus slots C/A/B through production boundaries."""
    import torch
    from routing_probe import RoutingProbeModel
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.multimodal.processing import ProcessorInputs, TimingContext
    from vtsfm.adapters.multimodal import ForecastDataParser, ForecastMultiModalProcessor
    from vtsfm.adapters.timesfm25.io_processor import TimesFMIOProcessor
    from vtsfm.worker import slot_request_ids

    cases = _cases()
    io = TimesFMIOProcessor(None, None)
    processor = object.__new__(ForecastMultiModalProcessor)
    new, states = [], {}
    for index, data in enumerate(cases):
        identity = f'original-{index}'
        prompt = io.pre_process(io.parse_data(data), request_id=identity)
        parsed = ForecastDataParser().parse_mm_data(prompt['multi_modal_data'])
        processed = processor.apply(ProcessorInputs([1], parsed), TimingContext(enabled=False))
        features = [SimpleNamespace(modality='timeseries',
                                   data=processed['mm_kwargs']['timeseries'][0])]
        new.append(SimpleNamespace(mm_features=features))
        states[f'unrelated-internal-{index}'] = SimpleNamespace(mm_features=features, num_prompt_tokens=1)
    runner = SimpleNamespace(is_multimodal_raw_input_only_model=True, device=torch.device('cpu'),
        pin_memory=False, requests=states, input_batch=SimpleNamespace(num_reqs=3,
        req_ids=['unrelated-internal-2', 'unrelated-internal-0', 'unrelated-internal-1']))
    transported = GPUModelRunner._extract_mm_kwargs(runner, SimpleNamespace(scheduled_new_reqs=new))
    assert slot_request_ids(runner) == ('original-2', 'original-0', 'original-1')
    model = RoutingProbeModel(vllm_config=SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(vtsfm_config={}))))
    model.set_request_order(lambda: slot_request_ids(runner))
    model.eval()
    values = model(torch.ones(3, dtype=torch.long), torch.zeros(3, dtype=torch.long), **transported)
    metadata = SimpleNamespace(pooling_params=[None] * 3, prompt_lens=torch.ones(3, dtype=torch.long),
        get_pooling_cursor=lambda: SimpleNamespace(num_scheduled_tokens_cpu=torch.ones(3, dtype=torch.long)))
    outputs = model.pooler(values, metadata)
    for output, index in zip(outputs, (2, 0, 1)):
        response = io.post_process([SimpleNamespace(outputs=SimpleNamespace(data=output))],
                                   request_id=f'original-{index}')
        _check_predictions(response['predictions'], cases[index])
    assert model.pooler.routes is None and not io.pending
    with pytest.raises(ValueError, match='unconsumed'):
        model.pooler(values, metadata)

    forecast = model.runtime.forecast
    model.runtime.forecast = lambda context, horizon: torch.zeros(len(context) + 1, 9, horizon)
    with pytest.raises(ValueError, match='must have shape'):
        model(torch.ones(3, dtype=torch.long), torch.zeros(3, dtype=torch.long), **transported)
    assert model.pooler.routes is None
    model.runtime.forecast = forecast
    values = model(torch.ones(3, dtype=torch.long), torch.zeros(3, dtype=torch.long), **transported)
    assert [tuple(output.shape) for output in model.pooler(values, metadata)] == [(3, 9, 3), (1, 9, 2), (2, 9, 5)]


@pytest.mark.skipif(os.environ.get('VTSFM_ROUTING_REAL_ADAPTER') not in ('timesfm25', 'toto2', 'moirai2'),
                    reason='Select a real adapter and run in its model environment')
def test_real_model_http_concurrent_matches_isolated(gpu, tmp_path):
    import numpy as np
    import requests
    from benchmarks.run import OwnedProcesses, ensure_port_free, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command

    adapter = os.environ['VTSFM_ROUTING_REAL_ADAPTER']
    model = get_adapter(adapter).model_id
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ensure_port_free('127.0.0.1', port)
    url = f'http://127.0.0.1:{port}'
    command, environment = build_vllm_command(adapter=adapter, port=port, max_num_seqs=3)
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)])
    command += ['--gpu-memory-utilization', '0.35']
    cases = _cases()
    with OwnedProcesses(tmp_path) as owner, requests.Session() as session:
        session.trust_env = False
        child = owner.spawn(command, environment, 'real-routing-server')
        wait_ready(child, url, session, timeout=180)
        isolated = [_post(url, data, model=model, check_expected=False) for data in cases]
        identities = [response['request_id'] for response in isolated]
        for order in ((0, 1, 2), (2, 1, 0), (1, 2, 0)):
            barrier = threading.Barrier(3)
            with ThreadPoolExecutor(max_workers=3) as executor:
                pending = {executor.submit(_post, url, cases[index], barrier, model, False): index
                           for index in order}
                for future in as_completed(pending):
                    index = pending[future]
                    response = future.result()
                    identities.append(response['request_id'])
                    actual = response['data']['predictions']
                    expected = isolated[index]['data']['predictions']
                    assert len(actual) == len(cases[index]['inputs'])
                    for row, baseline, item in zip(actual, expected, cases[index]['inputs']):
                        assert row['item_id'] == baseline['item_id'] == item['item_id']
                        levels = [str(q) for q in cases[index]['parameters']['quantile_levels']]
                        assert [key for key in row if key != 'item_id'] == levels
                        for level in levels:
                            assert len(row[level]) == cases[index]['parameters']['prediction_length']
                            assert np.isfinite(row[level]).all()
                            np.testing.assert_allclose(row[level], baseline[level], rtol=1e-4, atol=1e-4)
        assert len(set(identities)) == 12
    print(json.dumps({'real_routing_http': 'passed', 'adapter': adapter, 'requests': len(identities)}))
