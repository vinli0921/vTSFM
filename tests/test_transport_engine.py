"""Manual Linux/CUDA transport tests against the real pinned vLLM engine.

Run on an idle GPU in a configured model environment:

    VTSFM_RUN_TRANSPORT_ENGINE=1 PYTHONPATH=src:tests \\
      .venv-timesfm25/bin/python -m pytest tests/test_transport_engine.py -q -s

No checkpoint, model download, HTTP server, or upstream-vLLM patch is used.
CPU collection does not import Torch or vLLM. These checks are deliberately
separate from the small automatic Modal contract suite.
"""

import gc
import json
import os
import sys
from types import SimpleNamespace

import pytest


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get('VTSFM_RUN_TRANSPORT_ENGINE') != '1',
                       reason='Set VTSFM_RUN_TRANSPORT_ENGINE=1 on an idle Linux GPU'),
]


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    assert sys.platform == 'linux', 'The transport engine test targets Linux/CUDA'
    import pynvml

    pynvml.nvmlInit()
    try:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        handle = (pynvml.nvmlDeviceGetHandleByIndex(int(visible)) if visible.isdecimal()
                  else pynvml.nvmlDeviceGetHandleByUUID(visible))
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        assert not processes, 'Refusing to use a GPU with an existing compute process'
    finally:
        pynvml.nvmlShutdown()

    # Ensure worker IPC is exercised rather than an in-process model call.
    os.environ['VLLM_PLUGINS'] = 'vtsfm'
    os.environ['VTSFM_ADAPTER'] = 'timesfm25'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    import torch
    import vllm
    from vllm import LLM
    import transport_probe

    assert vllm.__version__ == '0.16.0', 'This probe targets the pinned vLLM 0.16.0 boundary'
    assert torch.cuda.is_available(), 'The engine probe requires working CUDA'
    directory = tmp_path_factory.mktemp('transport-model')
    (directory / 'config.json').write_text(json.dumps({
        'architectures': [transport_probe.ARCHITECTURE],
        'hidden_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 1,
        'vocab_size': 2, 'max_position_embeddings': 32,
    }))
    llm = LLM(
        model=str(directory), config_format='vtsfm', runner='pooling',
        hf_overrides={'architectures': [transport_probe.ARCHITECTURE]},
        load_format='dummy', dtype='float32', skip_tokenizer_init=True,
        enforce_eager=True, enable_prefix_caching=False, enable_mm_embeds=True,
        max_model_len=32, max_num_batched_tokens=32, max_num_seqs=4,
        gpu_memory_utilization=0.1,
        worker_extension_cls='transport_probe.ProbeWorkerExtension',
    )
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()


def _prompt(request_id, targets, horizon, quantiles=(0.1, 0.5, 0.9)):
    from vtsfm.adapters.timesfm25.io_processor import TimesFMIOProcessor

    processor = TimesFMIOProcessor(None)
    parsed = processor.parse_request({
        'inputs': [{'target': target, 'item_id': f'{request_id}-series-{index}'}
                   for index, target in enumerate(targets)],
        'parameters': {'prediction_length': horizon, 'quantile_levels': list(quantiles)},
    })
    return processor.pre_process(parsed, request_id=request_id)


def _expected_fingerprint(prompt):
    from transport_probe import fingerprint
    from vtsfm.adapters.multimodal import MODALITY
    from vtsfm.adapters.transport import decode_requests

    values = prompt['multi_modal_data'][MODALITY]
    context = values['context']
    metadata = values['request_metadata']
    request, = decode_requests(context, metadata)
    return fingerprint(request)


def _encode_and_check(engine, prompts):
    from vllm.pooling_params import PoolingParams

    expected = [_expected_fingerprint(prompt) for prompt in prompts]
    # LLM.encode submits sequentially while the separate EngineCore is already
    # scheduling, so its first request can finish before the next ADD arrives.
    # The pinned engine's acknowledged utility calls freeze scheduling while all
    # ADDs cross the actual IPC path. Resume uses the same ordered input socket.
    # These private LLM enqueue/drain helpers are test-only v0.16 integration;
    # normal serving is unchanged and the scheduler/worker are not patched.
    core = engine.llm_engine.engine_core
    core.call_utility('pause_scheduler')
    try:
        engine._validate_and_add_requests(
            prompts, PoolingParams(task='plugin'), use_tqdm=False, lora_request=None)
    finally:
        core.call_utility('resume_scheduler')
    outputs = engine._run_engine(use_tqdm=False)
    assert len(outputs) == len(prompts)
    fingerprints = [output.outputs.data.reshape(-1).tolist() for output in outputs]
    assert len({row[0] for row in fingerprints}) == 1, 'Requests did not share one worker forward'
    for actual, digest in zip(fingerprints, expected):
        assert actual[1] == len(prompts), 'Worker did not receive every request exactly once'
        assert actual[2:] == digest, 'A numerical payload or request metadata changed in transport'
    return {'invocation': int(fingerprints[0][0]), 'requests_in_forward': len(prompts),
            'request_ids': [output.request_id for output in outputs],
            'fingerprints': [row[2:] for row in fingerprints]}


def test_distinct_requests_share_one_real_engine_forward_and_survive_slot_reuse(engine):
    # Different row counts, contexts, ragged histories, horizons and quantiles
    # must survive transport even though they are not execution-compatible.
    first = _prompt('A', [[1, 2, 3, 4]], 2)
    second = _prompt('B', [[11, 12, 13], [21, 22, 23, 24, 25, 26]], 3, (0.2, 0.8))
    observations = [_encode_and_check(engine, [first, second])]
    observations.append(_encode_and_check(engine, [
        _prompt('C', [[31, 32]], 1),
        _prompt('D', [[41, 42, 43, 44], [51, 52, 53, 54], [61, 62, 63, 64]], 4),
    ]))
    # Exercise vLLM's stacked-tensor reducer too: identical numerical values
    # are still two distinct requests, rather than one shared field.
    observations.append(_encode_and_check(engine, [
        _prompt('E', [[1, 2, 3, 4]], 2), _prompt('F', [[1, 2, 3, 4]], 2),
    ]))
    observations.append(_encode_and_check(engine, [first, second]))
    assert len({row['invocation'] for row in observations}) == len(observations)
    assert observations[0]['fingerprints'] == observations[-1]['fingerprints']
    print(json.dumps({'transport_engine': 'passed', 'observations': observations}))


def test_legacy_shared_fields_reproduce_the_last_group_overwrite():
    """Record the old failure through the actual pinned worker extraction code."""
    import torch
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalFieldElem, MultiModalKwargsItem
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    requests = []
    contexts = [torch.tensor([[1., 2.]]), torch.tensor([[11., 12.], [21., 22.]])]
    for context, horizon in zip(contexts, (2, 3)):
        fields = {name: MultiModalFieldElem(
            data=value, field=MultiModalFieldConfig.shared('timeseries', batch_size=1).field)
            for name, value in {'context': context, 'horizon': horizon}.items()}
        requests.append(SimpleNamespace(mm_features=[SimpleNamespace(
            modality='timeseries', data=MultiModalKwargsItem(fields))]))
    runner = SimpleNamespace(is_multimodal_raw_input_only_model=True,
                             device=torch.device('cpu'), pin_memory=False)
    transported = GPUModelRunner._extract_mm_kwargs(runner,
        SimpleNamespace(scheduled_new_reqs=requests))
    assert torch.equal(transported['context'], contexts[-1])
    assert transported['horizon'] == 3


def test_production_wrapper_rejects_multiple_forecasts_before_running_model():
    import torch
    from torch import nn
    from vtsfm.adapters.multimodal import ForecastDataParser, ForecastMultiModalProcessor
    from vtsfm.adapters.pooling import ForecastModel
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # Exercise real parser/processor/reducer with a tiny deterministic runtime.
    processor = object.__new__(ForecastMultiModalProcessor)
    prompts = [_prompt('a', [[1, 2, 3]], 2), _prompt('b', [[11, 12]], 3)]
    requests = []
    for prompt in prompts:
        parsed = ForecastDataParser().parse_mm_data(prompt['multi_modal_data'])
        processed = processor.apply([1], parsed, {})
        requests.append(SimpleNamespace(mm_features=[SimpleNamespace(
            modality='timeseries', data=processed['mm_kwargs']['timeseries'][0])]))
    runner = SimpleNamespace(is_multimodal_raw_input_only_model=True,
                             device=torch.device('cpu'), pin_memory=False)
    transported = GPUModelRunner._extract_mm_kwargs(runner,
        SimpleNamespace(scheduled_new_reqs=requests))

    calls = []

    def forecast(context, horizon):
        calls.append((context.clone(), horizon))
        return context[:, -1:, None].expand(-1, 9, horizon)

    model = ForecastModel.__new__(ForecastModel)
    nn.Module.__init__(model)
    model.runtime = SimpleNamespace(forecast=forecast)
    with pytest.raises(ValueError, match='routing is not enabled'):
        model(torch.tensor([1, 1]), torch.tensor([0, 0]), **transported)
    assert calls == []
    single = GPUModelRunner._extract_mm_kwargs(runner,
        SimpleNamespace(scheduled_new_reqs=requests[:1]))
    result = model(torch.tensor([1]), torch.tensor([0]), **single)
    assert result.shape == (1, 1, 9, 2)
    assert result.unique().tolist() == [3.0]
    assert len(calls) == 1 and calls[0][1] == 2
