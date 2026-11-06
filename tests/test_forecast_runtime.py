"""Opt-in checkpoint parity checks; run in the selected adapter's environment."""

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ADAPTER = os.environ.get('VTSFM_TEST_ADAPTER')
pytestmark = pytest.mark.skipif(ADAPTER not in ('timesfm25', 'toto2', 'moirai2'),
                              reason='Set VTSFM_TEST_ADAPTER to run checkpoint inference')


def test_pooling_matches_official_forecast(monkeypatch):
    monkeypatch.setenv('VTSFM_ADAPTER', ADAPTER)
    monkeypatch.setenv('VLLM_PLUGINS', 'vtsfm')
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from vtsfm.adapters import get_adapter
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.engine.arg_utils import EngineArgs

    torch.set_num_threads(2)
    spec = get_adapter(ADAPTER)
    directory = os.environ.get('VTSFM_TEST_MODEL_PATH') or snapshot_download(
        spec.model_id, revision=spec.model_revision, allow_patterns=['config.json', 'model.safetensors'])
    model_config = EngineArgs(
        model=directory, config_format='vtsfm', runner='pooling', dtype='float32',
        skip_tokenizer_init=True, enforce_eager=True, max_model_len=8192,
        hf_overrides=json.loads(spec.vllm_args[spec.vllm_args.index('--hf-overrides') + 1]),
        io_processor_plugin='vtsfm', enable_mm_embeds=True,
    ).create_model_config()
    module, name = spec.model_class.split(':')
    wrapper = getattr(importlib.import_module(module), name)(vllm_config=SimpleNamespace(
        model_config=model_config))
    wrapper.load_weights(load_file(str(Path(directory) / 'model.safetensors')).items())
    wrapper.eval()
    io_module, io_name = spec.io_processor_class.rsplit('.', 1)
    io = getattr(importlib.import_module(io_module), io_name)(None, None)
    inputs = np.stack([np.sin(np.arange(128) / 7), np.cos(np.arange(128) / 11)]).astype('float32')
    horizon = {'timesfm25': 256, 'toto2': 800, 'moirai2': 160}[ADAPTER]

    if ADAPTER == 'timesfm25':
        import timesfm
        reference = timesfm.TimesFM_2p5_200M_torch.from_pretrained(directory, torch_compile=False)
        reference.compile(timesfm.ForecastConfig(
            max_context=128, max_horizon=horizon, per_core_batch_size=2,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        expected = reference.forecast(horizon=horizon, inputs=list(inputs))[1][..., 1:].transpose(0, 2, 1)
    elif ADAPTER == 'toto2':
        from toto2 import Toto2Model
        reference = Toto2Model.from_pretrained(directory).eval()
        target = torch.from_numpy(inputs).unsqueeze(1)
        expected = reference.forecast(
            {'target': target, 'target_mask': torch.ones_like(target, dtype=torch.bool),
             'series_ids': torch.zeros(2, 1, dtype=torch.long)},
            horizon=horizon, decode_block_size=768)[:, :, 0, :].permute(1, 0, 2).numpy()
    else:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        reference = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(directory), prediction_length=horizon,
            context_length=128, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        expected = reference.predict(list(inputs))

    data = {'inputs': [{'target': row.tolist(), 'item_id': str(i)} for i, row in enumerate(inputs)],
            'parameters': {'prediction_length': horizon, 'quantile_levels': [0.1, 0.5, 0.9]}}
    prepared = io.pre_process(io.parse_data(data), request_id='smoke')
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    parsed = processor.info.parse_mm_data(prepared['multi_modal_data'])
    transported = processor([1], parsed)
    assert transported['prompt_token_ids'] == [1]
    output = wrapper(torch.tensor([1]), torch.tensor([0]), **transported['mm_kwargs'].get_data())
    np.testing.assert_allclose(output[0].cpu().numpy(), expected, rtol=1e-4, atol=1e-4)
    response = io.post_process([SimpleNamespace(outputs=SimpleNamespace(data=output))], request_id='smoke')
    assert [row['item_id'] for row in response['predictions']] == ['0', '1']
    assert len(response['predictions'][0]['0.5']) == horizon
    assert not io.pending
    dummy = processor.get_dummy_mm_inputs({'timeseries': 1})
    tokens = torch.tensor(dummy['prompt_token_ids'])
    assert wrapper(tokens, torch.zeros_like(tokens), **dummy['mm_kwargs'].get_data()).shape == (8192, 1, 9, 1)


@pytest.fixture(scope='module')
def timesfm_runtime():
    import torch
    from model_integration_support import NativeForecaster

    from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime

    native = NativeForecaster('timesfm25')
    device = native.device
    runtime = TimesFMRuntime({})
    runtime.model.load_state_dict(native.reference.model.state_dict())
    runtime.model.to(device=device, dtype=torch.float32).eval()
    return runtime, native


@pytest.mark.gpu
@pytest.mark.skipif(ADAPTER != 'timesfm25', reason='Requires the TimesFM CUDA environment')
def test_timesfm_forecast_stays_on_cuda_and_matches_boundary_cases(monkeypatch, timesfm_runtime):
    import torch

    runtime, native = timesfm_runtime
    device = native.device
    constant = [3.0] * 31
    positive = [1 + step / 19 for step in range(32)]
    signed = [(step - 16) / 13 for step in range(33)]
    first = ([constant], 127)
    ragged = ([constant, positive, signed], 129)
    exact_boundary = ([positive], 128)

    def forbid_host_conversion(*args, **kwargs):
        pytest.fail('The TimesFM tensor forecast tried to move results or inputs to the host')

    # A→B→A exercises reuse after changing the series count, padded context
    # and horizon. The other case hits the patch and block sizes exactly.
    for histories, horizon in (first, ragged, first, exact_boundary):
        data = {
            'inputs': [{'target': history} for history in histories],
            'parameters': {'prediction_length': horizon},
        }
        expected = torch.as_tensor(native.predict_array(data), device=device)
        width = max(map(len, histories))
        context = torch.full((len(histories), width), float('nan'), device=device)
        for row, history in zip(context, histories):
            row[-len(history):] = torch.as_tensor(history, device=device)

        with monkeypatch.context() as inference:
            inference.setattr(torch.Tensor, 'cpu', forbid_host_conversion)
            inference.setattr(torch.Tensor, 'numpy', forbid_host_conversion)
            actual = runtime.forecast(context, horizon)

        assert actual.device == device
        assert actual.dtype == torch.float32
        assert actual.shape == (len(histories), 9, horizon)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(ADAPTER != 'timesfm25', reason='Requires the TimesFM CUDA environment')
def test_timesfm_prepared_batch_preserves_requested_quantiles_and_request_slices(timesfm_runtime):
    from dataclasses import replace

    import torch

    from vtsfm.adapters.timesfm25.spec import TIMESFM25
    from vtsfm.batching import BatchKey, BatchRequest, plan_batches
    from vtsfm.execution import PreparedBatch, PreparedRequest

    runtime, native = timesfm_runtime
    device = native.device
    capabilities = TIMESFM25.execution_capabilities
    width, length, horizon = 64, 33, 129
    levels = (0.2, 0.8)
    key = BatchKey(TIMESFM25.name, TIMESFM25.model_id, TIMESFM25.model_revision,
                   width, horizon, levels, execution_options=capabilities.execution_options)
    first = [[2.0] * length, [3.0 + step / 19 for step in range(length)]]
    second = [[(step - 16) / 13 for step in range(length)]]
    series = {'first': first, 'second': second}
    requests = []
    for request_id, histories in series.items():
        count = len(histories)
        context = torch.full((count, width), -1000.0, device=device)
        context[:, -length:] = torch.as_tensor(histories, device=device)
        observed = torch.zeros_like(context, dtype=torch.bool)
        observed[:, -length:] = True
        item_ids = tuple(f'{request_id}-{row}' for row in range(count))
        requests.append(PreparedRequest(BatchRequest(request_id, key, (1,) * count),
            context, (length,) * count, observed, item_ids, capabilities))

    plan, = plan_batches((request.metadata for request in requests), max_rows=3, max_requests=2)
    result = runtime.execute(PreparedBatch(plan, tuple(requests))).split()
    assert tuple(result) == ('first', 'second')
    for request in requests:
        request_id = request.metadata.request_id
        forecast = result[request_id]
        expected = native.predict_array({
            'inputs': [{'target': history} for history in series[request_id]],
            'parameters': {'prediction_length': horizon},
        })
        expected = torch.as_tensor(expected[:, [1, 7], :], device=device)
        assert forecast.item_ids == request.item_ids
        assert forecast.quantile_levels == levels
        assert forecast.values.device == device
        torch.testing.assert_close(forecast.values, expected, rtol=1e-4, atol=1e-4)

    grouped_capabilities = replace(capabilities, supports_grouped_rows=True)
    first = requests[0]
    grouped = replace(first, metadata=BatchRequest('first', key, (2,)),
                      capabilities=grouped_capabilities)
    grouped_plan, = plan_batches([grouped.metadata], max_rows=2, max_requests=1)
    with pytest.raises(ValueError, match='TimesFM execution capabilities'):
        runtime.execute(PreparedBatch(grouped_plan, (grouped,)))
