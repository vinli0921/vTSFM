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
