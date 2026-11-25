"""Native Moirai quantile recursion, batch routing and request-state lifetime."""
import gc
import os
import weakref

import pytest


@pytest.fixture
def model(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('uni2ts')
    from vtsfm.adapters.moirai2.runtime import MoiraiRuntime

    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')
    monkeypatch.setenv('VTSFM_PRECISION', 'fp32')
    monkeypatch.setenv('VTSFM_MOIRAI_ATTENTION', 'upstream')
    torch.manual_seed(21)
    runtime = MoiraiRuntime(dict(d_model=64, d_ff=128, num_layers=1, patch_size=16,
        num_predict_token=4, max_seq_len=128, attn_dropout_p=0.0, dropout_p=0.0, scaling=True))
    device = 'cuda' if os.environ.get('VTSFM_TEST_ADAPTER') == 'moirai2' else 'cpu'
    runtime.model.to(device).eval()
    return runtime


@pytest.mark.parametrize('horizon', [1, 64, 65, 96, 129, 1024])
def test_repeated_rounds_match_native(model, horizon):
    import torch
    from vtsfm.adapters.moirai2.stepping import MoiraiStepper

    device = next(model.model.parameters()).device
    history = torch.sin(torch.arange(134, device=device).reshape(2, 67) / 8)
    history[0, :14] = float('nan')
    expected = model.forecast(history, horizon)
    stepper = MoiraiStepper(model)
    state = stepper.initialize(history, horizon)
    assert state.blocks_done == 0
    stepper.prefill([state])
    while not stepper.complete(state):
        stepper.advance([state])
    assert state.blocks_done == (horizon + 63) // 64
    out = stepper.finalize(state)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
    assert out.untyped_storage().nbytes() == out.numel() * out.element_size()
    stepper.release(state)
    stepper.release(state)
    assert state.tensor_bytes == state.cache_bytes == 0


def test_shared_round_survives_peer_release_and_new_arrival(model):
    import torch
    from vtsfm.adapters.moirai2.stepping import MoiraiStepper

    device = next(model.model.parameters()).device
    x = torch.randn(3, 48, device=device)
    x[0, :5] = float('nan')
    y = torch.randn(1, 48, device=device) + 3
    expected, new_expected = model.forecast(x[1:], 143), model.forecast(y, 63)
    stepper = MoiraiStepper(model)
    cancelled, survivor = stepper.initialize(x[:1], 143), stepper.initialize(x[1:], 143)
    stepper.prefill([cancelled, survivor])
    stepper.advance([survivor, cancelled])
    ref = weakref.ref(cancelled.inputs[0])
    enabled = gc.isenabled()
    try:
        gc.disable()
        stepper.release(cancelled)
        assert ref() is None
    finally:
        if enabled:
            gc.enable()
    joining = stepper.initialize(y, 63)
    stepper.prefill([joining])
    stepper.advance([survivor])
    torch.testing.assert_close(stepper.finalize(survivor), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(stepper.finalize(joining), new_expected, atol=1e-4, rtol=1e-4)
    for state in [joining, survivor]:
        stepper.release(state)


@pytest.mark.skipif(not os.environ.get('VTSFM_MOIRAI_BLOCK_CHECKPOINT'),
                    reason='Set VTSFM_MOIRAI_BLOCK_CHECKPOINT=cpu or cuda in the pinned model environment')
def test_real_checkpoint_shared_recursion_against_independent_native():
    """An actual weight-parity check that can run with CUDA hidden as a separate CPU control."""
    import json
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from vtsfm.adapters.moirai2.runtime import MoiraiRuntime
    from vtsfm.adapters.moirai2.spec import MODEL_ID, MODEL_REVISION
    from vtsfm.adapters.moirai2.stepping import MoiraiStepper

    device = os.environ['VTSFM_MOIRAI_BLOCK_CHECKPOINT']
    assert device in ('cpu', 'cuda')
    snap = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                  allow_patterns=['config.json', 'model.safetensors']))
    model = MoiraiRuntime(json.loads((snap/'config.json').read_text()))
    model.model.load_state_dict(load_file(str(snap/'model.safetensors')), strict=True)
    model.model.to(device).eval()
    stepper = MoiraiStepper(model)
    ref_model = Moirai2Module.from_pretrained(snap).to(device).eval()
    histories = [torch.sin(torch.arange(61) / 17), torch.cos(torch.arange(63) / 13)]
    with torch.inference_mode():
        states = [stepper.initialize(x.to(device)[None], 147) for x in histories]
        stepper.prefill(states)
        while not all(stepper.complete(s) for s in states):
            stepper.advance(states)
        for x, state in zip(histories, states, strict=True):
            reference = Moirai2Forecast(module=ref_model, context_length=len(x), prediction_length=147,
                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(device).eval()
            official = torch.from_numpy(reference.predict([x.numpy()])).to(device)
            torch.testing.assert_close(stepper.finalize(state), official, atol=1e-4, rtol=1e-4)
            stepper.release(state)
