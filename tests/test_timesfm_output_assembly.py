"""Manual real-checkpoint test for forecasts retained across peer completion and cancellation."""

import json

import test_timesfm_stepping as stepping


pytestmark = stepping.pytestmark
runtime = stepping.runtime


def test_completed_forecasts_keep_only_their_own_output_storage(runtime, tmp_path):
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef
    from model_integration_support import NativeForecaster, QUANTILES
    from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

    # All requests pad to C64, but the late H129 owner joins a later decode position.
    cases = {
        'long': (stepping._context((49, 64), -1.5), 1024),
        'cancelled': (stepping._context((63,), -3.), 1024),
        'single': (stepping._context((52,), 2.), 96),
        'partial': (stepping._context((47, 62), .5), 129),
    }
    native = NativeForecaster('timesfm25')
    expected = {}
    for name in ('single', 'partial', 'long'):
        context, horizon = cases[name]
        payload = {'inputs': [{'target': row[torch.isfinite(row)].tolist()} for row in context.cpu()],
                   'parameters': {'prediction_length': horizon, 'quantile_levels': list(QUANTILES)}}
        expected[name] = torch.from_numpy(native.predict_array(payload))
    del native

    stepper = TimesFMStepper(runtime)
    states = {name: stepper.initialize(context, horizon) for name, (context, horizon) in cases.items()}
    stepper.prefill([states['long'], states['cancelled'], states['single']])
    held = {'single': stepper.finalize(states['single'])}
    stepper.release(states['single'])
    stepper.advance([states['cancelled'], states['long']])
    stepper.prefill([states['partial']])
    mixed = [states[name] for name in ('partial', 'long', 'cancelled')]
    positions = [state.cache_position for state in mixed]
    assert len(set(positions)) == 2
    stepper.advance(mixed)
    held['partial'] = stepper.finalize(states['partial'])
    stepper.release(states['partial'])

    # Keep both finished forecasts alive while cancelling a peer. Watch actual
    # storage for its state and pass tensors, including feedback and output-side buffers.
    cancelled = states['cancelled']
    buffers = {StorageWeakRef(value.untyped_storage()) for owner in (cancelled, *cancelled.passes)
               for value in vars(owner).values() if torch.is_tensor(value)}
    assert buffers and not any(buffer.expired() for buffer in buffers)
    stepper.release(cancelled)
    assert all(buffer.expired() for buffer in buffers)
    while not stepper.complete(states['long']):
        stepper.advance([states['long']])
    held['long'] = stepper.finalize(states['long'])
    stepper.release(states['long'])

    errors = {}
    assert len({result.untyped_storage().data_ptr() for result in held.values()}) == len(held)
    for name, result in held.items():
        assert result.untyped_storage().nbytes() == result.numel() * result.element_size()
        actual = result.cpu()
        torch.testing.assert_close(actual, expected[name], rtol=1e-4, atol=1e-4)
        errors[name] = float((actual - expected[name]).abs().max())
    (tmp_path / 'output-ownership.json').write_text(json.dumps({
        'synthetic': True, 'decode_positions': positions, 'released_peer_storages': len(buffers),
        'output_bytes': {name: output.untyped_storage().nbytes() for name, output in held.items()},
        'max_abs_error': errors, 'rtol': 1e-4, 'atol': 1e-4, 'status': 'passed',
    }, indent=2) + '\n')
