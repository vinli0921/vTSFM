"""Checkpoint-free IO lifecycle checks in the pinned vLLM environment."""

from concurrent.futures import ThreadPoolExecutor
import gc
import os
from types import SimpleNamespace
import weakref

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_ROUTING_ENGINE') != '1',
    reason='Requires the pinned serving environment (no checkpoint needed)')]


def _processor():
    from vtsfm.adapters.timesfm25.io_processor import TimesFMIOProcessor

    return TimesFMIOProcessor(None, None)


def _payload(index):
    return {'inputs': [{'item_id': f'电机-{index}', 'target': [index, index + 1]}],
            'parameters': {'prediction_length': index % 3 + 1,
                           'quantile_levels': [0.2, 0.8] if index % 2 else [0.5]}}


def test_offline_and_reused_external_ids_get_unique_wire_identity():
    from vtsfm.adapters.transport import decode_metadata

    processor = _processor()
    ids = []
    for external in (None, None, 'same-header', 'same-header'):
        prompt = processor.pre_process(processor.parse_data(_payload(1)), request_id=external)
        encoded = prompt['multi_modal_data']['timeseries']['request_metadata'][0]
        ids.append(decode_metadata(encoded)['request_id'])
    assert len(set(ids)) == len(ids)
    assert not hasattr(processor, 'pending')


def test_postprocessing_is_independent_of_order_thread_and_external_id():
    import torch
    from vtsfm.adapters.result_transport import encode_result_metadata, pack_result

    processor = _processor()

    def prepare(index):
        data = _payload(index)
        _, horizon, levels = processor.parse_data(data)
        processor.pre_process(processor.parse_data(data), request_id='same-header')
        values = torch.arange(9, dtype=torch.float32)[None, :, None].expand(1, 9, horizon)
        encoded = pack_result(values, encode_result_metadata([f'电机-{index}'], horizon, levels))
        return index, SimpleNamespace(outputs=SimpleNamespace(data=encoded))

    outputs = [prepare(index) for index in range(32)]

    def finish(pair):
        index, output = pair
        response = processor.post_process([output])  # Offline passes no ID.
        horizon = index % 3 + 1
        levels = [0.2, 0.8] if index % 2 else [0.5]
        assert response['predictions'] == [dict(item_id=f'电机-{index}', **{
            str(q): [float(round(q * 10) - 1)] * horizon for q in levels})]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(finish, reversed(outputs)))
    assert not hasattr(processor, 'pending')


def test_abandoned_invalid_and_repeated_preprocess_do_not_retain_input_tensors():
    processor = _processor()
    initial_attributes = set(vars(processor))
    references = []
    for index in range(256):
        prompt = processor.pre_process(processor.parse_data(_payload(index)),
                                       request_id=f'abandoned-{index}')
        values = prompt['multi_modal_data']['timeseries']
        references.extend(weakref.ref(values[field][0]) for field in ('context', 'request_metadata'))
        del values, prompt  # Submission failed or was cancelled before postprocess.
        with pytest.raises(ValueError):
            processor.parse_data({'inputs': []})
        with pytest.raises(ValueError, match='one whole-forecast'):
            processor.post_process([])
    gc.collect()
    assert all(reference() is None for reference in references)
    assert set(vars(processor)) == initial_attributes
    assert not hasattr(processor, 'pending')


def test_oversized_result_metadata_fails_before_engine_submission():
    processor = _processor()
    data = _payload(1)
    data['inputs'][0]['item_id'] = 'x' * (1024 * 1024 + 1)
    with pytest.raises(ValueError):
        processor.pre_process(processor.parse_data(data))
