"""Request identity and ragged payload boundaries survive field reduction."""

import json
import subprocess
import sys

import numpy as np
import pytest

from vtsfm.adapters.transport import decode_requests, encode_metadata


def metadata(name='a', lengths=(3,), items=('sensor',), horizon=2, levels=(0.1, 0.9)):
    return np.frombuffer(encode_metadata(name, items, lengths, horizon, levels), dtype=np.uint8)


def test_ragged_requests_keep_identity_lengths_and_settings():
    contexts = [np.array([[1, 2, 3]], dtype='float32'),
                np.array([[np.nan, 20, 30, 40], [50, 60, 70, 80]], dtype='float32')]
    encoded = [metadata(), metadata('b', (3, 4), (None, '电机'), horizon=5, levels=(0.5,))]
    a, b = decode_requests(contexts, encoded)
    assert a.context is contexts[0] and b.context is contexts[1]
    assert (a.request_id, a.item_ids, a.context_lengths, a.horizon, a.quantile_levels) == (
        'a', ('sensor',), (3,), 2, (0.1, 0.9))
    assert (b.request_id, b.item_ids, b.context_lengths, b.horizon, b.quantile_levels) == (
        'b', (None, '电机'), (3, 4), 5, (0.5,))


def test_equal_shapes_keep_separate_requests_even_for_identical_values():
    contexts = np.ones((2, 1, 3), dtype='float32')
    encoded = np.stack([metadata('a'), metadata('b')])
    requests = decode_requests(contexts, encoded)
    assert [x.request_id for x in requests] == ['a', 'b']
    assert all(x.context.shape == (1, 3) for x in requests)


def test_duplicate_request_ids_and_truncated_batches_fail():
    contexts = np.ones((2, 1, 3), dtype='float32')
    with pytest.raises(ValueError, match='Duplicate'):
        decode_requests(contexts, [metadata(), metadata()])
    with pytest.raises(ValueError, match='count'):
        decode_requests(contexts, [metadata()])


@pytest.mark.parametrize('change', [
    {'version': 2}, {'request_id': ''}, {'request_id': None}, {'context_lengths': [4]},
    {'context_lengths': [True]}, {'item_ids': []}, {'item_ids': [4]},
    {'horizon': 0}, {'horizon': True}, {'quantile_levels': [0.9, 0.1]},
    {'quantile_levels': [float('nan')]}, {'unexpected': 1},
])
def test_malformed_metadata_is_rejected(change):
    raw = json.loads(encode_metadata('a', ['sensor'], [3], 2, [0.1, 0.9]))
    raw.update(change)
    encoded = np.frombuffer(json.dumps(raw).encode(), dtype=np.uint8)
    with pytest.raises((ValueError, TypeError)):
        decode_requests(np.ones((1, 1, 3), dtype='float32'), [encoded])


@pytest.mark.parametrize('context', [np.ones((1, 3), dtype='float64'), np.ones((1, 3, 1), dtype='float32')])
def test_invalid_context_layout_rejected(context):
    with pytest.raises(ValueError):
        decode_requests([context], [metadata()])


def test_import_stays_cpu_safe():
    subprocess.run([sys.executable, '-c',
                    'import sys; import vtsfm.adapters.transport; '
                    'assert not {"torch", "vllm", "timesfm", "toto2", "uni2ts"} & sys.modules.keys()'],
                   check=True)
