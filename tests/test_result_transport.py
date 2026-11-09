"""Stateless result metadata and exact bytes through the pooler tensor boundary."""

import json
import struct
import subprocess
import sys

import pytest

from vtsfm.adapters.result_transport import (
    MAX_METADATA_BYTES,
    encode_result_metadata,
    pack_result,
    unpack_result,
)


def test_result_metadata_accepts_unicode_ids_and_is_bounded_before_execution():
    encoded = encode_result_metadata(('sensor', None, '电机'), 5, (0.1, 0.5, 0.9))
    assert json.loads(encoded) == {
        'item_ids': ['sensor', None, '电机'], 'horizon': 5,
        'quantile_levels': [0.1, 0.5, 0.9],
    }
    with pytest.raises(ValueError, match='1 MiB'):
        encode_result_metadata(('x' * MAX_METADATA_BYTES,), 5, (0.5,))


@pytest.mark.parametrize('items,horizon,levels', [
    ([], 5, [0.5]), ([2], 5, [0.5]), ('sensor', 5, [0.5]),
    (['sensor'], 0, [0.5]), (['sensor'], True, [0.5]),
    (['sensor'], 2 ** 32, [0.5]), (['sensor'], 5, []),
    (['sensor'], 5, [True]), (['sensor'], 5, [float('nan')]),
    (['sensor'], 5, [0.5, 0.5]), (['sensor'], 5, [0.9, 0.1]),
])
def test_invalid_response_metadata_fails_without_a_runtime(items, horizon, levels):
    with pytest.raises(ValueError):
        encode_result_metadata(items, horizon, levels)


def test_result_transport_import_stays_cpu_safe():
    subprocess.run([sys.executable, '-c',
                    'import sys; import vtsfm.adapters.result_transport; '
                    'assert not {"torch", "vllm", "timesfm", "toto2", "uni2ts"} '
                    '& sys.modules.keys()'], check=True)


@pytest.fixture(scope='module')
def cuda():
    try:
        import torch
    except ImportError:
        pytest.fail('Result transport GPU checks require Torch')
    assert torch.cuda.is_available(), 'Result transport GPU checks require a working CUDA GPU'
    yield torch, torch.device('cuda:0')
    torch.cuda.synchronize()


@pytest.mark.gpu
class TestResultTransportCUDA:
    def test_ragged_crop_preserves_exact_fp32_bits_and_metadata(self, cuda):
        torch, device = cuda
        # Include signed zero, subnormals, infinities and NaN payload bits. The
        # transport must preserve bytes; forecast validity is checked by IO.
        bits = torch.tensor([0, -2147483648, 1, 2139095039, 2139095040,
                             -8388608, 2143289345], dtype=torch.int32, device=device)
        output = bits.repeat(36).view(torch.float32)[:3 * 9 * 9].reshape(3, 9, 9)[:, :, :5]
        assert not output.is_contiguous()
        metadata = encode_result_metadata(('a', None, '电机'), 5, (0.1, 0.9))

        wire = pack_result(output, metadata)
        assert wire.dtype == torch.uint8 and wire.device == device and wire.ndim == 1
        host = wire.cpu()  # vLLM's pooler output transfer boundary
        decoded, items, horizon, levels = unpack_result(host)

        assert (items, horizon, levels) == (('a', None, '电机'), 5, (0.1, 0.9))
        assert torch.equal(decoded.view(torch.int32), output.cpu().contiguous().view(torch.int32))
        assert decoded.untyped_storage().data_ptr() == host.untyped_storage().data_ptr()
        assert decoded.storage_offset() > 0

    def test_interleaved_result_shapes_keep_independent_response_settings(self, cuda):
        torch, device = cuda
        cases = [(('a',), 3, (0.5,)), ((None, 'b'), 7, (0.1, 0.9))]
        wires = []
        for index, (items, horizon, levels) in enumerate(cases):
            output = torch.full((len(items), 9, horizon), index + 1., device=device)
            wires.append(pack_result(output, encode_result_metadata(items, horizon, levels)).cpu())
        for index in (1, 0, 1):
            output, items, horizon, levels = unpack_result(wires[index])
            assert (items, horizon, levels) == cases[index]
            assert output.shape == (len(items), 9, horizon)
            assert torch.all(output == index + 1.)

    @pytest.mark.parametrize('field,value', [
        (0, b'badmagic'), (1, 2), (2, 0), (2, MAX_METADATA_BYTES + 1),
        (3, 0), (3, 2), (4, 0), (5, 0), (5, 2 ** 32 - 1),
    ])
    def test_rejects_corrupted_version_lengths_and_dimensions(self, cuda, field, value):
        torch, device = cuda
        wire = pack_result(torch.ones((1, 9, 5), device=device),
                           encode_result_metadata(('a',), 5, (0.5,))).cpu()
        header = struct.Struct('<8s5I')
        fields = list(header.unpack(bytes(wire[:header.size].tolist())))
        fields[field] = value
        wire[:header.size] = torch.tensor(list(header.pack(*fields)), dtype=torch.uint8)
        with pytest.raises(ValueError):
            unpack_result(wire)

    def test_rejects_truncation_trailing_bytes_invalid_json_and_header_mismatch(self, cuda):
        torch, device = cuda
        wire = pack_result(torch.ones((1, 9, 5), device=device),
                           encode_result_metadata(('a',), 5, (0.5,))).cpu()
        for invalid in (wire[:3], wire[:-1], torch.cat((wire, torch.zeros(1, dtype=torch.uint8)))):
            with pytest.raises(ValueError):
                unpack_result(invalid)
        bad_json = wire.clone()
        bad_json[28] = ord('!')
        with pytest.raises(ValueError, match='metadata'):
            unpack_result(bad_json)
        # Change only JSON's horizon, preserving envelope size and header.
        raw = bytes(wire.tolist()).replace(b'"horizon":5', b'"horizon":4')
        with pytest.raises(ValueError, match='metadata'):
            unpack_result(torch.tensor(list(raw), dtype=torch.uint8))

    def test_rejects_wrong_dtype_device_shape_and_unaligned_views(self, cuda):
        torch, device = cuda
        output = torch.ones((1, 9, 5), device=device)
        metadata = encode_result_metadata(('a',), 5, (0.5,))
        for invalid in (output.double(), output.squeeze(0), output[:, :, :4]):
            with pytest.raises(ValueError):
                pack_result(invalid, metadata)
        wire = pack_result(output, metadata)
        host = wire.cpu()
        unaligned = torch.cat((torch.zeros(1, dtype=torch.uint8), host))[1:]
        for invalid in (wire, host.float(), host[::2], host[None, :], unaligned):
            with pytest.raises(ValueError):
                unpack_result(invalid)
