"""Self-contained whole-forecast results across vLLM's tensor-only pooler IO.

One contiguous uint8 vector contains a versioned header, small JSON response
metadata, alignment padding, and the exact FP32 forecast bytes. This replaces
frontend pending maps: aborting a request requires no plugin cleanup callback.
Importing this module does not import Torch or initialize a model runtime.
"""

import json
import math
import struct


MAX_METADATA_BYTES = 1024 * 1024
_MAGIC = b'vTSFMOUT'
_VERSION = 1
# magic, version, metadata byte count, rows, native quantiles, horizon
_HEADER = struct.Struct('<8s5I')


def _validate_metadata(data):
    if (not isinstance(data, dict)
            or set(data) != {'item_ids', 'horizon', 'quantile_levels'}):
        raise ValueError('Invalid forecast result metadata fields')
    items = data['item_ids']
    if (not isinstance(items, list) or not items
            or any(item is not None and not isinstance(item, str) for item in items)):
        raise ValueError('Forecast result item IDs must be a nonempty list of strings or nulls')
    if type(data['horizon']) is not int or not 0 < data['horizon'] < 2 ** 32:
        raise ValueError('Forecast result horizon must be a positive 32-bit integer')
    levels = data['quantile_levels']
    if (not isinstance(levels, list) or not levels
            or any(type(level) not in (int, float)
                   or not 0 < level < 1 or not math.isfinite(level) for level in levels)
            or levels != sorted(set(levels))):
        raise ValueError('Forecast result quantiles must be increasing unique probabilities')


def encode_result_metadata(item_ids, horizon, quantile_levels) -> bytes:
    """Validate and bound response metadata before submitting a forecast.

    Call this during preprocessing as well as when preparing pooler routes, so
    an oversized item ID is rejected before executing the upstream model.
    """
    if isinstance(item_ids, (str, bytes)) or isinstance(quantile_levels, (str, bytes)):
        raise ValueError('Forecast result item IDs and quantiles must be sequences')
    data = {'item_ids': list(item_ids), 'horizon': horizon,
            'quantile_levels': list(quantile_levels)}
    _validate_metadata(data)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError('Forecast result metadata exceeds the 1 MiB limit')
    return encoded


def _decode_metadata(encoded):
    if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_METADATA_BYTES:
        raise ValueError('Invalid forecast result metadata length')
    try:
        data = json.loads(encoded.decode('utf-8'))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid encoded forecast result metadata') from exc
    _validate_metadata(data)
    return data


def pack_result(output, metadata: bytes):
    """Pack FP32 [series, native quantiles, horizon] on its existing device.

    Only the small CPU metadata moves to the output device. Forecast data never
    visits the host here. A strided pooler crop is made contiguous if necessary;
    concatenation then copies its exact bytes into the returned envelope.
    """
    import torch

    data = _decode_metadata(metadata)
    if (not isinstance(output, torch.Tensor) or output.dtype != torch.float32
            or output.ndim != 3 or any(not 0 < size < 2 ** 32 for size in output.shape)):
        raise ValueError('Forecast result must be FP32 [series, native quantiles, horizon]')
    rows, quantiles, horizon = output.shape
    if rows != len(data['item_ids']) or horizon != data['horizon']:
        raise ValueError('Forecast result shape does not match its response metadata')
    if len(data['quantile_levels']) > quantiles:
        raise ValueError('Forecast result has fewer native quantiles than requested')
    prefix = _HEADER.pack(_MAGIC, _VERSION, len(metadata), rows, quantiles, horizon) + metadata
    # Keep the FP32 payload's CPU storage offset divisible by four for view().
    prefix += b'\0' * (-len(prefix) % 4)
    header = torch.frombuffer(bytearray(prefix), dtype=torch.uint8).to(device=output.device)
    payload = output.contiguous().view(torch.uint8).reshape(-1)
    return torch.cat((header, payload))


def unpack_result(encoded):
    """Decode a completed CPU envelope and return a zero-copy FP32 payload view.

    Returns ``(output, item_ids, horizon, quantile_levels)``. vLLM performs the
    device-to-host transfer; this function rejects GPU tensors rather than
    introducing an implicit synchronization or copy in response formatting.
    """
    import torch

    if (not isinstance(encoded, torch.Tensor) or encoded.dtype != torch.uint8
            or encoded.device.type != 'cpu' or encoded.ndim != 1
            or not encoded.is_contiguous() or encoded.storage_offset() % 4):
        raise ValueError('Forecast result must be an aligned contiguous CPU uint8 vector')
    if encoded.numel() < _HEADER.size:
        raise ValueError('Truncated forecast result header')
    magic, version, metadata_size, rows, quantiles, horizon = _HEADER.unpack(
        bytes(encoded[:_HEADER.size].tolist()))
    if magic != _MAGIC or version != _VERSION:
        raise ValueError('Unsupported forecast result format or version')
    if not 0 < metadata_size <= MAX_METADATA_BYTES or not all((rows, quantiles, horizon)):
        raise ValueError('Invalid forecast result dimensions or metadata length')
    metadata_end = _HEADER.size + metadata_size
    payload_start = (metadata_end + 3) // 4 * 4
    expected_size = payload_start + rows * quantiles * horizon * 4
    if encoded.numel() != expected_size:
        raise ValueError('Forecast result byte count does not match its header')
    data = _decode_metadata(bytes(encoded[_HEADER.size:metadata_end].tolist()))
    if (rows != len(data['item_ids']) or horizon != data['horizon']
            or len(data['quantile_levels']) > quantiles):
        raise ValueError('Forecast result header does not match its response metadata')
    if any(encoded[metadata_end:payload_start].tolist()):
        raise ValueError('Invalid forecast result alignment padding')
    output = encoded[payload_start:].view(torch.float32).reshape(rows, quantiles, horizon)
    return output, tuple(data['item_ids']), horizon, tuple(data['quantile_levels'])
