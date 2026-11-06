"""CPU-safe wire metadata for one whole-forecast request per placeholder.

vLLM owns field batching and host-to-device copies. Numerical contexts remain
separate, unnormalized [series, time] tensors, including their left NaN padding.
This boundary transports requests; it does not collate model execution.
"""

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class TransportRequest:
    request_id: str
    item_ids: tuple[str | None, ...]
    context_lengths: tuple[int, ...]
    horizon: int
    quantile_levels: tuple[float, ...]
    context: object


def _validate_metadata(data):
    expected = {'version', 'request_id', 'item_ids', 'context_lengths', 'horizon', 'quantile_levels'}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError('Invalid numerical request metadata fields')
    if type(data['version']) is not int or data['version'] != 1:
        raise ValueError('Unsupported numerical transport version')
    if not isinstance(data['request_id'], str) or not data['request_id']:
        raise ValueError('request_id must be a nonempty string')
    items, lengths = data['item_ids'], data['context_lengths']
    if (not isinstance(items, list) or not isinstance(lengths, list) or not lengths
            or len(items) != len(lengths)
            or any(x is not None and not isinstance(x, str) for x in items)
            or any(type(x) is not int or x < 1 for x in lengths)):
        raise ValueError('item IDs and positive context lengths must align')
    if type(data['horizon']) is not int or data['horizon'] < 1:
        raise ValueError('horizon must be a positive integer')
    levels = data['quantile_levels']
    if (not isinstance(levels, list) or not levels
            or any(type(x) not in (float, int) or not math.isfinite(x) or not 0 < x < 1 for x in levels)
            or levels != sorted(set(levels))):
        raise ValueError('quantile_levels must be increasing unique probabilities')


def encode_metadata(request_id, item_ids, context_lengths, horizon, quantile_levels) -> bytes:
    """Encode small CPU metadata; never serialize the numerical context here."""
    data = dict(version=1, request_id=request_id, item_ids=list(item_ids),
                context_lengths=list(context_lengths), horizon=horizon,
                quantile_levels=list(quantile_levels))
    _validate_metadata(data)
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def decode_requests(context, request_metadata) -> tuple[TransportRequest, ...]:
    """Read vLLM batched fields: stacked tensors if shapes match, lists otherwise.

    Contexts stay on their existing execution device. Metadata must stay on CPU
    (the processor sets keep_on_cpu=True); only this small byte tensor is read.
    The returned order is transport order, not a guarantee about pooling slots.
    """
    contexts, metadata = list(context), list(request_metadata)
    if not contexts or len(contexts) != len(metadata):
        raise ValueError('Context and metadata request count must match and be nonzero')
    requests, identities = [], set()
    for values, encoded in zip(contexts, metadata):
        if (str(getattr(encoded, 'device', 'cpu')) != 'cpu'
                or str(encoded.dtype).removeprefix('torch.') != 'uint8'
                or len(encoded.shape) != 1):
            raise ValueError('request_metadata must contain CPU uint8 byte vectors')
        try:
            data = json.loads(bytes(encoded.tolist()).decode('utf-8'))
        except (ValueError, UnicodeError, TypeError) as exc:
            raise ValueError('Invalid encoded numerical request metadata') from exc
        _validate_metadata(data)
        if (len(values.shape) != 2 or values.shape[0] != len(data['context_lengths'])
                or max(data['context_lengths']) != values.shape[1]
                or str(values.dtype).removeprefix('torch.') != 'float32'):
            raise ValueError('context must be FP32 [series, longest history] matching metadata')
        if data['request_id'] in identities:
            raise ValueError('Duplicate transported request_id')
        identities.add(data['request_id'])
        requests.append(TransportRequest(data['request_id'], tuple(data['item_ids']),
            tuple(data['context_lengths']), data['horizon'], tuple(data['quantile_levels']), values))
    return tuple(requests)
