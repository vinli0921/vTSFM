"""Univariate quantile IO helpers, with settings supplied by each adapter."""

import base64
import binascii
from dataclasses import dataclass
from numbers import Real

_FP32_MAX = 3.402823466e38
_TARGET_ERROR = 'target must be a finite univariate FP32 series'
_B64_KEYS = {'data', 'shape', 'item_ids'}
_B64_DATA_ERROR = 'inputs_b64.data must be base64 little-endian FP32 values matching shape'


@dataclass(frozen=True)
class ForecastSettings:
    max_context: int
    max_horizon: int
    quantiles: tuple[float, ...]
    default_horizon: int
    default_quantiles: tuple[float, ...]


def _check_target_types(target):
    """Reject booleans, strings and other non-numbers before NumPy can coerce them."""
    types = set(map(type, target))
    if not types <= {int, float} and any(issubclass(t, bool) or not issubclass(t, Real) for t in types):
        raise ValueError(_TARGET_ERROR)


def _target_array(target, settings: ForecastSettings):
    """Validate one target list and return it as an FP32 NumPy array.

    One C-level pass collects element types, so booleans, strings and other
    non-numbers are rejected before NumPy could coerce them (it would accept
    ``'1.5'`` or ``True``). JSON produces only int/float; other ``Real`` types,
    such as NumPy scalars from offline callers, remain accepted as before.
    """
    import numpy as np

    if not isinstance(target, list) or not 1 <= len(target) <= settings.max_context:
        raise ValueError(f'target must contain 1 to {settings.max_context} observations')
    _check_target_types(target)
    try:
        values = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):  # e.g. integers beyond float64
        raise ValueError(_TARGET_ERROR) from None
    # NaN/inf fail the comparison, so this also enforces finiteness.
    if values.shape != (len(target),) or not np.all(np.abs(values) <= _FP32_MAX):
        raise ValueError(_TARGET_ERROR)
    return values.astype(np.float32)


def _b64_inputs(encoded, settings: ForecastSettings, limits=None):
    """Decode the compact equal-length form into ``parse_request`` input dicts.

    ``data`` is base64 of little-endian FP32 values in row-major
    ``[rows, width]`` order. Values must be finite, exactly as for JSON lists;
    there is no missing-value encoding. With ``limits``, the declared row count
    is checked first, with the admission message, so an oversized request is
    rejected before anything is decoded or expanded. The padded base64 length
    is exact, so a mismatched payload is also rejected before decoding.
    """
    import numpy as np

    if not isinstance(encoded, dict) or not {'data', 'shape'} <= set(encoded) or set(encoded) - _B64_KEYS:
        raise ValueError('inputs_b64 requires data and shape, with optional item_ids')
    shape = encoded['shape']
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(size) is not int for size in shape) or shape[0] < 1):
        raise ValueError('inputs_b64.shape must be [rows, width] with at least one row')
    rows, width = shape
    if limits is not None:
        limits.validate_request_rows(rows)
    if not 1 <= width <= settings.max_context:
        raise ValueError(f'target must contain 1 to {settings.max_context} observations')
    item_ids = encoded.get('item_ids')
    if item_ids is not None:
        if not isinstance(item_ids, list) or len(item_ids) != rows:
            raise ValueError('inputs_b64.item_ids must contain one item_id per row')
        if not all(isinstance(item_id, str) for item_id in item_ids):
            raise ValueError('item_id must be a string')
    data = encoded['data']
    size = rows * width * 4
    if not isinstance(data, str) or len(data) != 4 * ((size + 2) // 3):
        raise ValueError(_B64_DATA_ERROR)
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):  # Invalid alphabet/padding, or non-ASCII text.
        raise ValueError(_B64_DATA_ERROR) from None
    if len(raw) != size:
        raise ValueError(_B64_DATA_ERROR)
    # astype makes one owned, native-order array; rows below are views of it.
    values = np.frombuffer(raw, dtype='<f4').astype(np.float32).reshape(rows, width)
    # Finite FP32 values are always within FP32 range.
    if not np.isfinite(values).all():
        raise ValueError(_TARGET_ERROR)
    if item_ids is None:
        return [{'target': row} for row in values]
    return [{'target': row, 'item_id': item_id} for row, item_id in zip(values, item_ids)]


def encode_inputs_b64(targets, item_ids=None):
    """Build an ``inputs_b64`` object for equal-length series (a client helper).

    Values are cast the way the server casts JSON lists (float64, then FP32),
    so both request forms stage identical tensors.
    """
    import numpy as np

    try:
        rows = [list(row) for row in targets]
    except TypeError:  # A row that is not a sequence of values.
        raise ValueError('inputs_b64 requires a [rows, width] array of series') from None
    if not rows or not rows[0]:
        raise ValueError('inputs_b64 requires a [rows, width] array of series')
    if len({len(row) for row in rows}) != 1:
        raise ValueError('inputs_b64 requires equal-length series')
    # The JSON path's element checks, before NumPy could coerce '1.5' or True.
    for row in rows:
        _check_target_types(row)
    values = np.asarray(rows, dtype=np.float64)
    encoded = {'data': base64.b64encode(values.astype('<f4').tobytes()).decode('ascii'),
               'shape': list(values.shape)}
    if item_ids is not None:
        encoded['item_ids'] = list(item_ids)
    return encoded


def parse_request(data, settings: ForecastSettings, limits=None):
    """Validate a forecast request and return ``(inputs, horizon, levels)``.

    Series arrive either as JSON lists in ``inputs`` (the default) or as one
    compact equal-length ``inputs_b64`` object; see ``_b64_inputs``. Optional
    ``limits`` (``ForecastLimits``) bounds the compact form's rows before
    decoding; callers still run full admission on the parsed request.
    ``inputs`` are new dicts (the caller's data is not modified) whose
    ``target`` is the validated 1-D FP32 NumPy array, ready for tensor staging.
    """
    if not isinstance(data, dict) or set(data) - {'inputs', 'inputs_b64', 'parameters'}:
        raise ValueError('Expected inputs or inputs_b64, and optional parameters')
    if 'inputs' in data and 'inputs_b64' in data:
        raise ValueError('Send either inputs or inputs_b64, not both')
    compact = 'inputs_b64' in data
    inputs = data.get('inputs')
    if not compact and (not isinstance(inputs, list) or not inputs):
        raise ValueError('inputs must be a nonempty list')
    parameters = data.get('parameters', {})
    if not isinstance(parameters, dict) or set(parameters) - {'prediction_length', 'quantile_levels'}:
        raise ValueError('Only prediction_length and quantile_levels are supported')
    horizon = parameters.get('prediction_length', settings.default_horizon)
    if type(horizon) is not int or not 1 <= horizon <= settings.max_horizon:
        raise ValueError(f'prediction_length must be an integer in [1,{settings.max_horizon}]')
    levels = parameters.get('quantile_levels', list(settings.default_quantiles))
    if (not isinstance(levels, list) or not levels
            or any(type(q) not in (int, float) or q not in settings.quantiles for q in levels)
            or levels != sorted(set(levels))):
        raise ValueError(f'quantile_levels must be increasing, unique values from {settings.quantiles}')
    if compact:
        return _b64_inputs(data['inputs_b64'], settings, limits), horizon, levels
    parsed = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) - {'target', 'item_id'}:
            raise ValueError('Each input supports only target and item_id')
        target = _target_array(item.get('target'), settings)
        if 'item_id' in item and not isinstance(item['item_id'], str):
            raise ValueError('item_id must be a string')
        parsed.append({**item, 'target': target})
    return parsed, horizon, levels


def format_predictions(quantiles, item_ids, levels, horizon, settings: ForecastSettings):
    """Format CPU ``[rows, native quantiles, horizon]`` NumPy/Torch forecasts as JSON."""
    import numpy as np

    if tuple(quantiles.shape) != (len(item_ids), len(settings.quantiles), horizon):
        raise ValueError(f'Unexpected forecast shape: {quantiles.shape}')
    # Check only the requested quantiles, in one vectorized pass.
    selected = quantiles[:, [settings.quantiles.index(q) for q in levels]]
    if not np.isfinite(np.asarray(selected)).all():
        raise ValueError('Model returned nonfinite forecasts')
    keys = [str(q) for q in levels]
    predictions = []
    for item_id, rows in zip(item_ids, selected.tolist()):
        prediction = dict(zip(keys, rows))
        if item_id is not None:
            prediction['item_id'] = item_id
        predictions.append(prediction)
    return {'predictions': predictions}
