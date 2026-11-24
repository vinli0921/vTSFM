"""HTTP cases and independent native references for manual GPU checks.

Imports remain CPU-safe. Constructing NativeForecaster imports model packages and
downloads the pinned checkpoint in the selected adapter's Linux GPU environment.
"""

import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform


ADAPTERS = ('timesfm25', 'toto2', 'moirai2')
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PAYLOAD_ENV = 'VTSFM_TEST_PAYLOAD'
PAYLOAD_FORMS = ('json', 'b64')


def payload_form():
    """The HTTP request form selected by scripts/test-models.sh --payload."""
    form = os.environ.get(PAYLOAD_ENV, 'json')
    if form not in PAYLOAD_FORMS:
        raise ValueError(f'{PAYLOAD_ENV} must be one of {PAYLOAD_FORMS}')
    return form


def wire_data(data, form='json'):
    """Return ``(http_data, form_sent)`` for a case's JSON-list ``data``.

    ``b64`` encodes equal-length series as ``inputs_b64``. Ragged cases, and
    cases with item IDs on only some series, have no compact form and are
    sent as JSON lists; ``form_sent`` records which form each request used.
    Native references and parity checks keep using the original ``data``.
    """
    if form not in PAYLOAD_FORMS:
        raise ValueError(f'payload form must be one of {PAYLOAD_FORMS}')
    inputs = data['inputs']
    item_ids = [item.get('item_id') for item in inputs]
    if (form == 'json' or len({len(item['target']) for item in inputs}) != 1
            or len({item_id is None for item_id in item_ids}) != 1):
        return data, 'json'
    from vtsfm.adapters.forecast import encode_inputs_b64

    compact = {'inputs_b64': encode_inputs_b64([item['target'] for item in inputs],
                                               None if item_ids[0] is None else item_ids)}
    if 'parameters' in data:
        compact['parameters'] = data['parameters']
    return compact, 'b64'


def integration_cases(adapter):
    """Deterministic synthetic smoke inputs; these do not measure accuracy."""
    if adapter not in ADAPTERS:
        raise ValueError(f'Unsupported integration adapter: {adapter!r}')

    def signal(length, phase=0.0):
        return [2.0 + phase + 0.01 * step + math.sin(step / 7 + phase)
                + 0.2 * math.cos(step / 13) for step in range(length)]

    def case(name, inputs, horizon, levels):
        return {'name': f'synthetic-{name}', 'data': {
            'inputs': [{'item_id': item_id, 'target': target} for item_id, target in inputs],
            'parameters': {'prediction_length': horizon, 'quantile_levels': levels},
        }}

    first = case('single-short', [('synthetic-single', signal(128))], 17, [0.1, 0.5, 0.9])
    repeated = deepcopy(first)
    repeated['name'] = 'synthetic-single-repeat-after-shape-changes'
    return [first,
        case('ragged-sparse-quantiles', [('synthetic-ragged-a', signal(97)),
             ('synthetic-ragged-b', signal(163, 1.5))], 33, [0.2, 0.8]),
        case('block-crossing', [('synthetic-block', signal(128, -1.0))],
             {'timesfm25': 256, 'toto2': 800, 'moirai2': 160}[adapter], list(QUANTILES)),
        repeated]


def _finite_number(value, location):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AssertionError(f'{location}: expected a finite number, got {value!r}')


def _check_native_shape(expected, case):
    name = case['name']
    count = len(case['data']['inputs'])
    horizon = case['data']['parameters']['prediction_length']
    shape = f'[{count}, 9, {horizon}]'
    if not isinstance(expected, list) or len(expected) != count:
        raise AssertionError(f'{name}: native forecast shape must be {shape}')
    for index, row in enumerate(expected):
        if not isinstance(row, list) or len(row) != len(QUANTILES):
            raise AssertionError(f'{name}: native series {index} shape must be [9, {horizon}]')
        for quantile, values in zip(QUANTILES, row):
            if not isinstance(values, list) or len(values) != horizon:
                raise AssertionError(f'{name}: native series {index} quantile {quantile} '
                                     f'horizon must be {horizon}')
            for step, value in enumerate(values):
                _finite_number(value, f'{name}: native series {index} quantile {quantile} step {step}')


def compare_response(response, case, expected, rtol=1e-4, atol=1e-4):
    """Validate the HTTP schema and compare requested canonical quantile channels.

    ``expected`` is a JSON-ready native forecast in [series, 9, horizon] order.
    Failures identify the exact routed series/channel/time position.
    """
    name = case['name']
    inputs = case['data']['inputs']
    parameters = case['data']['parameters']
    horizon = parameters['prediction_length']
    levels = parameters['quantile_levels']
    for label, tolerance in (('rtol', rtol), ('atol', atol)):
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f'{label} must be finite and nonnegative')
    _check_native_shape(expected, case)
    data = response.get('data') if isinstance(response, dict) else None
    predictions = data.get('predictions') if isinstance(data, dict) else None
    if not isinstance(predictions, list) or len(predictions) != len(inputs):
        raise AssertionError(f'{name}: HTTP data.predictions shape must contain {len(inputs)} series')
    max_abs_error = max_rel_error = 0.0
    compared = 0
    for index, (item, prediction) in enumerate(zip(inputs, predictions)):
        keys = {str(q) for q in levels} | ({'item_id'} if 'item_id' in item else set())
        if not isinstance(prediction, dict) or set(prediction) != keys:
            raise AssertionError(f'{name}: series {index} keys must be {sorted(keys)}, '
                                 f'got {sorted(prediction) if isinstance(prediction, dict) else prediction!r}')
        if 'item_id' in item and prediction['item_id'] != item['item_id']:
            raise AssertionError(f'{name}: series {index} item_id must be {item["item_id"]!r}, '
                                 f'got {prediction["item_id"]!r}')
        for quantile in levels:
            values = prediction[str(quantile)]
            if not isinstance(values, list) or len(values) != horizon:
                raise AssertionError(f'{name}: series {index} quantile {quantile} horizon must be {horizon}')
            reference = expected[index][QUANTILES.index(quantile)]
            for step, (observed, native) in enumerate(zip(values, reference)):
                location = f'{name}: series {index} quantile {quantile} step {step}'
                _finite_number(observed, location)
                error = abs(observed - native)
                tolerance = atol + rtol * abs(native)
                if error > tolerance:
                    raise AssertionError(f'{location}: expected {native:.9g}, observed {observed:.9g}; '
                                         f'absolute error {error:.9g} exceeds tolerance {tolerance:.9g} '
                                         f'(rtol={rtol}, atol={atol})')
                max_abs_error = max(max_abs_error, error)
                # Clamp the denominator so zero-valued references retain JSON-safe statistics.
                max_rel_error = max(max_rel_error, error / max(abs(native), 1e-12))
                compared += 1
    return {'case': name, 'series': len(inputs), 'prediction_length': horizon,
            'quantile_levels': list(levels), 'compared_values': compared,
            'max_abs_error': max_abs_error, 'max_rel_error': max_rel_error,
            'rtol': rtol, 'atol': atol}


def forecast_settings(adapter):
    """Resolve only the CPU-safe numerical contract for an integration adapter."""
    if adapter not in ADAPTERS:
        raise ValueError(f'Unsupported integration adapter: {adapter!r}')
    return importlib.import_module(f'vtsfm.adapters.{adapter}.spec').FORECAST_SETTINGS


def validate_cases(adapter, cases):
    """Validate and copy canonical cases before any model or CUDA initialization."""
    from vtsfm.adapters.forecast import parse_request

    settings = forecast_settings(adapter)
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty list')
    result = deepcopy(cases)
    names = set()
    for case in result:
        if (not isinstance(case, dict) or not {'name', 'data'} <= set(case)
                or set(case) - {'name', 'data', 'evaluation', 'provenance'}):
            raise ValueError('Each case must contain name and data')
        name = case['name']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Case names must be nonempty, unique strings')
        names.add(name)
        _, horizon, levels = parse_request(case['data'], settings)
        case['data']['parameters'] = {'prediction_length': horizon, 'quantile_levels': list(levels)}
    return result


class NativeForecaster:
    """One pinned official model, independent of adapter execution wrappers.

    Inputs use the pure adapter IO contract. ``predict_array`` returns the native
    FP32 array in [series, 9 canonical quantiles, horizon] order; ``predict`` makes
    finite JSON lists for reference artifacts. CUDA ordinal zero refers to the GPU
    selected by the caller's CUDA_VISIBLE_DEVICES. Forecast APIs can mutate
    compilation state, so callers must serialize forecast calls.
    """

    def __init__(self, adapter):
        self.settings = forecast_settings(adapter)
        self.adapter = adapter
        if platform.system() != 'Linux':
            raise RuntimeError('Native integration reference requires Linux and a CUDA GPU')

        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from vtsfm.adapters import get_adapter
        from vtsfm.precision import configure_precision

        if not torch.cuda.is_available():
            raise RuntimeError('Native integration reference requires a working CUDA GPU')
        precision = os.environ.get('VTSFM_TEST_PRECISION', 'fp32')
        configure_precision(adapter, precision)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(4)
        self.device = device = torch.device('cuda:0')
        probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
        torch.testing.assert_close(probe @ torch.eye(4, device=device), probe, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        spec = get_adapter(adapter)
        directory = snapshot_download(spec.model_id, revision=spec.model_revision,
                                      allow_patterns=['config.json', 'model.safetensors'])
        if adapter == 'timesfm25':
            import timesfm
            self.reference = timesfm.TimesFM_2p5_200M_torch.from_pretrained(directory, torch_compile=False)
            self.reference.model.to(device=device, dtype=torch.float32).eval()
            self.reference.model.device = device
            self.reference.model.device_count = 1
            self.forecast_config_class = timesfm.ForecastConfig
        elif adapter == 'toto2':
            from toto2 import Toto2Model
            self.reference = Toto2Model.from_pretrained(directory).to(device=device, dtype=torch.float32).eval()
        else:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            self.reference = Moirai2Module.from_pretrained(directory).to(device=device, dtype=torch.float32).eval()
            self.forecast_class = Moirai2Forecast
        if precision == 'bf16':
            if adapter == 'toto2':
                from dd_unit_scaling import RMSNorm
                from toto2.model import RotaryProjection, SelfAttention
                from vtsfm.adapters.toto2.precision import prepare_native_bf16

                self._precision_handles = prepare_native_bf16(
                    self.reference, RMSNorm, RotaryProjection, attention_type=SelfAttention)
            else:
                from vtsfm.adapters.moirai2.precision import prepare_native_bf16

                self._precision_handles = prepare_native_bf16(self.reference)
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        package = {'timesfm25': 'timesfm', 'toto2': 'toto-2', 'moirai2': 'uni2ts'}[adapter]
        uuid = getattr(properties, 'uuid', None)
        self._metadata = {
            'adapter': adapter, 'model_id': spec.model_id, 'model_revision': spec.model_revision,
            'model_path': str(directory), 'runtime': {
                'gpu': torch.cuda.get_device_name(device), 'uuid': str(uuid) if uuid is not None else None,
                'device': str(device), 'dtype': 'bfloat16' if precision == 'bf16' else 'float32',
                'precision': precision or 'fp32', 'output_dtype': 'float32', 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'python': platform.python_version(),
                'platform': platform.platform(), 'seed': 0,
                'torch_num_threads': torch.get_num_threads(),
                'package': package, 'package_version': importlib.metadata.version(package),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'nvidia_tf32_override': os.environ.get('NVIDIA_TF32_OVERRIDE'),
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'torch_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            },
        }

        if adapter == 'toto2' and precision == 'bf16':
            self._metadata['runtime']['oversized_variate_sdpa'] = (
                'efficient_or_math_for_cuda_bf16_singleton_batch_gt_65535')

    def predict_array(self, data):
        """Run official inference without materializing an intermediate JSON list."""
        import numpy as np
        import torch
        from vtsfm.adapters.forecast import parse_request

        inputs, horizon, _ = parse_request(data, self.settings)
        histories = [np.asarray(item['target'], dtype=np.float32) for item in inputs]
        reference, device = self.reference, self.device
        with torch.inference_mode():
            context_length = max(map(len, histories))
            if self.adapter == 'timesfm25':
                reference.compile(self.forecast_config_class(
                    max_context=context_length, max_horizon=horizon,
                    per_core_batch_size=len(histories), normalize_inputs=True,
                    use_continuous_quantile_head=True, force_flip_invariance=True,
                    infer_is_positive=True, fix_quantile_crossing=True))
                values = reference.forecast(horizon=horizon, inputs=histories)[1][..., 1:].transpose(0, 2, 1)
            elif self.adapter == 'toto2':
                # Pad raw histories independently; padding is never an observed target.
                context = torch.full((len(histories), context_length), float('nan'),
                                     dtype=torch.float32, device=device)
                for row, history in zip(context, histories):
                    row[-len(history):] = torch.as_tensor(history, device=device)
                padding = -context_length % reference.config.patch_size
                context = torch.nn.functional.pad(context, (padding, 0), value=float('nan'))
                target_mask = torch.isfinite(context).unsqueeze(1)
                target = torch.nan_to_num(context).unsqueeze(1)
                series_ids = torch.zeros((len(histories), 1), dtype=torch.long, device=device)
                values = reference.forecast(
                    {'target': target, 'target_mask': target_mask, 'series_ids': series_ids},
                    horizon=horizon, decode_block_size=768)[:, :, 0, :].permute(1, 0, 2)
                values = values.cpu().numpy()
            else:
                # Batched predict pads shorter histories by repeating their first
                # value as observed, which changes normalization. Forecast each
                # history length separately; equal lengths still share one call.
                values = np.empty((len(histories), len(self.settings.quantiles), horizon), dtype=np.float32)
                for length in sorted(set(map(len, histories))):
                    rows = [index for index, history in enumerate(histories) if len(history) == length]
                    # The wrapper owns the shared reference as a submodule.
                    # Move devices without undoing its selected compute dtype.
                    runner = self.forecast_class(
                        module=reference, prediction_length=horizon, context_length=length,
                        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                    ).to(device=device).eval()
                    values[rows] = runner.predict([histories[index] for index in rows])
        values = np.asarray(values)
        shape = (len(inputs), len(self.settings.quantiles), horizon)
        if values.shape != shape:
            raise ValueError(f'Unexpected native forecast shape: {values.shape}; expected {shape}')
        torch.cuda.synchronize(device)
        return values

    def predict(self, data):
        """Build the fully validated canonical JSON forecast for saved references."""
        from vtsfm.adapters.forecast import parse_request

        expected = self.predict_array(data).tolist()
        inputs, horizon, _ = parse_request(data, self.settings)
        _check_native_shape(expected, {'name': 'native-forecast', 'data': {
            'inputs': inputs, 'parameters': {'prediction_length': horizon}}})
        return expected

    def metadata(self):
        return deepcopy(self._metadata)


def native_reference(adapter, cases=None):
    """Run validated cases through official upstream CUDA FP32 inference once."""
    cases = validate_cases(adapter, integration_cases(adapter) if cases is None else cases)
    forecaster = NativeForecaster(adapter)
    outputs = []
    for case in cases:
        expected = forecaster.predict(case['data'])
        _check_native_shape(expected, case)
        outputs.append({'case': case, 'expected': expected})
    return {**forecaster.metadata(), 'cases': outputs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', choices=ADAPTERS, required=True)
    parser.add_argument('--cases', type=Path, help='JSON case list or {cases, provenance} manifest')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.cases.read_text()) if args.cases else None
    cases = manifest.get('cases') if isinstance(manifest, dict) else manifest
    if isinstance(manifest, dict) and 'cases' not in manifest:
        parser.error('The cases manifest must contain cases')
    if args.cases and not isinstance(cases, list):
        parser.error('cases must be a nonempty list')
    result = native_reference(args.adapter, cases)
    if isinstance(manifest, dict) and 'provenance' in manifest:
        result['provenance'] = manifest['provenance']
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Saved {len(result["cases"])} native CUDA reference cases to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
