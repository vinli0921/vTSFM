"""Serving commands and environment checks; help/dry-run are CPU safe."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

from vtsfm.config import (
    DEFAULT_PORT, FORECAST_BATCH_ROWS_ENV, MAX_FORECAST_BATCH_ROWS, PYTHON_VERSION, serving_pins,
)
from vtsfm.adapters import get_adapter, list_adapters
from vtsfm.adapters.moirai2.spec import MOIRAI_ATTENTION_ENV, resolve_attention_config
from vtsfm.adapters.toto2.spec import compile_scope as toto_compile_scope
from vtsfm.adapters.timesfm25.graph_config import (
    FORECAST_GRAPH_SHAPE, FORECAST_MODE_ENV, FORECAST_MODES, validate_fixed_forecast_execution,
)
from vtsfm.admission import FORECAST_LIMITS_ENV, resolve_cost, resolve_limits
from vtsfm.compilation import EXECUTION_ENV, EXECUTION_MODES, execution_mode
from vtsfm.stepping import CONTINUOUS_ENV
from vtsfm.precision import PRECISION_ENV, PRECISIONS, TF32_OVERRIDE_ENV, validate_precision


def _check_address(host: str, port: int, benchmark: bool) -> None:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('port must be an integer in [1,65535]')
    if not host:
        raise ValueError('host cannot be empty')
    if benchmark and host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('benchmark diagnostic RPC must bind to a loopback address')


def _environment(benchmark: bool = False) -> dict[str, str]:
    environment = dict(os.environ)
    environment['OMP_NUM_THREADS'] = '4' if benchmark else environment.get('OMP_NUM_THREADS', '4')
    environment['HF_HUB_DISABLE_TELEMETRY'] = '1'
    environment.pop('VLLM_SERVER_DEV_MODE', None)
    environment.pop(FORECAST_MODE_ENV, None)
    return environment


def build_vllm_command(backend: str = 'vtsfm', host: str = '127.0.0.1',
                       port: int = DEFAULT_PORT, benchmark: bool = False,
                       adapter: str = 'chronos2', max_num_seqs: int | None = None,
                       forecast_batch_rows: int | None = None,
                       forecast_limits: str | None = None,
                       continuous_batching: bool = False,
                       forecast_queue_limit: int | None = None,
                       forecast_precision: str | None = None,
                       forecast_execution: str | None = None,
                       timesfm_forecast_mode: str | None = None,
                       ) -> tuple[list[str], dict[str, str]]:
    selected = get_adapter(adapter)
    if timesfm_forecast_mode is not None:
        if timesfm_forecast_mode not in FORECAST_MODES:
            raise ValueError(f'--timesfm-forecast-mode must be one of {", ".join(FORECAST_MODES)}')
        if selected.name != 'timesfm25' or backend != 'vtsfm':
            raise ValueError('--timesfm-forecast-mode supports only TimesFM through the vtsfm backend')
        if continuous_batching:
            raise ValueError('--timesfm-forecast-mode cannot be combined with --continuous-batching')
        if forecast_batch_rows is not None:
            raise ValueError('--timesfm-forecast-mode cannot be combined with --forecast-batch-rows')
        if max_num_seqs is not None and (type(max_num_seqs) is not int or max_num_seqs != 1):
            raise ValueError('--timesfm-forecast-mode requires exactly one scheduled request (--max-num-seqs 1)')
        max_num_seqs = 1
    if forecast_precision is None:
        forecast_precision = os.environ.get(PRECISION_ENV)
    validate_precision(adapter, forecast_precision)
    if forecast_precision is not None and backend != 'vtsfm':
        raise ValueError('Experimental precision requires the vTSFM backend')
    if backend == 'vtsfm':
        plugin = 'vtsfm'
    elif backend in ('upstream', selected.upstream_io_plugin) and selected.upstream_io_plugin:
        plugin = selected.upstream_io_plugin
    else:
        raise ValueError(f'Backend {backend!r} is unavailable for adapter {adapter!r}')
    _check_address(host, port, benchmark)
    mode = execution_mode(forecast_execution)
    if timesfm_forecast_mode is not None:
        validate_fixed_forecast_execution(forecast_precision, mode)
    if mode == 'compile' and (selected.name == 'chronos2' or backend != 'vtsfm'):
        raise ValueError('Compiled forecasts support TimesFM, Toto and Moirai through vTSFM')
    if continuous_batching:
        if selected.name != 'timesfm25' or backend != 'vtsfm':
            raise ValueError('Forecast-block continuous batching currently supports only TimesFM/vtsfm')
        max_num_seqs = 4 if max_num_seqs is None else max_num_seqs
        if type(max_num_seqs) is not int or max_num_seqs < 2:
            raise ValueError('Forecast-block continuous batching requires at least two scheduled slots')
        forecast_batch_rows = 16 if forecast_batch_rows is None else forecast_batch_rows
        forecast_queue_limit = 16 if forecast_queue_limit is None else forecast_queue_limit
    environment = _environment(benchmark)
    environment[EXECUTION_ENV] = mode
    if selected.name == 'moirai2':
        environment[MOIRAI_ATTENTION_ENV], forecast_precision = resolve_attention_config(
            mode, forecast_precision, environment)
    if selected.name == 'toto2' and toto_compile_scope(environment, mode, forecast_precision) == 'whole':
        forecast_precision = 'fp32'
    if mode == 'compile':
        # Set before the child imports Torch/Triton, preserving native arithmetic.
        environment['TRITON_DEFAULT_FP_FUSION'] = '0'
    environment.pop(FORECAST_BATCH_ROWS_ENV, None)
    environment.pop(FORECAST_LIMITS_ENV, None)
    environment.pop(CONTINUOUS_ENV, None)
    # An explicit CLI selection overrides the worker environment.
    environment.pop(PRECISION_ENV, None)
    if forecast_precision is not None:
        environment[PRECISION_ENV] = forecast_precision
        environment[TF32_OVERRIDE_ENV] = '1' if forecast_precision == 'tf32' else '0'
    environment['VLLM_PLUGINS'] = plugin
    environment['VTSFM_ADAPTER'] = selected.name
    # MRV2 does not support the plugin pooling task in the published releases.
    environment['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    command = [str(Path(sys.executable).with_name('vllm')), 'serve', selected.model_id,
               '--revision', selected.model_revision, '--host', host, '--port', str(port),
               '--io-processor-plugin', plugin, *selected.vllm_args]
    if timesfm_forecast_mode is not None:
        environment[FORECAST_MODE_ENV] = timesfm_forecast_mode
        environment[TF32_OVERRIDE_ENV] = '0'
        if timesfm_forecast_mode.startswith('paged-'):
            environment['TRITON_F32_DEFAULT'] = 'ieee'
        # The adapter owns its CUDA graph; keep the pinned vLLM bridge synchronous.
        command += ['--no-async-scheduling', '--served-model-name', selected.model_id]
    if max_num_seqs is not None:
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError('max_num_seqs must be a positive integer')
        if selected.name == 'chronos2' and max_num_seqs != 1:
            raise ValueError('Concurrent forecast routing is enabled only for TimesFM, Toto, and Moirai')
        command[command.index('--max-num-seqs') + 1] = str(max_num_seqs)
    if forecast_batch_rows is not None:
        if selected.admission_limits is None or backend != 'vtsfm':
            raise ValueError('Forecast batching requires a vTSFM adapter with a numerical admission policy')
        if max_num_seqs is None or max_num_seqs < 2:
            raise ValueError('Forecast batching requires --max-num-seqs of at least 2')
        if type(forecast_batch_rows) is not int or not 2 <= forecast_batch_rows <= MAX_FORECAST_BATCH_ROWS:
            raise ValueError(f'--forecast-batch-rows must be between 2 and {MAX_FORECAST_BATCH_ROWS}')
        environment[FORECAST_BATCH_ROWS_ENV] = str(forecast_batch_rows)
    if selected.admission_limits is not None:
        encoded = Path(forecast_limits).read_text() if forecast_limits is not None else None
        limits = resolve_limits(selected.admission_limits, encoded)
        slots = int(command[command.index('--max-num-seqs') + 1])
        # The adapter owns this cost; its descriptor imports no model runtime.
        forecast_cost = resolve_cost(selected)
        limits.validate_launch(slots, forecast_cost(
            limits.max_request_rows, limits.max_context, limits.max_horizon))
        # Resolve a file once so frontend/worker see the exact same policy.
        environment[FORECAST_LIMITS_ENV] = json.dumps(asdict(limits), separators=(',', ':'))
    elif forecast_limits is not None:
        raise ValueError('--forecast-limits requires an adapter with a numerical admission policy')
    if forecast_queue_limit is not None:
        if selected.admission_limits is None or type(forecast_queue_limit) is not int or forecast_queue_limit < 1:
            raise ValueError('--forecast-queue-limit requires a numerical admission policy and a positive integer')
        command += ['--max-num-queued-reqs', str(forecast_queue_limit)]
    if continuous_batching:
        environment[CONTINUOUS_ENV] = '1'
        command[command.index('--worker-cls') + 1] = 'vtsfm.block_worker.ForecastBlockWorker'
        command += ['--scheduler-cls', 'vtsfm.continuous_scheduler.ForecastBlockScheduler',
                    '--enable-chunked-prefill', '--no-async-scheduling']
    if benchmark:
        environment['VLLM_SERVER_DEV_MODE'] = '1'
        command += ['--worker-extension-cls', 'vtsfm.benchmark_memory.MemoryMetrics']
    return command, environment


def build_native_command(host: str = '127.0.0.1', port: int = DEFAULT_PORT,
                         benchmark: bool = False, adapter: str = 'chronos2'
                         ) -> tuple[list[str], dict[str, str]]:
    selected = get_adapter(adapter)
    if selected.native_server_module is None:
        raise ValueError(f'Adapter {adapter!r} has no native comparator')
    _check_address(host, port, benchmark)
    environment = _environment(benchmark)
    environment['VLLM_PLUGINS'] = ''
    environment['VTSFM_ADAPTER'] = selected.name
    return ([sys.executable, '-m', 'uvicorn', f'{selected.native_server_module}:app',
             '--host', host, '--port', str(port), '--workers', '1',
             '--log-level', 'warning'], environment)


def doctor(cpu: bool = False, adapter: str = 'chronos2') -> tuple[dict, bool]:
    selected = get_adapter(adapter)
    packages = {}
    runtime_package = {'chronos2': 'chronos-forecasting', 'timesfm25': 'timesfm',
                       'toto2': 'toto-2', 'moirai2': 'uni2ts'}[selected.name]
    for name in ('vtsfm', 'numpy', 'requests', 'torch', 'vllm', 'transformers', runtime_package):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {'mode': 'cpu' if cpu else 'gpu', 'python': platform.python_version(),
              'reference_python': '3.12.12' if selected.name == 'toto2' else PYTHON_VERSION, 'packages': packages,
              'adapter': selected.name, 'model': selected.model_id,
              'model_revision': selected.model_revision}
    if cpu:
        result['ready'] = packages['vtsfm'] is not None
        return result, result['ready']
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        result['gpu'] = torch.cuda.get_device_name(0)
        result['cuda_runtime'] = torch.version.cuda
        result['cuda_test'] = (torch.ones(4, device='cuda') * 2).cpu().tolist()
        actual = {**packages, 'cuda_runtime': result['cuda_runtime']}
        mismatches = {name: {'expected': version, 'actual': actual[name]}
                      for name, version in serving_pins(selected.name).items()
                      if (actual[name] or '').split('+')[0] != version}
        result['version_mismatches'] = mismatches
        result['ready'] = not mismatches and packages[runtime_package] is not None
    except (ImportError, RuntimeError) as error:
        result.update(ready=False, error=str(error), hint='Run scripts/setup.sh on the Linux GPU host, or use doctor --cpu.')
    return result, result['ready']


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    serve = commands.add_parser('serve', help='Serve a numerical forecasting backend')
    serve.add_argument('--backend', choices=('vtsfm', 'upstream', 'chronos2', 'native'), default='vtsfm')
    serve.add_argument('--adapter', default='chronos2', help='Model adapter; use the adapters command to list support')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=DEFAULT_PORT)
    serve.add_argument('--benchmark', action='store_true', help='Enable local benchmark-only diagnostics')
    serve.add_argument('--max-num-seqs', type=int,
                       help='Maximum scheduled forecast requests (default: 1)')
    serve.add_argument('--forecast-batch-rows', type=int,
                       help='Enable bounded whole-forecast batching with this fused series limit (2–16)')
    serve.add_argument('--forecast-limits', metavar='JSON_FILE',
                       help='Tighten the selected adapter\'s numerical limits using a JSON policy file')
    serve.add_argument('--continuous-batching', action='store_true',
                       help='Enable experimental TimesFM forecast-block scheduling (default: 4 slots, 16 fused rows)')
    serve.add_argument('--forecast-queue-limit', type=int,
                       help='Maximum numerical HTTP requests waiting/running (block default: 16; overflow: 503)')
    rows, context, horizon = FORECAST_GRAPH_SHAPE
    serve.add_argument('--timesfm-forecast-mode', choices=FORECAST_MODES,
                       help=f'Experimental TimesFM/vtsfm whole forecast at B{rows}/C{context}/H{horizon}; one scheduled request')
    serve.add_argument('--forecast-precision', choices=PRECISIONS,
                       help='Select FP32, TF32 matmuls, or native BF16 (Toto/Moirai)')
    serve.add_argument('--forecast-execution', choices=EXECUTION_MODES,
                       help='Opt-in neural compilation (default: environment, otherwise eager)')
    serve.add_argument('--dry-run', action='store_true', help='Print a command without starting a server')
    diagnostic = commands.add_parser('doctor', help='Check the installed environment')
    diagnostic.add_argument('--cpu', action='store_true', help='Skip imports and checks of GPU dependencies')
    diagnostic.add_argument('--adapter', default='chronos2')
    commands.add_parser('adapters', help='List registered model adapters without loading model runtimes')
    args = parser.parse_args(argv)
    if args.action == 'adapters':
        print(json.dumps({'adapters': [asdict(adapter) for adapter in list_adapters()]}, indent=2))
        return 0
    if args.action == 'doctor':
        try:
            result, ready = doctor(args.cpu, args.adapter)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0 if ready else 1
    try:
        builder = build_native_command if args.backend == 'native' else build_vllm_command
        kwargs = {'host': args.host, 'port': args.port, 'benchmark': args.benchmark,
                  'adapter': args.adapter}
        if args.backend != 'native':
            kwargs['backend'] = args.backend
            kwargs['max_num_seqs'] = args.max_num_seqs
            kwargs['forecast_batch_rows'] = args.forecast_batch_rows
            kwargs['forecast_limits'] = args.forecast_limits
            kwargs['continuous_batching'] = args.continuous_batching
            kwargs['forecast_queue_limit'] = args.forecast_queue_limit
            kwargs['timesfm_forecast_mode'] = args.timesfm_forecast_mode
            kwargs['forecast_precision'] = args.forecast_precision
            kwargs['forecast_execution'] = args.forecast_execution
        elif args.timesfm_forecast_mode is not None:
            raise ValueError('--timesfm-forecast-mode applies to the vLLM backend')
        elif args.max_num_seqs is not None:
            raise ValueError('--max-num-seqs applies to the vLLM backend')
        elif args.forecast_batch_rows is not None:
            raise ValueError('--forecast-batch-rows applies to the vLLM backend')
        elif args.forecast_limits is not None:
            raise ValueError('--forecast-limits applies to the vLLM backend')
        elif args.continuous_batching or args.forecast_queue_limit is not None:
            raise ValueError('Forecast-block and queue settings apply to the vLLM backend')
        elif args.forecast_precision is not None:
            raise ValueError('--forecast-precision applies to the vTSFM backend')
        elif args.forecast_execution == 'compile':
            raise ValueError('--forecast-execution compile applies to the vLLM backend')
        command, environment = builder(**kwargs)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        keys = ('VTSFM_ADAPTER', 'VLLM_PLUGINS', 'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_SERVER_DEV_MODE',
                'OMP_NUM_THREADS', 'HF_HUB_DISABLE_TELEMETRY', FORECAST_BATCH_ROWS_ENV, FORECAST_LIMITS_ENV,
                CONTINUOUS_ENV, FORECAST_MODE_ENV, PRECISION_ENV, TF32_OVERRIDE_ENV,
                EXECUTION_ENV, MOIRAI_ATTENTION_ENV, 'TRITON_F32_DEFAULT', 'TRITON_DEFAULT_FP_FUSION')
        print(json.dumps({'command': command, 'environment': {
            key: environment[key] for key in keys if key in environment}}, indent=2))
        return 0
    dependency = 'torch' if args.backend == 'native' else 'vllm'
    if importlib.util.find_spec(dependency) is None:
        parser.error('Serving dependencies are missing; run scripts/setup.sh on the Linux GPU host.')
    os.execvpe(command[0], command, environment)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
