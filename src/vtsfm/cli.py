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

from vtsfm.config import DEFAULT_PORT, PYTHON_VERSION, serving_pins
from vtsfm.adapters import get_adapter, list_adapters


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
    return environment


def build_vllm_command(backend: str = 'vtsfm', host: str = '127.0.0.1',
                       port: int = DEFAULT_PORT, benchmark: bool = False,
                       adapter: str = 'chronos2', max_num_seqs: int | None = None
                       ) -> tuple[list[str], dict[str, str]]:
    selected = get_adapter(adapter)
    if backend == 'vtsfm':
        plugin = 'vtsfm'
    elif backend in ('upstream', selected.upstream_io_plugin) and selected.upstream_io_plugin:
        plugin = selected.upstream_io_plugin
    else:
        raise ValueError(f'Backend {backend!r} is unavailable for adapter {adapter!r}')
    _check_address(host, port, benchmark)
    environment = _environment(benchmark)
    environment['VLLM_PLUGINS'] = plugin
    environment['VTSFM_ADAPTER'] = selected.name
    # MRV2 does not support the plugin pooling task in the published releases.
    environment['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    command = [str(Path(sys.executable).with_name('vllm')), 'serve', selected.model_id,
               '--revision', selected.model_revision, '--host', host, '--port', str(port),
               '--io-processor-plugin', plugin, *selected.vllm_args]
    if max_num_seqs is not None:
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError('max_num_seqs must be a positive integer')
        if selected.name == 'chronos2' and max_num_seqs != 1:
            raise ValueError('Concurrent forecast routing is enabled only for TimesFM, Toto, and Moirai')
        command[command.index('--max-num-seqs') + 1] = str(max_num_seqs)
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
                       help='Scheduled forecast requests; newer adapters execute them separately (default: 1)')
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
        elif args.max_num_seqs is not None:
            raise ValueError('--max-num-seqs applies to the vLLM backend')
        command, environment = builder(**kwargs)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        keys = ('VTSFM_ADAPTER', 'VLLM_PLUGINS', 'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_SERVER_DEV_MODE',
                'OMP_NUM_THREADS', 'HF_HUB_DISABLE_TELEMETRY')
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
