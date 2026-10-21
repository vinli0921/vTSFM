"""Serving commands and environment checks; help/dry-run are CPU safe."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

from vtsfm.config import (
    DEFAULT_PORT, MODEL_ID, MODEL_REVISION, PYTHON_VERSION, TORCH_VERSION,
    TRANSFORMERS_VERSION, VLLM_VERSION,
)


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
                       port: int = DEFAULT_PORT, benchmark: bool = False
                       ) -> tuple[list[str], dict[str, str]]:
    if backend not in ('vtsfm', 'chronos2'):
        raise ValueError('backend must be vtsfm or chronos2')
    _check_address(host, port, benchmark)
    environment = _environment(benchmark)
    environment['VLLM_PLUGINS'] = backend
    command = [str(Path(sys.executable).with_name('vllm')), 'serve', MODEL_ID,
               '--revision', MODEL_REVISION, '--host', host, '--port', str(port),
               '--io-processor-plugin', backend, '--runner', 'pooling',
               '--enforce-eager', '--no-enable-prefix-caching',
               '--skip-tokenizer-init', '--enable-mm-embeds', '--dtype', 'float32',
               '--max-model-len', '8192', '--max-num-batched-tokens', '8192',
               '--max-num-seqs', '1']
    if benchmark:
        environment['VLLM_SERVER_DEV_MODE'] = '1'
        command += ['--worker-extension-cls', 'vtsfm.benchmark_memory.MemoryMetrics']
    return command, environment


def build_native_command(host: str = '127.0.0.1', port: int = DEFAULT_PORT,
                         benchmark: bool = False) -> tuple[list[str], dict[str, str]]:
    _check_address(host, port, benchmark)
    environment = _environment(benchmark)
    environment['VLLM_PLUGINS'] = ''
    return ([sys.executable, '-m', 'uvicorn', 'vtsfm.native_server:app',
             '--host', host, '--port', str(port), '--workers', '1',
             '--log-level', 'warning'], environment)


def doctor(cpu: bool = False) -> tuple[dict, bool]:
    packages = {}
    for name in ('vtsfm', 'numpy', 'requests', 'torch', 'vllm', 'transformers', 'chronos-forecasting'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {'mode': 'cpu' if cpu else 'gpu', 'python': platform.python_version(),
              'reference_python': PYTHON_VERSION, 'packages': packages,
              'model': MODEL_ID, 'model_revision': MODEL_REVISION}
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
        expected = {'vllm': VLLM_VERSION, 'torch': TORCH_VERSION,
                    'transformers': TRANSFORMERS_VERSION}
        mismatches = {name: {'expected': version, 'actual': packages[name]}
                      for name, version in expected.items()
                      if (packages[name] or '').split('+')[0] != version}
        result['version_mismatches'] = mismatches
        result['ready'] = not mismatches and packages['chronos-forecasting'] is not None
    except (ImportError, RuntimeError) as error:
        result.update(ready=False, error=str(error), hint='Run scripts/setup.sh on the Linux GPU host, or use doctor --cpu.')
    return result, result['ready']


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    serve = commands.add_parser('serve', help='Serve a numerical forecasting backend')
    serve.add_argument('--backend', choices=('vtsfm', 'chronos2', 'native'), default='vtsfm')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=DEFAULT_PORT)
    serve.add_argument('--benchmark', action='store_true', help='Enable local benchmark-only diagnostics')
    serve.add_argument('--dry-run', action='store_true', help='Print a command without starting a server')
    diagnostic = commands.add_parser('doctor', help='Check the installed environment')
    diagnostic.add_argument('--cpu', action='store_true', help='Skip imports and checks of GPU dependencies')
    args = parser.parse_args(argv)
    if args.action == 'doctor':
        result, ready = doctor(args.cpu)
        print(json.dumps(result, indent=2))
        return 0 if ready else 1
    try:
        builder = build_native_command if args.backend == 'native' else build_vllm_command
        kwargs = {'host': args.host, 'port': args.port, 'benchmark': args.benchmark}
        if args.backend != 'native':
            kwargs['backend'] = args.backend
        command, environment = builder(**kwargs)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        keys = ('VLLM_PLUGINS', 'VLLM_SERVER_DEV_MODE', 'OMP_NUM_THREADS', 'HF_HUB_DISABLE_TELEMETRY')
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
