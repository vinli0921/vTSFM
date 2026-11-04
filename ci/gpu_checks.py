"""Local submission gate and the small, provider-independent GPU test command."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import sys


def require_setup(environment: Mapping[str, str]) -> None:
    """Check explicit owner acknowledgement, not the provider's actual budget.

    The owner must first save the usage budget and $0 net-spend limit in Modal.
    Neither this flag nor a timeout can guarantee that a run uses free credits.
    """
    if environment.get('MODAL_GPU_ENABLED') != 'true':
        raise RuntimeError('Set MODAL_GPU_ENABLED=true only after verifying Modal credit '
                           'allowance, workspace usage budget, and $0 spend limit; '
                           'see docs/gpu-testing.md')
    for name in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET'):
        if not environment.get(name, '').strip():
            raise RuntimeError(f'Missing {name}; add the matching Modal repository secrets')


def run_gpu_checks(root: Path = Path('/workspace')) -> dict[str, str]:
    """Execute a fixed correctness suite; model downloads and benchmarks are excluded."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('GPU job has no working CUDA device')
    info = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': str(torch.version.cuda)}
    print(json.dumps(info), flush=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(root / 'src')
    environment['VTSFM_TEST_ADAPTER'] = ''
    subprocess.run([
        sys.executable, '-m', 'pytest', '-q', '-s', '-p', 'no:cacheprovider',
        'tests/test_cuda_contract.py', 'tests/test_execution.py', 'tests/test_batching.py',
    ], cwd=root, env=environment, check=True, timeout=240)
    torch.cuda.synchronize()
    return {**info, 'result': 'passed'}


if __name__ == '__main__':
    require_setup(os.environ)
    print('Local setup acknowledged. Provider billing caps must be verified in Modal.')
