"""Manual, bounded GPU correctness checks: modal run -m ci.modal_gpu.

    Setup and required billing caps: docs/gpu-testing.md.
    This defines an ephemeral App; do not deploy it as an always-on service.
"""

import json
import os
from pathlib import Path

import modal

from ci.gpu_checks import require_setup, run_gpu_checks


# Refuse local submissions before image construction or App startup. On Modal
# workers the client credentials are intentionally absent and this gate is skipped.
if modal.is_local():
    require_setup(os.environ)

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version='3.11.17')
    .apt_install('libgomp1')
    .pip_install('torch==2.9.1', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('numpy==2.2.6', 'pytest==8.4.2')
    .add_local_dir(ROOT / 'src', '/workspace/src')
    .add_local_dir(ROOT / 'tests', '/workspace/tests')
    .add_local_dir(ROOT / 'examples', '/workspace/examples')
    .add_local_file(ROOT / 'pyproject.toml', '/workspace/pyproject.toml')
    .env({'PYTHONPATH': '/workspace/src:/workspace'})
)

app = modal.App('vtsfm-gpu-tests', include_source=True)


@app.function(image=image, gpu='L4', cpu=1, memory=4096,
              min_containers=0, max_containers=1, buffer_containers=0,
              scaledown_window=2, single_use_containers=True, retries=0,
              startup_timeout=120, timeout=300)
def gpu_tests():
    return run_gpu_checks()


@app.local_entrypoint()
def main():
    print(json.dumps(gpu_tests.remote(), indent=2))
