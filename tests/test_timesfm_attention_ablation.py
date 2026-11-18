"""CPU-safe entry points for the manual attention-only diagnostic."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_attention_ablation_help_is_cpu_safe():
    result = subprocess.run(
        [sys.executable, '-m', 'benchmarks.timesfm_attention_ablation', '--help'],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'precomputed' in result.stdout.lower()
    assert '--cuda-graphs' in result.stdout


def test_attention_ablation_requires_optin_before_output(tmp_path):
    environment = os.environ.copy()
    environment.pop('VTSFM_RUN_TIMESFM_NATIVE_KV', None)
    destination = tmp_path / 'not-submitted'
    result = subprocess.run(
        [sys.executable, '-m', 'benchmarks.timesfm_attention_ablation',
         '--output', str(destination)],
        cwd=ROOT, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 2 and 'VTSFM_RUN_TIMESFM_NATIVE_KV' in result.stderr
    assert not destination.exists()
