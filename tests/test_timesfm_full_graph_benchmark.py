"""CPU-safe entry points for the manual full-forecast graph comparison."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_full_graph_benchmark_help_is_cpu_safe():
    result = subprocess.run([sys.executable, '-m', 'benchmarks.timesfm_full_graph', '--help'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for arm in ('dense-eager', 'paged-eager', 'dense-graph', 'paged-graph'):
        assert arm in result.stdout
    assert 'full-forecast' in result.stdout.lower()


def test_full_graph_benchmark_requires_optin_before_output(tmp_path):
    environment = os.environ.copy()
    environment.pop('VTSFM_RUN_TIMESFM_NATIVE_KV', None)
    output = tmp_path / 'not-submitted'
    result = subprocess.run([sys.executable, '-m', 'benchmarks.timesfm_full_graph',
                             '--output', str(output)], cwd=ROOT, env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 2 and 'VTSFM_RUN_TIMESFM_NATIVE_KV' in result.stderr
    assert not output.exists()
