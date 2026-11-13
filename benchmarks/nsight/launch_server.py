"""Exec the vTSFM vLLM server, optionally under `nsys launch` (profiling starts later via `nsys start`).

Command/env come from vtsfm.cli.build_vllm_command with the flags benchmarks/continuous.py adds
(--no-async-scheduling, --gpu-memory-utilization .35). The NVTX hook in nvtx_hook/ is put first
on PYTHONPATH and enabled with VTSFM_NVTX=1, so both the API server and EngineCore are annotated.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vtsfm.cli import build_vllm_command  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--mode', default='single', choices=['single', 'whole', 'continuous'])
ap.add_argument('--port', type=int, default=18516)
ap.add_argument('--session', default='vtsfm_nsight')
ap.add_argument('--queue-limit', type=int, default=128)
ap.add_argument('--no-nsys', action='store_true')
a = ap.parse_args()

kw = dict(adapter=a.adapter, port=a.port, benchmark=False)
if a.mode != 'single':
    kw.update(max_num_seqs=4, forecast_batch_rows=8, forecast_queue_limit=a.queue_limit,
              continuous_batching=a.mode == 'continuous')
command, env = build_vllm_command(**kw)
if '--no-async-scheduling' not in command:
    command.append('--no-async-scheduling')
command += ['--gpu-memory-utilization', '.35']
env.update(NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4', VTSFM_NVTX='1',
           PYTHONPATH=os.pathsep.join([str(HERE / 'nvtx_hook'), str(ROOT / 'src'), str(ROOT)]))
prefix = [] if a.no_nsys else [
    os.environ.get('NSYS', 'nsys'), 'launch', f'--session-new={a.session}',
    '--trace=cuda,nvtx,osrt,python-gil', '--cuda-graph-trace=node',
    '--trace-fork-before-exec=true', '--wait=all']
print('exec:', ' '.join(prefix + command), flush=True)
os.chdir(ROOT)
os.execvpe((prefix + command)[0], prefix + command, env)
