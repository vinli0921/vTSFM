"""Issue #16 HTTP client; optionally starts/stops an `nsys launch` session around timed requests.

  --round 1      batch 1, context 2048, H96, window advances one patch per request (harness round 1)
  --round 0b     batch 64, context 2048, H96, fixed windows (harness round 0b)
  --round 0b16   as 0b with 16 rows: TimesFM's admission policy caps a request at 16 rows
  --round sat    benchmarks/continuous.py saturated case: B2/C512/H1024 MetroPT windows offered
                 1 ms apart, all N requests in flight
Sequential rounds keep one request outstanding. All nine quantiles are requested.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import workload  # noqa: E402

NSYS = os.environ.get('NSYS', 'nsys')


def payloads(a):
    if a.round == 'sat':
        from benchmarks.continuous import _cases
        case = _cases(Path(a.data_dir), 4)[3]
        assert case['name'].endswith('b2-c512-h1024'), case['name']
        body = json.dumps({'model': a.model_id, 'task': 'plugin', 'data': case['data']}).encode()
        return lambda i: (body, 2)
    make = workload.round_inputs(workload.load_series(), a.round, a.patch)

    def encode(i):
        ctx = make(i)
        return json.dumps({'model': a.model_id, 'task': 'plugin', 'data': {
            'inputs': [{'item_id': f's{j}', 'target': row.tolist()} for j, row in enumerate(ctx)],
            'parameters': {'prediction_length': workload.HORIZON, 'quantile_levels': workload.LEVELS}},
        }).encode(), len(ctx)
    if a.round == '1':
        return encode
    fixed = encode(0)
    return lambda i: fixed


def post(session, url, body, series, timeout):
    response = session.post(url, data=body, headers={'Content-Type': 'application/json'}, timeout=timeout)
    response.raise_for_status()
    if len(response.json()['data']['predictions']) != series:
        raise ValueError('Response has the wrong series count')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18516')
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--round', required=True, choices=['1', '0b', '0b16', 'sat'])
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--warm', type=int, default=5)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--data-dir', help='prepared MetroPT data (benchmarks/prepare_data.py), for --round sat')
    ap.add_argument('--session', help='nsys launch session to start/stop around timed requests')
    ap.add_argument('--trace-out', help='nsys output path (no extension)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    make = payloads(a)
    url = a.url + '/pooling'
    session = requests.Session()
    session.trust_env = False
    for i in range(a.warm):
        post(session, url, *make(i), 600)
    if a.session:
        subprocess.run([NSYS, 'start', f'--session={a.session}', '-o', a.trace_out,
                        '--force-overwrite=true'], check=True)
        time.sleep(1.0)
    latencies, lock = [], threading.Lock()
    started = time.perf_counter()
    if a.round == 'sat':
        def worker(i):
            own = requests.Session()
            own.trust_env = False
            t0 = time.perf_counter()
            post(own, url, *make(i), 1200)
            with lock:
                latencies.append(time.perf_counter() - t0)
        threads = []
        for i in range(a.n):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(0.001)
        for thread in threads:
            thread.join()
        if len(latencies) != a.n:
            raise RuntimeError('Some saturated requests failed')
    else:
        for i in range(a.n):
            t0 = time.perf_counter()
            post(session, url, *make(a.warm + i), 600)
            latencies.append(time.perf_counter() - t0)
    wall = time.perf_counter() - started
    if a.session:
        time.sleep(0.5)
        subprocess.run([NSYS, 'stop', f'--session={a.session}'], check=True)
    ms = sorted(1e3 * x for x in latencies)
    result = {'round': a.round, 'n': a.n, 'wall_s': wall, 'req_per_s': a.n / wall,
              'mean_ms': float(np.mean(ms)), 'p50_ms': float(np.percentile(ms, 50)),
              'p95_ms': float(np.percentile(ms, 95))}
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
