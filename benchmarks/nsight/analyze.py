"""Issue #16: per-request host-time breakdown from nsys sqlite exports.

usage: python3 analyze.py TRACE.sqlite [...]  -> prints one JSON object per trace

Windows:
  in-process traces: NVTX push/pop 'request' ranges (only timed requests are captured)
  HTTP traces:       NVTX 'PoolingBaseServing.__call__' ranges (API server handler)
Compute thread = the thread issuing the most cudaLaunchKernel calls (main thread in-process,
EngineCore main thread for HTTP). Its time inside the windows is partitioned into
CUDA launch / sync wait / memcpy / other CUDA API / Python+framework (remainder).
GPU busy = union of kernel, memcpy and memset intervals inside the windows.
"""
import json
import sqlite3
import sys
from collections import defaultdict

LAUNCH = ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernelEx',
          'cudaLaunchCooperativeKernel', 'cudaGraphLaunch', 'cuGraphLaunch')
SYNC = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize',
        'cuStreamSynchronize', 'cuCtxSynchronize', 'cuEventSynchronize', 'cudaStreamWaitEvent',
        'cudaStreamQuery', 'cudaEventQuery')
MEMCPY = ('cudaMemcpy', 'cudaMemcpyAsync', 'cuMemcpyDtoHAsync', 'cuMemcpyHtoDAsync', 'cudaMemcpy2DAsync',
          'cuMemcpyAsync', 'cudaMemcpyFromSymbol', 'cudaMemcpyToSymbol')


def base(name):
    return name.split('_v')[0] if name else name


def cat(name):
    n = base(name)
    if n in LAUNCH:
        return 'launch'
    if n in SYNC:
        return 'sync'
    if n in MEMCPY or n.startswith('cudaMemcpy') or n.startswith('cuMemcpy'):
        return 'memcpy'
    return 'other_api'


def union(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def clip(s, e, wins):
    """Overlap of [s,e) with sorted, disjoint windows."""
    t = 0
    for ws, we in wins:
        if we <= s:
            continue
        if ws >= e:
            break
        t += min(e, we) - max(s, ws)
    return t


def tables(c):
    return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


def nvtx_ranges(c):
    q = """select n.start, n.end, coalesce(n.text, s.value) as name, n.globalTid
           from NVTX_EVENTS n left join StringIds s on n.textId = s.id
           where n.end is not null and n.end > n.start"""
    return [(s, e, name or '', tid) for s, e, name, tid in c.execute(q)]


def analyze(path):
    c = sqlite3.connect(path)
    T = tables(c)
    rng = nvtx_ranges(c)
    req = sorted((s, e) for s, e, n, _ in rng if n == 'request')
    kind = 'inproc'
    if not req:
        req = sorted((s, e) for s, e, n, _ in rng if n == 'PoolingBaseServing.__call__')
        kind = 'http'
    if not req:
        return {'trace': path, 'error': 'no request windows'}
    concurrent = any(req[i + 1][0] < req[i][1] for i in range(len(req) - 1))
    # analysis windows: the request windows, or for concurrent load the busy span
    wins = [(req[0][0], max(e for _, e in req))] if concurrent else req
    n_req = len(req)
    wall = sum(e - s for s, e in wins)

    api = list(c.execute("""select r.start, r.end, s.value, r.globalTid, r.correlationId
                            from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id"""))
    launches = defaultdict(int)
    for s, e, name, tid, _ in api:
        if cat(name) == 'launch':
            launches[tid] += 1
    comp_tid = max(launches, key=launches.get) if launches else None
    by_cat = defaultdict(int)
    by_name = defaultdict(lambda: [0, 0])
    n_launch = 0
    for s, e, name, tid, _ in api:
        if tid != comp_tid:
            continue
        o = clip(s, e, wins)
        if o <= 0:
            continue
        by_cat[cat(name)] += o
        by_name[base(name)][0] += o
        by_name[base(name)][1] += 1
        if cat(name) == 'launch':
            n_launch += 1
    gpu_iv = []
    n_kern, kern_time = 0, 0
    if 'CUPTI_ACTIVITY_KIND_KERNEL' in T:
        for s, e in c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL'):
            if clip(s, e, wins) > 0:
                gpu_iv.append((s, e))
                n_kern += 1
                kern_time += e - s
    memcpy = defaultdict(lambda: [0, 0, 0])
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in T:
        for s, e, k, b in c.execute('select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY'):
            if clip(s, e, wins) > 0:
                gpu_iv.append((s, e))
                key = {1: 'H2D', 2: 'D2H', 8: 'D2D'}.get(k, str(k))
                memcpy[key][0] += 1
                memcpy[key][1] += e - s
                memcpy[key][2] += b
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in T:
        for s, e in c.execute('select start, end from CUPTI_ACTIVITY_KIND_MEMSET'):
            if clip(s, e, wins) > 0:
                gpu_iv.append((s, e))
    # clip GPU intervals to windows before the union
    clipped = []
    for s, e in gpu_iv:
        for ws, we in wins:
            if we > s and ws < e:
                clipped.append((max(s, ws), min(e, we)))
    gpu_busy = union(clipped)

    # NVTX range means per request (all threads), and GIL wait on the compute thread
    per_range = defaultdict(lambda: [0, 0])
    gil_wait = 0
    gil_wait_all = defaultdict(int)
    for s, e, n, tid in rng:
        if n.startswith('GIL Trace'):
            if 'Waiting' in n:
                o = clip(s, e, wins)
                gil_wait_all[tid] += o
                if tid == comp_tid:
                    gil_wait += o
            continue
        o = clip(s, e, wins)
        if o > 0:
            per_range[n][0] += o
            per_range[n][1] += 1
    api_total = sum(by_cat.values())
    ms = lambda x: round(x / n_req / 1e6, 3)  # noqa: E731  per-request ms
    out = {
        'trace': path.split('/')[-1].replace('.sqlite', ''), 'kind': kind, 'requests': n_req,
        'concurrent': concurrent, 'window_ms_total': round(wall / 1e6, 1),
        'per_request_ms': {
            'wall': ms(wall), 'gpu_busy': ms(gpu_busy), 'gpu_kernel_sum': ms(kern_time),
            'host_launch': ms(by_cat['launch']), 'host_sync_wait': ms(by_cat['sync']),
            'host_memcpy_api': ms(by_cat['memcpy']), 'host_other_cuda_api': ms(by_cat['other_api']),
            'host_python_framework': ms(wall - api_total),
            'compute_thread_gil_wait': ms(gil_wait)},
        'gpu_busy_pct': round(100 * gpu_busy / wall, 1),
        'kernels_per_request': round(n_kern / n_req, 1),
        'launch_calls_per_request': round(n_launch / n_req, 1),
        'mean_launch_us': round(by_cat['launch'] / max(n_launch, 1) / 1e3, 2),
        'memcpy_per_request': {k: {'count': round(v[0] / n_req, 1), 'gpu_ms': ms(v[1]),
                                   'kB': round(v[2] / n_req / 1e3, 1)} for k, v in memcpy.items()},
        'top_cuda_api_ms_per_request': {k: [ms(v[0]), round(v[1] / n_req, 1)] for k, v in
                                        sorted(by_name.items(), key=lambda kv: -kv[1][0])[:8]},
        'nvtx_ms_per_request': {k: [ms(v[0]), round(v[1] / n_req, 2)] for k, v in
                                sorted(per_range.items(), key=lambda kv: -kv[1][0])
                                if k not in ('request',) and v[0] / n_req > 2e4},
        'gil_wait_ms_per_request_by_thread': {str(k): ms(v) for k, v in
                                              sorted(gil_wait_all.items(), key=lambda kv: -kv[1])[:4]
                                              if v / n_req > 1e5},
    }
    return out


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(analyze(p)))
