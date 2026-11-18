"""Manual precomputed-QKV diagnostic for native versus paged TimesFM attention.

This synthetic, steady-cache replay excludes neural layers, normalization, cache
leases, metadata construction and forecast scheduling. Replaying the same four
append slots is intentional; this is not a growing forecast or HTTP benchmark.
The two backends coexist in one process: no independent memory claim is made.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
import traceback

from benchmarks.timesfm_native_kv import (
    GPU_OWNERSHIP_METHOD, _assert_gpu_owned, _integers, _installed_metadata, _source_metadata, _write,
)


TOLERANCE = {'rtol': 1e-4, 'atol': 1e-4}
PATHS = ('native-attention', 'paged-attention', 'native-update-attention', 'paged-update-attention')


def _measure(torch, operation, iterations):
    """Stream elapsed includes possible host launch gaps, not just kernel time."""
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    start.record()
    for _ in range(iterations):
        result = operation()
    stop.record()
    stop.synchronize()
    wall_us = (time.perf_counter() - begin) * 1e6 / iterations
    stream_us = start.elapsed_time(stop) * 1000 / iterations
    del result
    return stream_us, wall_us


def _summary(samples):
    from benchmarks.common import percentile

    result = {}
    for name in ('stream_us', 'wall_us'):
        values = [sample[name] for sample in samples]
        result[name] = {'mean': sum(values) / len(values), 'p50': percentile(values, .5),
                        'p95': percentile(values, .95)}
    return result


def _point(args, torch, monitor, prefix, rows, destination):
    from timesfm.torch.transformer import _torch_dot_product_attention, make_attn_mask
    from vtsfm.adapters.timesfm25.paged_cache import PagedCacheArena

    query_length, heads, head_size = 4, 16, 80
    total = prefix + query_length
    record = {
        'status': 'running', 'prefix_patches': prefix, 'context_observations': prefix * 32,
        'rows': rows, 'query_patches': query_length, 'heads': heads, 'head_size': head_size,
        'samples_per_round': args.samples, 'iterations_per_sample': args.iterations,
        'warmups_per_path_per_round': args.warmups, 'repetitions': args.repetitions,
        'execution_mode': 'cuda_graph_replay' if args.cuda_graphs else 'eager',
        'data': 'Synthetic precomputed Q/K/V, CPU generator seed 140000 + prefix*100 + rows; randn/3.',
        'timing_boundary': 'Fixed-cache replay of attention only, or four-slot update plus attention. Metadata, leases, tensor preparation and QKV/MLP/head/normalization excluded. CUDA stream elapsed includes possible CPU launch gaps; it is not summed kernel duration. Wall time includes final event synchronization, amortized over iterations.',
        'layout_scope': (
            'The dense SDPA control uses a compact prefix+4 contiguous cache and an explicit causal mask. '
            'The optimized full-forecast dense path uses the full padded cache width. Paged attention uses '
            'the same valid positions in newly leased pages. Eager paths include output allocation. Graph mode '
            'captures allocation outside timing and retains static outputs. Fragmentation and prefix sharing '
            'are outside this diagnostic.'
        ),
        'memory_scope': 'Both paths and inputs coexist; fixed pool bytes describe geometry only, not matched process/peak memory.',
        'rounds': [],
    }
    arena = None
    try:
        _assert_gpu_owned(monitor, record)
        generator = torch.Generator(device='cpu').manual_seed(140000 + prefix * 100 + rows)
        cpu_query = torch.randn(rows, query_length, heads, head_size, generator=generator) / 3
        cpu_key, cpu_value = [torch.randn(rows, total, heads, head_size, generator=generator) / 3
                              for _ in range(2)]
        record['input_sha256'] = {name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                                   for name, tensor in [('query', cpu_query), ('key', cpu_key), ('value', cpu_value)]}
        query, key, value = [tensor.cuda() for tensor in (cpu_query, cpu_key, cpu_value)]
        packed_query = query.reshape(-1, heads, head_size)
        append_key = key[:, prefix:].contiguous()
        append_value = value[:, prefix:].contiguous()
        packed_key = append_key.reshape(-1, heads, head_size)
        packed_value = append_value.reshape(-1, heads, head_size)
        dense_key, dense_value = key.clone(), value.clone()
        mask = make_attn_mask(query_length, torch.zeros(rows, dtype=torch.int32, device='cuda'),
                              torch.full((rows,), prefix, dtype=torch.int32, device='cuda'), total)
        arena = PagedCacheArena(1, heads, head_size, rows, total, streams=1)
        identities = tuple(f'micro-row-{row}' for row in range(rows))
        for identity in identities:
            arena.reserve(identity, total)
        record['fixed_pool_bytes'] = arena.cache_bytes
        # Populate the prefix using the validated arena API once, outside timing.
        history_key = key[:, :prefix].contiguous().reshape(-1, heads, head_size)
        history_value = value[:, :prefix].contiguous().reshape(-1, heads, head_size)
        arena.attend(0, 0, torch.zeros_like(history_key), history_key, history_value,
                     identities, (0,) * rows, (prefix,) * rows)
        cumulative = tuple(index * query_length for index in range(rows + 1))
        metadata = arena._metadata(identities, (prefix,) * rows, (query_length,) * rows,
                                   cumulative, (total,) * rows)
        cache = arena.caches[arena._names[0, 0]]

        def native_attention():
            return _torch_dot_product_attention(query, dense_key, dense_value, mask=mask)

        def native_update_attention():
            dense_key[:, prefix:].copy_(append_key)
            dense_value[:, prefix:].copy_(append_value)
            return native_attention()

        def paged_attention():
            paged_output = torch.empty_like(packed_query)
            arena.impl.forward(arena._layer, packed_query, packed_key, packed_value,
                               cache, metadata, output=paged_output)
            return paged_output

        def paged_update_attention():
            arena.impl.do_kv_cache_update(arena._layer, packed_key, packed_value,
                                          cache, metadata.slot_mapping)
            return paged_attention()

        operations = dict(zip(PATHS, (native_attention, paged_attention,
                                     native_update_attention, paged_update_attention), strict=True))
        expected = _torch_dot_product_attention(query, key, value, mask=mask)
        paged_update_attention()
        record['parity'] = {}
        for name, operation in operations.items():
            output = operation().reshape(rows, query_length, heads, head_size)
            torch.testing.assert_close(output, expected, **TOLERANCE)
            record['parity'][name] = dict(TOLERANCE, passed=True,
                                         max_abs_error=float((output - expected).abs().max()))
        del output
        # Compile/warm every path before collecting any comparison samples.
        for operation in operations.values():
            for _ in range(args.warmups):
                operation()
        torch.cuda.synchronize()
        _assert_gpu_owned(monitor, record)
        graphs, static_outputs = {}, {}
        timed_operations = operations
        if args.cuda_graphs:
            # All shapes, buffers and metadata are fixed in this diagnostic.
            # Graphs cover only one attention/update operation, not a forecast.
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for operation in operations.values():
                    for _ in range(max(1, args.warmups)):
                        operation()
            capture_stream.synchronize()
            record['graph_capture'] = {
                'operations': len(operations),
                'scope': 'One fixed operation per graph, captured and parity-checked outside timing. Static outputs retained. This does not enable CUDA graphs for full forecasts or production serving.',
            }
            for name, operation in operations.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    static_outputs[name] = operation()
                graphs[name] = graph
                graph.replay()
                torch.cuda.synchronize()
                output = static_outputs[name].reshape(rows, query_length, heads, head_size)
                torch.testing.assert_close(output, expected, **TOLERANCE)
                record['parity']['captured-' + name] = dict(
                    TOLERANCE, passed=True,
                    max_abs_error=float((output - expected).abs().max()),
                )
            del output
            timed_operations = {name: graph.replay for name, graph in graphs.items()}
            _assert_gpu_owned(monitor, record)
        del expected
        for repetition in range(args.repetitions):
            offset = repetition % len(PATHS)
            order = PATHS[offset:] + PATHS[:offset]
            entry = {'repetition': repetition + 1, 'order': order, 'paths': {}}
            record['rounds'].append(entry)
            for name in order:
                operation = timed_operations[name]
                samples = []
                entry['paths'][name] = {'status': 'running', 'samples': samples}
                try:
                    for _ in range(args.warmups):
                        operation()
                    torch.cuda.synchronize()
                    _assert_gpu_owned(monitor, record)
                    for _ in range(args.samples):
                        _assert_gpu_owned(monitor, record)
                        stream_us, wall_us = _measure(torch, operation, args.iterations)
                        samples.append({'stream_us': stream_us, 'wall_us': wall_us})
                        _assert_gpu_owned(monitor, record)
                    entry['paths'][name].update(status='ok', summary=_summary(samples))
                finally:
                    _write(destination, record)
            _write(destination, record)
        record['summary'] = {name: _summary([sample for entry in record['rounds']
                                             for sample in entry['paths'][name]['samples']])
                             for name in PATHS}
        record['status'] = 'ok'
    except BaseException as error:
        record.update(status='cancelled' if isinstance(error, (SystemExit, KeyboardInterrupt)) else 'error',
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        record.pop('summary', None)
        for entry in record['rounds']:
            for path in entry['paths'].values():
                path.pop('summary', None)
                path['status'] = 'invalidated'
        raise
    finally:
        try:
            if arena is not None:
                arena.close()
                record['leased_cache_bytes_after_cleanup'] = arena.leased_cache_bytes
        except BaseException as error:
            record.update(status='error', cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            _write(destination, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--prefixes', type=_integers, default=(16, 64, 256), help='Already cached patch tokens; multiply by 32 for observation context')
    parser.add_argument('--rows', type=_integers, default=(2, 4, 16), help='Internal independent rows, including both signs if applicable')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=50, help='Repeated identical-slot replays per sample')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=4, help='Four rounds balance all path order positions')
    parser.add_argument('--cuda-graphs', action='store_true',
                        help='Capture one fixed operation per path outside timing and measure graph replay; diagnostic only, not full-forecast graph support')
    parser.add_argument('--gpu', type=int, default=0, help='Physical NVML GPU index; expose only this device')
    args = parser.parse_args()
    if (min(args.samples, args.iterations, args.repetitions) < 1 or args.warmups < 0
            or args.gpu < 0 or max(args.prefixes) > 284 or max(args.rows) > 32):
        parser.error('Require positive samples/iterations/repetitions, nonnegative warmups/gpu, prefixes<=284, rows<=32')
    if os.environ.get('VTSFM_RUN_TIMESFM_NATIVE_KV') != '1':
        parser.error('Set VTSFM_RUN_TIMESFM_NATIVE_KV=1 to opt in to the manual GPU diagnostic')
    if os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
        parser.error('Set TRITON_F32_DEFAULT=ieee before launching the diagnostic')
    if os.environ.get('NVIDIA_TF32_OVERRIDE') != '0' or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        parser.error('Set NVIDIA_TF32_OVERRIDE=0 and CUDA_VISIBLE_DEVICES to the selected physical GPU index before launch')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        parser.error('Set CUDA_DEVICE_ORDER=PCI_BUS_ID before launching the diagnostic')
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(_source_metadata(), benchmark='timesfm-precomputed-attention-ablation',
                    gpu_ownership_method=GPU_OWNERSHIP_METHOD,
                    configuration={key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()}, status='running', results=[])
    metadata['source_sha256']['benchmarks/timesfm_attention_ablation.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write(args.output / 'manifest.json', metadata)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))
    monitor = None
    try:
        from benchmarks.run import GpuMonitor, ensure_worker_gpu
        monitor = GpuMonitor(args.gpu)
        monitor.assert_idle()  # before Torch/model imports and CUDA initialization
        import torch
        from timesfm.torch.transformer import _torch_dot_product_attention

        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        properties = torch.cuda.get_device_properties(0)
        ensure_worker_gpu({'gpu': {'uuid': getattr(properties, 'uuid', None)}}, monitor.uuid, require_uuid=True)
        metadata.update(python=sys.version, gpu=monitor.metadata(), packages=_installed_metadata(),
                        torch_cuda_runtime=torch.version.cuda,
                        precision={'dtype': 'float32', 'TRITON_F32_DEFAULT': os.environ['TRITON_F32_DEFAULT'],
                                   'NVIDIA_TF32_OVERRIDE': os.environ['NVIDIA_TF32_OVERRIDE'],
                                   'matmul_precision': torch.get_float32_matmul_precision(),
                                   'num_threads': torch.get_num_threads()},
                        native_function_sha256=hashlib.sha256(textwrap.dedent(inspect.getsource(_torch_dot_product_attention)).strip().encode()).hexdigest())
        _write(args.output / 'manifest.json', metadata)
        with torch.inference_mode():
            for prefix in args.prefixes:
                for rows in args.rows:
                    destination = args.output / f'P{prefix}-R{rows}.json'
                    record = _point(args, torch, monitor, prefix, rows, destination)
                    metadata['results'].append({'path': destination.name, 'status': record['status'],
                                                'summary': record['summary']})
                    _write(args.output / 'manifest.json', metadata)
                    print(json.dumps(metadata['results'][-1]), flush=True)
                    torch.cuda.empty_cache()
        metadata['status'] = 'ok'
        return 0
    except BaseException as error:
        metadata.update(status='cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'error',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if monitor is not None:
            monitor.close()
        metadata['finished_at'] = time.time()
        _write(args.output / 'manifest.json', metadata)
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
