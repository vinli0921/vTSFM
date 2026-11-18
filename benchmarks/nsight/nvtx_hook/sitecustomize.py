"""Issue #16: NVTX ranges around vLLM / vTSFM request-path functions, added from outside the repo.

Active only when VTSFM_NVTX=1 (put this directory on PYTHONPATH). A post-import hook wraps the
listed methods once their module has loaded, so no source file is changed and nothing is imported
early. Ranges use start/end (not push/pop) so they stay correct across `await` in the API server.
"""
import os

if os.environ.get('VTSFM_NVTX') == '1':
    import functools
    import importlib.abc
    import importlib.util
    import inspect
    import sys

    TARGETS = {
        # API server process
        'vllm.entrypoints.pooling.base.serving': {
            'PoolingBaseServing': ['__call__', '_preprocessing', '_postprocessing', '_init_ctx',
                                   '_prepare_generators', '_collect_batch', '_build_response']},
        'vllm.entrypoints.pooling.pooling.io_processor': {
            'PluginWithIOProcessorPlugins': ['get_request_factory_online', 'post_process_online']},
        'vllm.v1.engine.input_processor': {'InputProcessor': ['process_inputs']},
        'vllm.v1.engine.output_processor': {'OutputProcessor': ['process_outputs']},
        'vllm.v1.serial_utils': {'MsgpackEncoder': ['encode'], 'MsgpackDecoder': ['decode']},
        'vtsfm.adapters.pooling': {
            'ForecastIOProcessor': ['parse_data', 'pre_process', 'post_process'],
            'ForecastModel': ['forward'], 'ForecastPooler': ['forward']},
        'vtsfm.adapters.multimodal': {'ForecastMultiModalProcessor': ['apply']},
        'vtsfm.adapters.forecast': {None: ['parse_request', 'format_predictions']},
        'vtsfm.adapters.transport': {None: ['decode_requests']},
        'vtsfm.adapters.result_transport': {None: ['pack_result', 'unpack_result']},
        # EngineCore process
        'vllm.v1.engine.core': {
            'EngineCore': ['step', 'step_with_batch_queue', 'add_request', 'preprocess_add_request'],
            'EngineCoreProc': ['_process_input_queue', '_process_engine_step']},
        'vllm.v1.core.sched.scheduler': {'Scheduler': ['schedule', 'update_from_output']},
        'vllm.v1.worker.gpu_worker': {'Worker': ['execute_model']},
        'vllm.v1.worker.gpu_model_runner': {
            'GPUModelRunner': ['execute_model', '_update_states', '_prepare_inputs', '_preprocess',
                               '_pool', '_execute_mm_encoder', '_gather_mm_embeddings',
                               '_model_forward']},
        'vtsfm.worker': {None: ['slot_request_ids']},
        'vtsfm.block_worker': {None: ['block_slots'], 'ForecastBlockWorker': ['execute_model']},
        'vtsfm.continuous_scheduler': {'ForecastBlockScheduler': ['schedule']},
        'vtsfm.block_pooling': {'BlockExecutor': ['forward'], 'BlockForecastPooler': ['forward']},
        'vtsfm.adapters.execution_batch': {None: ['execute_scheduled']},
        'vtsfm.adapters.timesfm25.runtime': {'TimesFMRuntime': ['forecast', 'execute', '_forecast']},
        'vtsfm.adapters.timesfm25.stepping': {
            'TimesFMStepper': ['initialize', 'prefill', 'advance', 'finalize', 'release',
                               '_merge_caches', '_store', '_finish_block']},
        'vtsfm.adapters.toto2.runtime': {'TotoRuntime': ['forecast']},
        'vtsfm.adapters.moirai2.runtime': {'MoiraiRuntime': ['forecast']},
    }

    def _nvtx():
        import torch
        return torch.cuda.nvtx

    def _wrap(fn, label):
        if getattr(fn, '_i16', False):
            return fn
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def w(*a, **k):
                n = _nvtx()
                rid = n.range_start(label)
                try:
                    return await fn(*a, **k)
                finally:
                    n.range_end(rid)
        else:
            @functools.wraps(fn)
            def w(*a, **k):
                n = _nvtx()
                rid = n.range_start(label)
                try:
                    return fn(*a, **k)
                finally:
                    n.range_end(rid)
        w._i16 = True
        return w

    def _patch(module):
        for cls_name, methods in TARGETS[module.__name__].items():
            owner = module if cls_name is None else getattr(module, cls_name, None)
            if owner is None:
                continue
            for m in methods:
                raw = owner.__dict__.get(m) if cls_name else getattr(owner, m, None)
                if raw is None:
                    continue
                short = module.__name__.split('.')[-1]
                label = f'{cls_name or short}.{m}'
                if isinstance(raw, (staticmethod, classmethod)):
                    setattr(owner, m, type(raw)(_wrap(raw.__func__, label)))
                else:
                    setattr(owner, m, _wrap(raw, label))

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            try:
                _patch(module)
            except Exception as e:  # never break the server over instrumentation
                print(f'[i16 nvtx] patch failed for {module.__name__}: {e!r}', file=sys.stderr)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
