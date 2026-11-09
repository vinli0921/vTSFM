"""Opt-in guarded neural compilation; importable without Torch or model packages.

This first implementation retains native normalization, attention and activation
boundaries as eager calls. Graph breaks are intentional. It is not the opaque
C++ backend qualified in the research copy, and has no serving speedup claim.
"""

import hashlib
import inspect
import os

from vtsfm.config import TORCH_VERSION


EXECUTION_ENV = 'VTSFM_FORECAST_EXECUTION'
EXECUTION_MODES = ('eager', 'compile')
_TARGETS = {'timesfm25': (), 'toto2': ('transformer',), 'moirai2': ('encoder',)}
_STATE = '_vtsfm_compiled_forecast'
_BUDGET = {'recompile_limit': 64, 'accumulated_recompile_limit': 256,
           'fail_on_recompile_limit_hit': True, 'suppress_errors': False}
_OPTIONS = {'pattern_matcher': False, 'epilogue_fusion': False,
            'prologue_fusion': False, 'triton.cudagraphs': False}

# These method bodies come from the pinned third-party source revisions. Refuse
# an unfamiliar implementation instead of guessing which arithmetic is safe.
_NATIVE_SOURCES = {
    'timesfm25': {
        'timesfm.torch.normalization.RMSNorm':
            '47a5e7b8eef7fcb9e57c8a4dd9841dba54807ecae59087414f816ef110fe5710',
        'timesfm.torch.transformer.MultiHeadAttention':
            'b8854e6195987e06fe13a6f20b8f2bea37a5bafc3fd9a773a2e9436444af61fa',
    },
    'toto2': {
        'dd_unit_scaling._modules.RMSNorm':
            '4de90b83c6ea213ab4658d0ae78af196fe3cc6f05a506db6ebdf73d64f6837b5',
        'toto2.model.SelfAttention':
            '3522f5428bcfaeccf235c191a49a34897ca33602054b272118bd9aa547288401',
        'toto2.model.GatedLinearUnitFeedForwardNetwork':
            '60eb777202c599c01def0c7395e7d3535361b991f5d6a9f25499aca1545c1cd8',
    },
    'moirai2': {
        'uni2ts.module.norm.RMSNorm':
            '289a06ee47ef58558f96a70ccd665932afa8b9f8143ffa54f52bb32635890d2d',
        'uni2ts.module.attention.GroupedQueryAttention':
            '0cbd7ac8bcde724a399390d4e4cf48f877dc341e5eede35561e2412c2a33d5c3',
        'uni2ts.module.attention.MultiQueryAttention':
            '0cbd7ac8bcde724a399390d4e4cf48f877dc341e5eede35561e2412c2a33d5c3',
        'uni2ts.module.attention.MultiHeadAttention':
            '0cbd7ac8bcde724a399390d4e4cf48f877dc341e5eede35561e2412c2a33d5c3',
        'uni2ts.module.ffn.FeedForward':
            'a3deccf283285378e45e4d1262a113760cb46389475f365f6356118016f86663',
        'uni2ts.module.ffn.GatedLinearUnitFeedForward':
            'a3deccf283285378e45e4d1262a113760cb46389475f365f6356118016f86663',
    },
}


def validate_execution(adapter, mode, *, continuous=False):
    if mode not in EXECUTION_MODES:
        raise ValueError(f'Unsupported forecast execution mode: {mode!r}')
    if mode == 'eager':
        return
    if adapter not in _TARGETS:
        raise ValueError('Compiled forecasts support TimesFM, Toto, and Moirai only')
    if continuous:
        raise ValueError('Compiled forecasts do not yet support continuous batching')
    if os.environ.get('VTSFM_PRECISION', 'fp32') != 'fp32':
        raise ValueError('Guarded compiled forecasts require fp32 precision')


def _restore(changes):
    for module, original, had_local in reversed(changes):
        if had_local:
            module.forward = original
        else:
            delattr(module, 'forward')


class CompiledForecast:
    """Own the installed callable; restore native methods explicitly on close.

    Settings are active during lazy compilation and every forward, then restored.
    A worker executes these calls serially; concurrent calls into one instance
    are unsupported. A fresh worker is required after a failed compiled forward.
    """

    def __init__(self, target, compiled, changes, torch, adapter, native_count):
        self.target, self.compiled = target, compiled
        self.changes, self.torch = changes, torch
        self.adapter, self.native_count = adapter, native_count
        self.calls = 0
        self.closed = False

    def __call__(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError('Compiled forecast has been closed')
        if self.torch.is_grad_enabled():
            raise RuntimeError('Compiled forecasts require inference/no-grad execution')
        # Keep the hard-fail specialization policy active for lazy execution,
        # not just torch.compile(), which does not compile immediately.
        with self.torch._dynamo.config.patch(_BUDGET):
            result = self.compiled(*args, **kwargs)
        self.calls += 1
        return result

    def describe(self):
        return {'mode': 'compile', 'backend': 'guarded eager boundaries',
                'active': not self.closed,
                'adapter': self.adapter, 'native_boundaries': self.native_count,
                'successful_core_calls': self.calls, 'specialization_budget': dict(_BUDGET),
                'inductor_options': dict(_OPTIONS), 'intentional_graph_breaks': True,
                'current_stack_gpu_qualified': False}

    def close(self):
        if not self.closed:
            _restore(self.changes)
            self.closed = True


def configure_compilation(runtime, adapter):
    """Install selected neural boundaries without replacing model parameters.

    TimesFM compiles model.forward, Toto transformer.forward, and Moirai
    encoder.forward. Scaling, forecasting recursion and response IO stay eager.
    """
    mode = os.environ.get(EXECUTION_ENV, 'eager')
    validate_execution(adapter, mode, continuous=os.environ.get('VTSFM_CONTINUOUS_BATCHING') == '1')
    if mode == 'eager':
        return None
    if hasattr(runtime, _STATE):
        raise RuntimeError('Compiled forecasts are already installed on this runtime')
    if os.environ.get('TRITON_DEFAULT_FP_FUSION') != '0':
        raise ValueError('Set TRITON_DEFAULT_FP_FUSION=0 before importing Torch')

    import torch

    if torch.__version__.split('+')[0] != TORCH_VERSION:
        raise RuntimeError(f'Compiled forecasts require the pinned Torch {TORCH_VERSION}')
    if (torch.get_float32_matmul_precision() != 'highest'
            or torch.backends.cuda.matmul.allow_tf32
            or os.environ.get('NVIDIA_TF32_OVERRIDE') == '1'):
        raise ValueError('Guarded compiled forecasts require highest FP32 matmul with TF32 disabled')
    for name in _BUDGET:
        if not hasattr(torch._dynamo.config, name):
            raise RuntimeError(f'Pinned Dynamo compiler setting is unavailable: {name}')
    if any(parameter.dtype != torch.float32 for parameter in runtime.model.parameters()):
        raise ValueError('Guarded compiled forecasts require FP32 model parameters')
    target = runtime.model
    for attribute in _TARGETS[adapter]:
        target = getattr(target, attribute)

    native = []
    expected = _NATIVE_SOURCES[adapter]
    covered = set()
    activations = (torch.nn.GELU, torch.nn.ReLU, torch.nn.SiLU)
    for _, module in target.named_modules():
        qualified = type(module).__module__ + '.' + type(module).__name__
        if qualified in expected:
            try:
                digest = hashlib.sha256(inspect.getsource(module.forward).encode()).hexdigest()
            except (OSError, TypeError) as error:
                raise RuntimeError(f'Cannot audit native compiler boundary: {qualified}') from error
            if digest != expected[qualified]:
                raise RuntimeError(f'Pinned native compiler boundary changed: {qualified}')
            covered.add(qualified)
            native.append(module)
        elif type(module) in activations:
            native.append(module)
    # Moirai's attention subclasses and FFN forms are alternatives. Norms and
    # at least one attention class must always have been found in every adapter.
    if (not any(name.endswith('.RMSNorm') for name in covered)
            or not any('Attention' in name for name in covered)):
        raise RuntimeError('Compiled core lacks the required audited native boundaries')
    if ((adapter == 'toto2' and not any(name.endswith('.GatedLinearUnitFeedForwardNetwork')
                                       for name in covered))
            or (adapter == 'moirai2' and not any(name.endswith(('.FeedForward', '.GatedLinearUnitFeedForward'))
                                               for name in covered))
            or (adapter == 'timesfm25' and not any(type(module) in activations for module in native))):
        raise RuntimeError('Compiled core lacks the required native activation boundary')

    changes = []
    try:
        for module in native:
            original, local = module.forward, 'forward' in module.__dict__
            module.forward = torch.compiler.disable(original, recursive=True)
            changes.append((module, original, local))
        compiled = torch.compile(target.forward, backend='inductor', dynamic=False,
                                 fullgraph=False, options=dict(_OPTIONS))
        original, local = target.forward, 'forward' in target.__dict__
        state = CompiledForecast(target, compiled, changes, torch, adapter, len(native))
        target.forward = state
        changes.append((target, original, local))
        setattr(runtime, _STATE, state)
        return state
    except BaseException:
        _restore(changes)
        raise
