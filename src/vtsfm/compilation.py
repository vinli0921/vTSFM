"""Opt-in neural compilation; importing this module does not import Torch."""

import os


EXECUTION_ENV = 'VTSFM_FORECAST_EXECUTION'
EXECUTION_MODES = ('eager', 'compile')

# These settings come from the retained-shape and arithmetic-drift experiments.
# CUDA Graph replay needs its own buffer lifecycle and is not enabled here.
INDUCTOR_OPTIONS = {
    'pattern_matcher': False,
    'epilogue_fusion': False,
    'prologue_fusion': False,
    'triton.cudagraphs': False,
}


def execution_mode(value=None):
    """An explicit selection overrides the environment; eager remains the default."""
    mode = os.environ.get(EXECUTION_ENV, 'eager') if value is None else value
    if mode not in EXECUTION_MODES:
        raise ValueError(f'Unknown forecast execution mode: {mode!r}')
    return mode


def compile_core(core, native_calls):
    """Compile one adapter-selected forward, retaining native numerical calls.

    ``native_calls`` contains (object, callable attribute name) pairs owned by
    this model instance. No modules/parameters are replaced, so checkpoint keys
    remain unchanged. Lazy compilation and execution errors propagate normally.
    """
    if os.environ.get('TRITON_DEFAULT_FP_FUSION') != '0':
        raise ValueError('Compiled forecasts require TRITON_DEFAULT_FP_FUSION=0 before importing Torch')

    import torch
    from torch._dynamo import config

    # TimesFM's retained A/B/A decode phases exceed the default eight graphs.
    # These bounded, process-wide settings persist for the inference worker.
    config.recompile_limit = 64
    config.accumulated_recompile_limit = 256
    # A legal new serving shape must not kill the worker after cache exhaustion.
    # Dynamo warns and runs uncached frames eagerly; genuine compiler errors
    # still propagate because suppress_errors stays disabled.
    config.fail_on_recompile_limit_hit = False
    config.suppress_errors = False

    originals = []
    try:
        for owner, attribute in native_calls:
            original = getattr(owner, attribute)
            local = attribute in vars(owner)
            setattr(owner, attribute, torch.compiler.disable(original, recursive=True))
            originals.append((owner, attribute, original, local))
        core.forward = torch.compile(core.forward, backend='inductor', dynamic=False,
                                     fullgraph=False, options=dict(INDUCTOR_OPTIONS))
    except BaseException:
        for owner, attribute, original, local in reversed(originals):
            if local:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        raise
