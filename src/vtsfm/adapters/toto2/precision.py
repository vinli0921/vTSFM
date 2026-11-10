"""Native BF16 Toto projections with FP32 normalization and numerical state."""

from vtsfm.adapters.precision import prepare_bf16, prepare_rotary_tables


def prepare_native_tf32(model, rotary_type):
    """Match the official CPU-built tables without changing model dtypes."""
    for module in model.modules():
        if isinstance(module, rotary_type):
            prepare_rotary_tables(module)


def _large_variate_forward(module, state, *args, **kwargs):
    """Avoid Flash's CUDA grid limit only for oversized singleton attention."""
    import torch

    forward = module._vtsfm_native_attention_forward
    if (state.device.type == 'cuda' and state.dtype == torch.bfloat16
            and state.shape[0] > 65535 and state.shape[-2] == 1):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Keep the working efficient backend, with the upstream math fallback.
        # The context restores the caller's flags even if forward raises.
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return forward(state, *args, **kwargs)
    return forward(state, *args, **kwargs)


def prepare_native_bf16(model, norm_type, rotary_type, *, attention_type=None):
    if getattr(model, '_vtsfm_native_bf16', False):
        raise ValueError('Toto native BF16 is already configured')
    if getattr(model, '_kv_cache', None) is not None:
        raise ValueError('Configure Toto precision before its first cached forecast')
    modules = [module for module in model.modules() if isinstance(module, (norm_type, rotary_type))]
    for module in modules:
        if isinstance(module, rotary_type):
            prepare_rotary_tables(module)
    handles = prepare_bf16(model, fp32_modules=modules,
                           input_projection=model.patch_proj, output_projection=model.output_head)
    if attention_type is not None:
        from types import MethodType

        for module in model.modules():
            if isinstance(module, attention_type) and module.is_variate_layer:
                # A bound method follows deepcopy/device moves to this instance;
                # a closure over its original bound forward would not.
                module._vtsfm_native_attention_forward = module.forward
                module.forward = MethodType(_large_variate_forward, module)
    model._vtsfm_native_bf16 = True
    return handles
