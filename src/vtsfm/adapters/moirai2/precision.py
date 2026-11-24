"""Native BF16 Moirai projections, retaining upstream FP32 forecast arithmetic."""

from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import RotaryProjection

from vtsfm.adapters.precision import prepare_bf16, prepare_rotary_tables


def prepare_native_rotary_tables(model):
    """Use the official CPU-built rotary tables in FP32/TF32 without changing model dtypes."""
    for module in model.modules():
        if isinstance(module, RotaryProjection):
            prepare_rotary_tables(module)


def prepare_native_bf16(model):
    if getattr(model, '_vtsfm_native_bf16', False):
        raise ValueError('Moirai native BF16 is already configured')
    # RoPE multiplies by FP32 tables; cast its output back so Q/K/V still agree
    # at SDPA. Scaling and recursive quantile reduction remain upstream FP32.
    modules = [module for module in model.modules() if isinstance(module, (RMSNorm, RotaryProjection))]
    for module in modules:
        if isinstance(module, RotaryProjection):
            prepare_rotary_tables(module)
    handles = prepare_bf16(model, fp32_modules=modules,
                           input_projection=model.in_proj, output_projection=model.out_proj)
    model._vtsfm_native_bf16 = True
    return handles
