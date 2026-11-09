"""Native BF16 Toto projections with FP32 numerical boundaries.

Hooks belong only to the selected model instance. Module hierarchy and parameter
names stay unchanged so the ordinary checkpoint loader still loads strictly.
"""

import torch


def prepare_native_bf16(model, norm_type, rotary_type):
    """Install before checkpoint loading/forecasting; keep public IO in FP32.

    Official forecast buffers and scaler inputs remain FP32. Patch projection is
    the first BF16 boundary; the output head returns FP32 before sinh, inverse
    scaling, sorting and median feedback. Norms retain their original operation
    sequence in FP32. SDPA owns attention/softmax accumulation.
    """
    if getattr(model, '_vtsfm_native_bf16', False):
        raise ValueError('Toto native BF16 is already configured')
    if getattr(model, '_kv_cache', None) is not None:
        raise ValueError('Configure Toto precision before its first cached forecast')
    norms = [module for module in model.modules() if isinstance(module, norm_type)]
    rotary = [module for module in model.modules() if isinstance(module, rotary_type)]
    if not norms or not rotary:
        raise ValueError('Toto precision requires the pinned norm and rotary boundaries')
    # Restore exact values, not float32 copies of already-rounded BF16 tables.
    preserved = [(module, {
        name: tensor.detach().clone() for name, tensor in
        (*module.named_parameters(), *module.named_buffers()) if tensor.is_floating_point()
    }) for module in (*norms, *rotary)]
    # RoPE tables and unit-scaling residual constants are numerical buffers,
    # not compute weights. Keep their original dtype and exact values.
    buffers = [(module, name, value.detach().clone()) for module in model.modules()
               for name, value in module.named_buffers(recurse=False) if value.is_floating_point()]
    model.to(dtype=torch.bfloat16)
    for module, values in preserved:
        module.float()
        current = dict((*module.named_parameters(), *module.named_buffers()))
        with torch.no_grad():
            for name, value in values.items():
                current[name].copy_(value)
    for module, name, value in buffers:
        setattr(module, name, value)

    def projection_input(module, args):
        return (args[0].to(torch.bfloat16), *args[1:])

    def norm_input(module, args):
        return (args[0].float(), *args[1:])

    def norm_output(module, args, output):
        return output.to(torch.bfloat16)

    def head_output(module, args, output):
        return output.float()

    handles = [model.patch_proj.register_forward_pre_hook(projection_input),
               model.output_head.register_forward_hook(head_output)]
    for norm in norms:
        handles.extend((norm.register_forward_pre_hook(norm_input),
                        norm.register_forward_hook(norm_output)))
    model._vtsfm_native_bf16 = True
    return tuple(handles)
