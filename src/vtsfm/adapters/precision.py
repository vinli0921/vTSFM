"""Dtype and numerical-state preparation shared by the model adapters."""


def prepare_rotary_tables(module):
    """Preserve upstream CPU-built FP32 tables for BF16 and TF32 forecasts.

    The official loaders construct these buffer-only projections on CPU.
    vLLM constructs on the target device; CUDA sin/cos can differ by one FP32
    ulp, enough to change a later BF16 rounding and recursive prediction.
    Call only before the first forecast, preserving theta and table length.
    """
    device = next(module.buffers()).device
    if device.type == 'cpu':
        return
    length = module.cos.shape[-2]
    module.cpu()
    module.cos = None
    module.sin = None
    module._init_freq(max_len=length)
    module.to(device=device)


def prepare_bf16(model, *, fp32_modules, input_projection, output_projection):
    """Cast compute weights once, retaining FP32 numerical state and public IO.

    Call before the first forecast. Module names and parameters stay in place
    for the normal strict loader. Hooks belong only to this model instance.
    """
    import torch

    # .to(dtype) also casts buffers. Keep exact norm parameters and numerical
    # tables, rather than promoting already-rounded BF16 copies back to FP32.
    parameters = [(parameter, parameter.detach().clone())
                  for module in fp32_modules for parameter in module.parameters()]
    buffers = [(module, name, value) for module in model.modules()
               for name, value in module.named_buffers(recurse=False)
               if value.is_floating_point()]
    model.to(dtype=torch.bfloat16)
    for module in fp32_modules:
        module.float()
    with torch.no_grad():
        for parameter, original in parameters:
            parameter.copy_(original)
    for module, name, original in buffers:
        setattr(module, name, original)

    def input_bf16(module, args):
        return (args[0].to(torch.bfloat16), *args[1:])

    def input_fp32(module, args):
        return (args[0].float(), *args[1:])

    def output_bf16(module, args, output):
        return output.to(torch.bfloat16)

    def output_fp32(module, args, output):
        return output.float()

    handles = [input_projection.register_forward_pre_hook(input_bf16),
               output_projection.register_forward_hook(output_fp32)]
    for module in fp32_modules:
        handles.extend((module.register_forward_pre_hook(input_fp32),
                        module.register_forward_hook(output_bf16)))
    return handles
