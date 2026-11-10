"""Native origin shapes stay local to one Chronos calibration call."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
F = pytest.importorskip('torch.nn.functional')
TorchFunctionMode = pytest.importorskip('torch.overrides').TorchFunctionMode
ChronosOriginMode = pytest.importorskip('vtsfm.recipes._chronos_ops').ChronosOriginMode


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)
        self.block = torch.nn.Module()
        self.block.layer = torch.nn.ModuleList([
            torch.nn.Identity(), torch.nn.ModuleDict({'projection': torch.nn.Linear(3, 4)})])


def pipeline():
    return SimpleNamespace(model=Model(), model_context_length=8, model_prediction_length=4)


def test_normalization_preserves_ragged_widths_and_capped_unrolled_paths():
    native = pipeline()
    rows = [torch.arange(5, dtype=torch.float32)[None],
            torch.arange(6, dtype=torch.float32)[None] + 20]
    padded = torch.cat([F.pad(rows[0], (1, 0), value=float('nan')), rows[1]])
    expected_loc = torch.cat([row.nanmean(-1, keepdim=True) for row in rows])
    expected_var = torch.cat([(row - loc).square().nanmean(-1, keepdim=True)
                             for row, loc in zip(rows, expected_loc)])
    with torch.no_grad(), ChronosOriginMode(native, [5, 6]):
        loc = torch.nanmean(padded, dim=-1, keepdim=True)
        variance = (padded - loc).square().nanmean(dim=-1, keepdim=True)
        torch.testing.assert_close(loc, expected_loc, atol=0, rtol=0)
        torch.testing.assert_close(variance, expected_var, atol=0, rtol=0)
        # Nine paths per origin follow the native long-horizon expansion. Both
        # original widths plus the first block now reach the context cap.
        paths = torch.arange(18 * 8, dtype=torch.float32).reshape(18, 8)
        paths_loc = paths.nanmean(-1, keepdim=True)
        paths_var = (paths - paths_loc).square().nanmean(-1, keepdim=True)
    torch.testing.assert_close(paths_loc, paths.mean(-1, keepdim=True), atol=0, rtol=0)
    torch.testing.assert_close(paths_var, (paths - paths_loc).square().mean(-1, keepdim=True),
                               atol=0, rtol=0)
    assert torch.isnan(padded[0, 0])


def test_projection_axes_restore_native_geometry_without_model_mutation():
    native = pipeline()
    modules = list(native.model.modules())
    forwards = [module.forward for module in modules]
    parameters = {name: (parameter, parameter.clone())
                  for name, parameter in native.model.named_parameters()}
    calls = []

    class Observe(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is F.linear:
                calls.append(tuple(args[0].shape))
            return func(*args, **(kwargs or {}))

    time = torch.randn(2, 5, 3)
    grouped = time.transpose(0, 1)
    with torch.no_grad():
        expected = torch.cat([native.model.projection(row) for row in time.split(1)])
        expected_group = torch.cat([native.model.block.layer[1]['projection'](row.contiguous())
                                    for row in grouped.split(1, dim=1)], dim=1)
        with Observe(), ChronosOriginMode(native, [5, 6]):
            actual = native.model.projection(time)
            actual_group = native.model.block.layer[1]['projection'](grouped)
    assert calls == [(1, 5, 3), (1, 5, 3), (5, 1, 3), (5, 1, 3)]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_group, expected_group, atol=0, rtol=0)
    assert [module.forward for module in modules] == forwards
    for name, parameter in native.model.named_parameters():
        original, value = parameters[name]
        assert parameter is original
        torch.testing.assert_close(parameter, value, atol=0, rtol=0)


def test_invalid_schedule_unwinds_scope_and_native_calls_remain_available():
    native = pipeline()
    values = torch.randn(2, 7)
    expected = values.nanmean(-1, keepdim=True)
    with pytest.raises(ValueError, match='native schedule'):
        with ChronosOriginMode(native, [5, 6]):
            values.nanmean(-1, keepdim=True)
    torch.testing.assert_close(values.nanmean(-1, keepdim=True), expected, atol=0, rtol=0)
    # The mode cannot turn a training invocation into detached calibration.
    with pytest.raises(RuntimeError, match='inference-only'):
        with ChronosOriginMode(native, [5, 6]):
            native.model.projection(torch.randn(2, 5, 3))
    assert native.model.projection(torch.randn(2, 5, 3)).requires_grad


@pytest.mark.gpu
def test_cuda_normalization_matches_native_allocations_for_unaligned_rows():
    if not torch.cuda.is_available():
        pytest.skip('CUDA normalization regression needs a GPU')
    native = pipeline()
    native.model_context_length = 8192
    generator = torch.Generator(device='cuda').manual_seed(42)
    padded = 50 + torch.randn(2, 922, device='cuda', generator=generator) * 0.1
    padded[0, 0] = float('nan')
    rows = [padded[0:1, 1:].clone(), padded[1:2].clone()]
    expected_loc = torch.cat([row.nanmean(-1, keepdim=True) for row in rows])
    expected_var = torch.cat([(row - loc).square().nanmean(-1, keepdim=True)
                             for row, loc in zip(rows, expected_loc)])
    with torch.no_grad(), ChronosOriginMode(native, [921, 922]):
        loc = padded.nanmean(-1, keepdim=True)
        variance = (padded - loc).square().nanmean(-1, keepdim=True)
    torch.testing.assert_close(loc, expected_loc, atol=0, rtol=0)
    torch.testing.assert_close(variance, expected_var, atol=0, rtol=0)
