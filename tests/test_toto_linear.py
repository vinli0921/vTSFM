"""Optional Torch CPU checks for Toto's scoped native linear geometry."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

torch = pytest.importorskip('torch')
F = torch.nn.functional
TorchFunctionMode = torch.overrides.TorchFunctionMode
_OriginLinearMode = pytest.importorskip('vtsfm.recipes._toto_linear')._OriginLinearMode


class RecordingMode(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is F.linear:
            value = args[0] if args else kwargs['input']
            self.calls.append((threading.get_ident(), tuple(value.shape)))
        return func(*args, **kwargs)


def test_contiguous_origin_geometry_preserves_keywords_and_inputs():
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(3, 4, 6, generator=generator).transpose(1, 2)
    weight = torch.randn(5, 4, generator=generator)
    bias = torch.randn(5, generator=generator)
    saved = inputs.clone()
    recorder = RecordingMode()
    with torch.no_grad():
        expected = torch.cat([F.linear(part.contiguous(), weight, bias)
                              for part in inputs.split(1)])
        with recorder, _OriginLinearMode(3):
            actual = F.linear(input=inputs, weight=weight, bias=bias)
    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert torch.equal(inputs, saved)
    assert [shape for _, shape in recorder.calls] == [(1, 6, 4)] * 3


def test_flattened_variate_geometry_and_outside_scope():
    inputs, weight = torch.randn(15, 1, 8), torch.randn(7, 8)
    recorder = RecordingMode()
    with torch.no_grad(), recorder:
        before = F.linear(inputs, weight)
        with _OriginLinearMode(3):
            inside = F.linear(inputs, weight)
        after = F.linear(inputs, weight)
    assert torch.equal(before, after)
    assert torch.equal(inside, torch.cat([F.linear(part, weight)
                                         for part in inputs.split(5)]))
    assert [shape for _, shape in recorder.calls] == [
        (15, 1, 8), (5, 1, 8), (5, 1, 8), (5, 1, 8), (15, 1, 8)]


def test_scoped_mode_does_not_split_another_threads_calls():
    inputs, weight = torch.randn(6, 4), torch.randn(5, 4)
    barrier = threading.Barrier(2)
    main_calls, other_calls = RecordingMode(), RecordingMode()
    main_id = threading.get_ident()

    def other_thread():
        with torch.no_grad(), other_calls:
            barrier.wait(timeout=10)
            value = F.linear(inputs, weight)
            barrier.wait(timeout=10)
            return threading.get_ident(), value

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(other_thread)
        with torch.no_grad(), main_calls, _OriginLinearMode(3):
            barrier.wait(timeout=10)
            local = F.linear(inputs, weight)
            barrier.wait(timeout=10)
        other_id, remote = pending.result(timeout=10)
    assert main_id != other_id
    assert main_calls.calls == [(main_id, (2, 4))] * 3
    assert other_calls.calls == [(other_id, (6, 4))]
    assert torch.equal(remote, F.linear(inputs, weight))
    assert torch.equal(local, torch.cat([F.linear(part, weight)
                                        for part in inputs.split(2)]))


def test_exception_leaves_original_linear_path_and_rejects_gradients():
    for value in [0, 1, True, 2.5]:
        with pytest.raises(ValueError):
            _OriginLinearMode(value)
    recorder = RecordingMode()
    with torch.no_grad(), recorder:
        with pytest.raises(ValueError, match='independent origin'):
            with _OriginLinearMode(3):
                F.linear(torch.randn(5, 4), torch.randn(2, 4))
        F.linear(torch.randn(5, 4), torch.randn(2, 4))
    assert [shape for _, shape in recorder.calls] == [(5, 4)]
    with torch.enable_grad(), pytest.raises(RuntimeError, match='inference-only'):
        with _OriginLinearMode(2):
            F.linear(torch.randn(2, 4), torch.randn(2, 4))
