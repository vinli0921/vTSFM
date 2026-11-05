"""Opt-in CUDA checks for the tensor contract; no model or network access.

Run explicitly with ``python -m pytest -m gpu tests/test_cuda_contract.py``.
Selecting these tests without working CUDA is a failure, not a successful skip.
"""

import json

import pytest

from vtsfm.batching import BatchKey, BatchRequest, plan_batches
from vtsfm.execution import ExecutionCapabilities, ExecutionResult, PreparedBatch, PreparedRequest


pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def cuda():
    # Keep CPU-only test collection independent of Torch and CUDA initialization.
    try:
        import torch
    except ImportError:
        pytest.fail('CUDA contract tests require an environment with Torch installed')
    assert torch.cuda.is_available(), 'CUDA contract tests require a working CUDA GPU'
    device = torch.device('cuda:0')
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(device)}))
    yield torch, device
    torch.cuda.synchronize(device)


@pytest.fixture
def cuda_batch(cuda):
    torch, device = cuda
    key = BatchKey('cuda-fixture', 'test/no-checkpoint', 'contract', 4, 2, (0.1, 0.5, 0.9))
    capabilities = ExecutionCapabilities(8, 8, (0.1, 0.5, 0.9),
                                        input_patch_size=2, output_block_size=2)

    def prepare(request_id, rows, item_ids):
        context = torch.tensor(rows, dtype=torch.float32, device=device)
        return PreparedRequest(BatchRequest(request_id, key, (1,) * len(rows)),
            context, (4,) * len(rows), torch.ones_like(context, dtype=torch.bool),
            item_ids, capabilities)

    first = prepare('A', [[1, 2, 3, 4], [5, 6, 7, 8]], ('a0', 'a1'))
    second = prepare('B', [[9, 10, 11, 12]], ('b0',))
    plan, = plan_batches([first.metadata, second.metadata], max_rows=3, max_requests=2)
    batch = PreparedBatch(plan, (first, second))
    context = torch.cat([request.context for request in batch.requests])
    offsets = torch.tensor([-1., 0., 1.], dtype=torch.float32, device=device)
    values = (context[:, -1, None, None] + offsets[None, :, None]).expand(3, 3, 2).clone()
    return batch, values


def test_cuda_matrix_math_executes(cuda):
    torch, device = cuda
    values = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    result = values @ torch.eye(4, device=device)
    assert result.tolist() == [[0., 1., 2., 3.], [4., 5., 6., 7.],
                               [8., 9., 10., 11.], [12., 13., 14., 15.]]


def test_cuda_forecasts_route_distinct_rows_and_quantile_channels(cuda, cuda_batch):
    torch, device = cuda
    batch, values = cuda_batch
    assert batch.plan.group_ids == (0, 1, 2)
    assert [(part.start, part.stop) for part in batch.plan.request_slices] == [(0, 2), (2, 3)]
    routed = ExecutionResult(batch, values).split()

    assert list(routed) == ['A', 'B']
    assert routed['A'].item_ids == ('a0', 'a1')
    assert routed['B'].item_ids == ('b0',)
    assert routed['A'].values.tolist() == [
        [[3., 3.], [4., 4.], [5., 5.]], [[7., 7.], [8., 8.], [9., 9.]],
    ]
    assert routed['B'].values.tolist() == [[[11., 11.], [12., 12.], [13., 13.]]]
    for forecast in routed.values():
        assert forecast.quantile_levels == (0.1, 0.5, 0.9)
        assert forecast.values.device == device
        assert forecast.values.dtype == torch.float32
        assert forecast.values.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()


def test_cuda_results_reject_host_outputs_wrong_precision_and_axes(cuda_batch):
    batch, values = cuda_batch
    for invalid, reason in [(values.cpu(), 'device'), (values.double(), 'dtype'),
                            (values.transpose(1, 2), 'shape')]:
        with pytest.raises(ValueError, match=reason):
            ExecutionResult(batch, invalid)


def test_cuda_context_rejects_a_cpu_observed_mask(cuda_batch):
    batch, _ = cuda_batch
    first = batch.requests[0]
    with pytest.raises(ValueError, match='device'):
        PreparedRequest(first.metadata, first.context, first.context_lengths,
            first.observed_mask.cpu(), first.item_ids, first.capabilities)


def test_cuda_batch_rejects_permuted_request_mapping(cuda_batch):
    batch, _ = cuda_batch
    with pytest.raises(ValueError, match='identity'):
        PreparedBatch(batch.plan, tuple(reversed(batch.requests)))
