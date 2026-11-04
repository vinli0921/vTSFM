"""CPU example for the shared contract; not a learned forecasting baseline.

Uses NumPy from the benchmark environment. Real adapters transfer tensors to
their model device during preparation, before building PreparedRequest. They
own Torch collation, normalization, and inference behind execute().
"""

import numpy as np

from vtsfm.execution import ExecutionResult, PreparedBatch


class LastValueAdapter:
    def execute(self, batch: PreparedBatch) -> ExecutionResult:
        context = np.concatenate([request.context for request in batch.requests])
        key = batch.plan.key
        # This deliberately simple fixture requires the last value observed.
        if not all(request.observed_mask[:, -1].all() for request in batch.requests):
            raise ValueError('LastValueAdapter requires the final observation')
        values = np.broadcast_to(context[:, -1, None, None],
            (len(context), max(1, len(key.quantile_levels)), key.prediction_length)).copy()
        return ExecutionResult(batch, values)
