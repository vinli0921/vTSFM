"""Numerical forecast IO extension point, preserving the upstream schema."""

from chronos.chronos2.vllm.io_processor import Chronos2IOProcessor


class VTSFMIOProcessor(Chronos2IOProcessor):
    """Inherit tested preprocessing, covariates, grouping, and quantile outputs.

    Future bounded batching and cross-request packing must preserve task groups
    and target-to-response mappings. This starter implements neither change.
    """
