"""Chronos-specific numerical preparation, group semantics and output schema."""

from chronos.chronos2.vllm.io_processor import Chronos2IOProcessor


class VTSFMChronos2IOProcessor(Chronos2IOProcessor):
    """Preserve upstream preprocessing, covariates and probabilistic outputs.

    A future execution adapter can supply prepared group sizes to shared batching.
    Patching, normalization and target/covariate handling remain family-specific.
    """
