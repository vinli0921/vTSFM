"""Runnable Chronos adapter; computation remains in the pinned upstream model.

This class is the extension point for future model execution and attention
optimizations. The starter intentionally changes no forecast mathematics.
"""

from chronos.chronos2.vllm.model import Chronos2ForForecasting
from chronos.chronos2.vllm.multimodal import (
    ChronosInputBuilder,
    ChronosMultiModalProcessor,
    ChronosProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    ChronosMultiModalProcessor,
    info=ChronosProcessingInfo,
    dummy_inputs=ChronosInputBuilder,
)
class VTSFMChronos2ForForecasting(Chronos2ForForecasting):
    """vTSFM-owned model class with inherited upstream forward/weight loading.

    Register the processor on this subclass explicitly: registration on the
    upstream class alone is insufficient for a new model-registry target.
    """
