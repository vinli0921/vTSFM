"""Chronos-2 model execution and the boundary for family-specific optimizations.

The current forward pass and weight loading remain inherited from upstream.
Future grouped attention changes belong here, not in the shared batch planner.
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
    """Register the processor on the concrete subclass used by vLLM."""
