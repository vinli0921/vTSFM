"""Whole forecasts with CPU-known KV positions and an optional compiled neural stage."""

from vtsfm.compilation import compile_method

from .block_decode import block_decode_forward
from .stepping import TimesFMStepper


# Preserve default Inductor fusion while keeping explicit CUDA graphs separate.
INDUCTOR_OPTIONS = {'triton.cudagraphs': False}


class PositionedForecast(TimesFMStepper):
    """Serialize prefill and decode with each call owning its normal/flipped KV."""

    def __init__(self, runtime, *, compiled):
        super().__init__(runtime)
        if compiled:
            # Prefill always writes at position zero. Compile that stage once per
            # input shape and keep the later CPU positions on the eager path.
            compile_method(self, '_prefill_forward', inductor_options=INDUCTOR_OPTIONS)

    def _forward(self, inputs, masks, caches, positions, *, prefill):
        return block_decode_forward(self.model, inputs, masks, caches, positions,
                                    prefill=prefill, optimize_cache=False)

    def _prefill_forward(self, inputs, masks, caches):
        outputs, groups = self._forward(inputs, masks, [caches], [0], prefill=True)
        return outputs, groups[0]

    def _block_decode_forward(self, states, inputs, masks, cache_groups, positions):
        return self._forward(inputs, masks, cache_groups, positions, prefill=False)

    def run(self, context, observed, horizon):
        state = self.initialize(context.masked_fill(~observed, float('nan')), horizon)
        try:
            self.prefill([state])
            while not self.complete(state):
                self.advance([state])
            return self.finalize(state)
        finally:
            self.release(state)
