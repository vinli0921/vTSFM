"""The small, runtime-independent contract for a supported TSFM family."""

from dataclasses import dataclass


FORECAST_VLLM_ARGS = (
    '--runner', 'pooling', '--config-format', 'vtsfm', '--enforce-eager',
    '--no-enable-prefix-caching', '--skip-tokenizer-init', '--enable-mm-embeds',
    '--dtype', 'float32', '--max-model-len', '8192',
    '--max-num-batched-tokens', '8192', '--max-num-seqs', '1',
)


@dataclass(frozen=True)
class AdapterSpec:
    """Describe lazy implementations and declared launch settings.

    Group attention is a model capability, not a promise of an optimized kernel.
    Model-specific preparation and execution live behind the dotted paths here.
    """

    name: str
    model_id: str
    model_revision: str
    architectures: tuple[str, ...]
    model_class: str
    io_processor_class: str
    upstream_io_plugin: str | None
    native_server_module: str | None
    vllm_args: tuple[str, ...]
    supports_group_attention: bool = False
