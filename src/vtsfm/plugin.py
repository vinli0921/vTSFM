"""Lightweight discovery hooks loaded in every vLLM worker process."""

import os

from vtsfm.adapters import get_adapter, list_adapters


def register_models() -> None:
    # Keep GPU/model imports lazy: vLLM may discover plugins before spawning.
    from vllm import ModelRegistry

    for adapter in list_adapters():
        for architecture in adapter.architectures:
            ModelRegistry.register_model(architecture, adapter.model_class)
    if os.environ.get('VTSFM_ADAPTER', 'chronos2') in ('timesfm25', 'toto2', 'moirai2'):
        from vllm.transformers_utils.config import register_config_parser
        from vtsfm.adapters.hf_config import ForecastConfigParser

        register_config_parser('vtsfm')(ForecastConfigParser)


def get_io_processor() -> str:
    return get_adapter(os.environ.get('VTSFM_ADAPTER', 'chronos2')).io_processor_class
