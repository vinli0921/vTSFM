"""Lightweight discovery hooks loaded in every vLLM worker process."""


def register_models() -> None:
    # Keep GPU/model imports lazy: vLLM may discover plugins before spawning.
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        'Chronos2Model',
        'vtsfm.models.chronos2:VTSFMChronos2ForForecasting',
    )


def get_io_processor() -> str:
    return 'vtsfm.io_processor.VTSFMIOProcessor'
