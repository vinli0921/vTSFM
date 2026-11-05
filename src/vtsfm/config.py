"""Runtime pins for the numerical adapters and the historical Chronos benchmark."""

from vtsfm.adapters.chronos2.spec import (
    CHRONOS_COMMIT as CHRONOS_COMMIT,
    MODEL_ID as MODEL_ID,
    MODEL_REVISION as MODEL_REVISION,
)

VLLM_VERSION = '0.33.0'
TORCH_VERSION = '2.14.0'
TRANSFORMERS_VERSION = '5.20.2'
CUDA_RUNTIME = '13.0'
PYTHON_VERSION = '3.11.17'
DEFAULT_PORT = 8001


def serving_pins(adapter: str) -> dict[str, str]:
    if adapter == 'chronos2':
        return {'vllm': '0.16.0', 'torch': '2.9.1', 'transformers': '4.57.6', 'cuda_runtime': '12.8'}
    return {'vllm': VLLM_VERSION, 'torch': TORCH_VERSION,
            'transformers': TRANSFORMERS_VERSION, 'cuda_runtime': CUDA_RUNTIME}
