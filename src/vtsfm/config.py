"""Runtime pins and backward-compatible defaults for the Chronos benchmark."""

from vtsfm.adapters.chronos2.spec import (
    CHRONOS_COMMIT as CHRONOS_COMMIT,
    MODEL_ID as MODEL_ID,
    MODEL_REVISION as MODEL_REVISION,
)

VLLM_VERSION = '0.16.0'
TORCH_VERSION = '2.9.1'
TRANSFORMERS_VERSION = '4.57.6'
PYTHON_VERSION = '3.11.17'
DEFAULT_PORT = 8001
