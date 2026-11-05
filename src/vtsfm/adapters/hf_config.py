"""Read upstream TSFM configs without requiring Transformers model classes."""

from transformers import PreTrainedConfig
from vllm.transformers_utils.config_parser_base import ConfigParserBase


class ForecastConfigParser(ConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, code_revision=None, **kwargs):
        raw, _ = PreTrainedConfig.get_config_dict(model, revision=revision)
        raw = {key: value for key, value in raw.items() if not key.startswith('_')}
        config = PreTrainedConfig(
            vtsfm_config=raw,
            hidden_size=raw.get('hidden_size', raw.get('d_model')),
            num_hidden_layers=raw.get('num_hidden_layers', raw.get('num_layers')),
            num_attention_heads=raw.get('num_attention_heads', raw.get('num_heads')),
            # Token/pooling bookkeeping only; model execution uses vtsfm_config.
            vocab_size=2, max_position_embeddings=16384, projection_dim=0,
        )
        return raw, config
