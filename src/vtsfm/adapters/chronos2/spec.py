"""Chronos-2 pins, implementations and conservative reference launch settings."""

from vtsfm.adapters.base import AdapterSpec

MODEL_ID = 'amazon/chronos-2'
MODEL_REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'
CHRONOS_COMMIT = 'ba47d25a04aedac33af2fb9780609afd54744d5e'

CHRONOS2 = AdapterSpec(
    name='chronos2',
    model_id=MODEL_ID,
    model_revision=MODEL_REVISION,
    architectures=('Chronos2Model',),
    model_class='vtsfm.adapters.chronos2.model:VTSFMChronos2ForForecasting',
    io_processor_class='vtsfm.adapters.chronos2.io_processor.VTSFMChronos2IOProcessor',
    upstream_io_plugin='chronos2',
    native_server_module='vtsfm.adapters.chronos2.native_server',
    vllm_args=(
        '--runner', 'pooling', '--enforce-eager', '--no-enable-prefix-caching',
        '--skip-tokenizer-init', '--enable-mm-embeds', '--dtype', 'float32',
        '--max-model-len', '8192', '--max-num-batched-tokens', '8192',
        '--max-num-seqs', '1',
    ),
    supports_group_attention=True,
)
