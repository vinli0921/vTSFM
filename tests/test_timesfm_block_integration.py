"""Use the shared independent-native HTTP checks with the block-mode server.

Select TimesFM with VTSFM_TEST_ADAPTER=timesfm25 and run this file explicitly
on an idle GPU. The native reference, payloads and tolerances are unchanged.
"""

import os

import pytest

from test_model_integration import test_http_forecast_matches_native  # noqa: F401


pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def block_backend():
    from vtsfm import cli

    if os.environ.get('VTSFM_TEST_ADAPTER') != 'timesfm25':
        pytest.fail('The block integration checks require VTSFM_TEST_ADAPTER=timesfm25')
    original = cli.build_vllm_command

    def command(*args, **kwargs):
        kwargs.update(continuous_batching=True, max_num_seqs=4, forecast_batch_rows=8)
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, 'build_vllm_command', command)
        yield
