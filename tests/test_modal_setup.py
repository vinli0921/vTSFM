"""CI must refuse GPU submission until the owner configures billing and tokens."""

import pytest


@pytest.mark.parametrize('value', [None, '', 'false', 'True', '1'])
def test_modal_submission_requires_explicit_enablement(value):
    from ci.gpu_checks import require_setup

    env = {'MODAL_TOKEN_ID': 'test-id', 'MODAL_TOKEN_SECRET': 'test-secret'}
    if value is not None:
        env['MODAL_GPU_ENABLED'] = value
    with pytest.raises(RuntimeError, match='MODAL_GPU_ENABLED'):
        require_setup(env)


@pytest.mark.parametrize('missing', ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET'])
def test_modal_submission_rejects_missing_credentials_without_disclosing_other_values(missing):
    from ci.gpu_checks import require_setup

    env = {'MODAL_GPU_ENABLED': 'true', 'MODAL_TOKEN_ID': 'private-token-id',
           'MODAL_TOKEN_SECRET': 'private-token-secret'}
    env[missing] = ''
    with pytest.raises(RuntimeError, match=missing) as error:
        require_setup(env)
    assert 'private-token' not in str(error.value)


def test_enabled_and_configured_submission_passes_local_setup_check():
    from ci.gpu_checks import require_setup

    require_setup({'MODAL_GPU_ENABLED': 'true', 'MODAL_TOKEN_ID': 'test-id',
                   'MODAL_TOKEN_SECRET': 'test-secret'})
