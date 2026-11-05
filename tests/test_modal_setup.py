"""CI must refuse GPU submission until the owner configures billing and tokens."""

import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest


class RuntimeVersion(str):
    """Like TorchVersion, this subclass would need its module on the client."""


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


def test_gpu_result_metadata_unpickles_without_runtime_or_test_packages():
    from ci.gpu_checks import runtime_metadata

    runtime = SimpleNamespace(__version__=RuntimeVersion('2.9.1+cu128'),
        version=SimpleNamespace(cuda='12.8'),
        cuda=SimpleNamespace(get_device_name=lambda index: 'fixture GPU'))
    info = runtime_metadata(runtime)
    # An isolated client cannot import this test's RuntimeVersion (or Torch).
    result = subprocess.run([sys.executable, '-I', '-c',
        'import pickle,sys; data=pickle.loads(sys.stdin.buffer.read()); '
        'assert data == {"gpu":"fixture GPU","torch":"2.9.1+cu128","cuda":"12.8"}'],
        input=pickle.dumps(info), capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
