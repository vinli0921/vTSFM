"""CI must refuse GPU submission until the owner configures billing and tokens."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest


class RuntimeVersion(str):
    """Like TorchVersion, this subclass would need its module on the client."""


@pytest.mark.parametrize('event_name,event,acknowledged,allowed', [
    ('pull_request', {'pull_request': {'head': {'repo': {'full_name': 'team/vtsfm'}}}}, '', True),
    ('pull_request', {'pull_request': {'head': {'repo': {'full_name': 'fork/vtsfm'}}}}, 'true', False),
    ('pull_request', {'pull_request': {'head': {'repo': None}}}, 'true', False),
    ('workflow_dispatch', {}, 'true', True),
    ('workflow_dispatch', {}, '', False),
    ('workflow_dispatch', {}, 'false', False),
    ('push', {}, 'true', False),
    ('pull_request_target', {}, 'true', False),
])
def test_ci_only_accepts_same_repository_prs_or_acknowledged_manual_runs(
        event_name, event, acknowledged, allowed, tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = tmp_path / 'event.json'
    payload.write_text(json.dumps(event))
    environment = {**os.environ, 'GITHUB_ACTIONS': 'true', 'GITHUB_EVENT_NAME': event_name,
                   'GITHUB_REPOSITORY': 'team/vtsfm', 'GITHUB_EVENT_PATH': str(payload),
                   'BILLING_LIMITS_VERIFIED': acknowledged, 'MODAL_GPU_ENABLED': 'true',
                   'MODAL_TOKEN_ID': 'private-token-id', 'MODAL_TOKEN_SECRET': 'private-token-secret'}
    result = subprocess.run([sys.executable, '-m', 'ci.gpu_checks'], cwd=root,
                            env=environment, capture_output=True, text=True)
    assert (result.returncode == 0) is allowed, result.stderr
    assert 'private-token' not in result.stdout + result.stderr


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
