"""Exercise the manual runner without model packages, network, or a GPU."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ('timesfm25', 'toto2', 'moirai2')


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'clone with spaces'
    (root / 'scripts').mkdir(parents=True)
    source = ROOT / 'scripts/test-models.sh'
    if source.exists():
        shutil.copy2(source, root / 'scripts/test-models.sh')
    (root / 'tests').mkdir()
    for profile in ('integration', 'accuracy', 'performance'):
        (root / f'tests/test_model_{profile}.py').touch()
    commands = tmp_path / 'commands'
    commands.mkdir()
    uname = commands / 'uname'
    uname.write_text('#!/bin/sh\nprintf "%s\\n" "${STUB_OS:-Linux}"\n')
    uname.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=f'{commands}:/usr/bin:/bin', STUB_CALLS=str(tmp_path / 'calls.jsonl'))
    env.pop('PYTHONPATH', None)
    return root, env


def install_environments(root, failures=None):
    failures = failures or {}
    for adapter in ADAPTERS:
        install_stub_python(root / f'.venv-{adapter}/bin/python', adapter, failures.get(adapter, 0))


def install_stub_python(python, adapter, exit_code=0):
    """A fake interpreter that logs its label, arguments and environment."""
    python.parent.mkdir(parents=True)
    body = f'''import json, os, pathlib, sys
record = {{'adapter': {adapter!r}, 'args': sys.argv[1:], 'cwd': os.getcwd(),
          'profile': os.environ.get('VTSFM_TEST_PROFILE'),
          'data_dir': os.environ.get('VTSFM_TEST_DATA'), 'cases': os.environ.get('VTSFM_TEST_CASES'),
          'performance': {{key: os.environ.get('VTSFM_PERF_' + key) for key in
              ('REQUESTS','WARMUPS','BATCHES','CONTEXTS','CLIENTS','HORIZON','ORDER')}},
          'env': {{k: os.environ.get(k) for k in ('VTSFM_TEST_ADAPTER',
          'VTSFM_TEST_GPU_INDEX', 'VTSFM_TEST_PORT', 'VTSFM_TEST_OUTPUT',
          'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'NVIDIA_TF32_OVERRIDE',
          'OMP_NUM_THREADS', 'PYTHONUNBUFFERED', 'PYTHONPATH')}}}}
with open(os.environ['STUB_CALLS'], 'a') as log:
    log.write(json.dumps(record) + '\\n')
if '--junitxml' in sys.argv:
    target = pathlib.Path(sys.argv[sys.argv.index('--junitxml') + 1])
    target.write_text('<testsuite/>')
print('fake pytest for {adapter}')
raise SystemExit({exit_code})
'''
    python.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable)
                      + ' -c ' + shlex.quote(body) + ' "$@"\n')
    python.chmod(0o755)


def run(root, env, *args, cwd=None):
    return subprocess.run(['/bin/bash', str(root / 'scripts/test-models.sh'), *args],
                          cwd=cwd or root, env=env, capture_output=True, text=True, timeout=10)


def calls(env):
    path = Path(env['STUB_CALLS'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_help_has_no_environment_or_output_side_effects(checkout):
    root, env = checkout
    result = run(root, env, '--help')
    assert result.returncode == 0, result.stderr
    assert '--adapter' in result.stdout
    assert not (root / 'results').exists()
    assert calls(env) == []


@pytest.mark.parametrize('profile', ['accuracy', 'performance'])
def test_profiles_select_their_tests_and_forward_frozen_settings(checkout, profile):
    root, env = checkout
    install_environments(root)
    manifest = root / 'frozen cases.json'
    manifest.write_text('{"cases":[]}')
    arguments = ['--profile', profile, '--adapter', 'timesfm25', '--data-dir', 'input cache']
    if profile == 'accuracy':
        arguments += ['--cases', str(manifest)]
    else:
        arguments += ['--requests', '12', '--warmups', '2', '--batches', '2,4',
                      '--contexts', '128', '--clients', '1,2', '--horizon', '32',
                      '--backend-order', 'vtsfm,native']
    result = run(root, env, *arguments)
    assert result.returncode == 0, result.stderr
    record, *summary = calls(env)
    if profile == 'accuracy':
        # Accuracy runs end with one table over the whole run directory. Without
        # a CPU .venv, the selected adapter's interpreter builds it.
        output = record['env']['VTSFM_TEST_OUTPUT'].rsplit('/', 1)[0]
        assert [entry['args'] for entry in summary] == [
            ['-m', 'benchmarks.accuracy_table', output, '--output', output]]
        assert summary[0]['cwd'] == str(root) and summary[0]['adapter'] == 'timesfm25'
    else:
        assert summary == []
    assert record['profile'] == profile
    assert f'tests/test_model_{profile}.py' in record['args']
    assert record['data_dir'] == str(root / 'input cache')
    if profile == 'accuracy':
        assert record['cases'] == str(manifest)
    else:
        assert record['performance'] == {'REQUESTS':'12', 'WARMUPS':'2', 'BATCHES':'2,4',
            'CONTEXTS':'128', 'CLIENTS':'1,2', 'HORIZON':'32', 'ORDER':'vtsfm,native'}


@pytest.mark.parametrize('arguments', [('--profile','unknown'), ('--requests','0'),
    ('--warmups','-1'), ('--batches','0,4'), ('--clients','one'), ('--horizon','1025'),
    ('--backend-order','native,native'), ('--cases','anything.json')])
def test_invalid_profile_settings_do_not_start_model_processes(checkout, arguments):
    root, env = checkout
    install_environments(root)
    result = run(root, env, *arguments)
    assert result.returncode != 0
    assert calls(env) == []
    assert not (root / 'results').exists()


def test_all_models_continue_after_failure_and_preserve_logs_and_exit_codes(checkout):
    root, env = checkout
    install_environments(root, failures={'toto2': 7})
    output = root.parent / 'output with spaces'
    result = run(root, env, '--gpu', '2', '--port', '19002', '--output', str(output))
    assert result.returncode != 0
    records = calls(env)
    assert [record['adapter'] for record in records] == list(ADAPTERS)
    assert (output / 'summary.tsv').read_text().splitlines() == [
        'adapter\tpytest_exit_code\tlog_exit_code',
        'timesfm25\t0\t0', 'toto2\t7\t0', 'moirai2\t0\t0',
    ]
    for record in records:
        adapter = record['adapter']
        folder = output / adapter
        assert record['cwd'] == str(root)
        assert record['args'] == ['-m', 'pytest', '-m', 'gpu',
            'tests/test_model_integration.py', '-q', '-s', '--junitxml', str(folder / 'junit.xml')]
        assert record['env'] == {
            'VTSFM_TEST_ADAPTER': adapter, 'VTSFM_TEST_GPU_INDEX': '2',
            'VTSFM_TEST_PORT': '19002', 'VTSFM_TEST_OUTPUT': str(folder),
            'CUDA_VISIBLE_DEVICES': '2', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'NVIDIA_TF32_OVERRIDE': '0', 'OMP_NUM_THREADS': '4',
            'PYTHONUNBUFFERED': '1', 'PYTHONPATH': f'{root}/src:{root}',
        }
        assert (folder / 'pytest.log').read_text() == f'fake pytest for {adapter}\n'
        assert (folder / 'junit.xml').exists()


def test_one_adapter_uses_its_own_environment_without_cpu_venv(checkout):
    root, env = checkout
    install_environments(root)
    for adapter in ('timesfm25', 'moirai2'):
        shutil.rmtree(root / f'.venv-{adapter}')
    result = run(root, env, '--adapter', 'toto2')
    assert result.returncode == 0, result.stderr
    record, = calls(env)
    assert record['adapter'] == 'toto2'
    folder = Path(record['env']['VTSFM_TEST_OUTPUT']).parent
    assert folder.parent == root / 'results/model-integration'
    assert re.fullmatch(r'\d{8}T\d{6}Z-[a-zA-Z0-9]+', folder.name)
    assert not (root / '.venv').exists()


def test_validates_every_selected_environment_before_launching_any_model(checkout):
    root, env = checkout
    install_environments(root)
    shutil.rmtree(root / '.venv-moirai2')
    output = root / 'results/missing'
    result = run(root, env, '--output', str(output))
    assert result.returncode != 0
    assert 'setup.sh --adapter moirai2' in result.stderr
    assert calls(env) == []
    assert not output.exists()


def test_relative_output_is_resolved_from_callers_directory(checkout):
    root, env = checkout
    install_environments(root)
    result = run(root, env, '--adapter', 'timesfm25', '--output', 'relative output', cwd=root.parent)
    assert result.returncode == 0, result.stderr
    record, = calls(env)
    assert record['env']['VTSFM_TEST_OUTPUT'] == str(root.parent / 'relative output/timesfm25')


def test_refuses_to_overwrite_an_existing_output_directory(checkout):
    root, env = checkout
    install_environments(root)
    output = root / 'results/previous'
    output.mkdir(parents=True)
    marker = output / 'keep.txt'
    marker.write_text('old results')
    result = run(root, env, '--output', str(output))
    assert result.returncode != 0
    assert calls(env) == []
    assert marker.read_text() == 'old results'


@pytest.mark.parametrize('args', [
    ('--adapter', 'chronos2'), ('--gpu', '-1'), ('--gpu', 'all'),
    ('--port', '0'), ('--port', '65536'), ('--port', 'hello'),
    ('--adapter',), ('--output',), ('--unknown',),
])
def test_invalid_arguments_fail_before_model_or_output_side_effects(checkout, args):
    root, env = checkout
    install_environments(root)
    result = run(root, env, *args)
    assert result.returncode != 0
    assert calls(env) == []
    assert not (root / 'results').exists()


def test_non_linux_host_is_rejected_before_launch(checkout):
    root, env = checkout
    install_environments(root)
    env['STUB_OS'] = 'Darwin'
    result = run(root, env)
    assert result.returncode != 0
    assert 'Linux' in result.stderr
    assert calls(env) == []
    assert not (root / 'results').exists()


def test_log_write_failure_is_not_hidden_by_successful_pytest(checkout):
    root, env = checkout
    install_environments(root)
    commands = Path(env['PATH'].split(':')[0])
    tee = commands / 'tee'
    tee.write_text('#!/bin/sh\ncat\nexit 6\n')
    tee.chmod(0o755)
    output = root / 'results/log-failure'
    result = run(root, env, '--output', str(output))
    assert result.returncode != 0
    assert [record['adapter'] for record in calls(env)] == list(ADAPTERS)
    assert (output / 'summary.tsv').read_text().splitlines()[1:] == [
        'timesfm25\t0\t6', 'toto2\t0\t6', 'moirai2\t0\t6',
    ]


def test_existing_pythonpath_is_appended_after_checkout_sources(checkout):
    root, env = checkout
    install_environments(root)
    env['PYTHONPATH'] = '/extra module path'
    result = run(root, env, '--adapter', 'timesfm25')
    assert result.returncode == 0, result.stderr
    record, = calls(env)
    assert record['env']['PYTHONPATH'] == f'{root}/src:{root}:/extra module path'


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_cancelling_runner_only_waits_for_owned_pytest_cleanup(checkout, signum):
    root, env = checkout
    install_environments(root)
    ready = root.parent / 'pytest-ready.json'
    cleaned = root.parent / 'pytest-cleaned.json'
    env.update(STUB_READY=str(ready), STUB_CLEANED=str(cleaned))
    body = '''import json, os, pathlib, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def cleanup(signum, frame):
    child.terminate()
    child.wait(timeout=3)
    pathlib.Path(os.environ['STUB_CLEANED']).write_text(json.dumps({'signal': signum}))
    print('owned child cleanup complete', flush=True)
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, cleanup)
pathlib.Path(os.environ['STUB_READY']).write_text(json.dumps({'pytest': os.getpid(), 'child': child.pid}))
time.sleep(60)
'''
    launcher = root / '.venv-timesfm25/bin/python'
    launcher.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable)
                        + ' -u -c ' + shlex.quote(body) + ' "$@"\n')
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    output = root / 'results/cancelled'
    runner = subprocess.Popen(['/bin/bash', str(root / 'scripts/test-models.sh'),
                               '--output', str(output)], cwd=root, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            assert runner.poll() is None, runner.communicate()
            time.sleep(0.02)
        assert ready.exists(), 'Owned pytest did not start'
        owned = json.loads(ready.read_text())
        runner.send_signal(signum)  # Only the shell PID; no group-wide signal.
        stdout, stderr = runner.communicate(timeout=5)
        assert runner.returncode == 128 + signum, stdout + stderr
        assert json.loads(cleaned.read_text()) == {'signal': signal.SIGTERM}
        with pytest.raises(ProcessLookupError):
            os.kill(owned['child'], 0)
        assert foreign.poll() is None
        assert (output / 'timesfm25/pytest.log').read_text() == 'owned child cleanup complete\n'
        assert not (output / 'toto2').exists()
        assert not (output / 'moirai2').exists()
    finally:
        # The isolated group contains only this test's runner and fake children.
        try:
            os.killpg(runner.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        runner.wait(timeout=3)
        foreign.terminate()
        foreign.wait(timeout=3)


def test_accuracy_table_prefers_the_cpu_environment(checkout):
    root, env = checkout
    install_environments(root)
    install_stub_python(root / '.venv/bin/python', 'cpu')
    manifest = root / 'cases.json'
    manifest.write_text('{"cases":[]}')
    result = run(root, env, '--profile', 'accuracy', '--adapter', 'toto2', '--cases', str(manifest))
    assert result.returncode == 0, result.stderr
    assert [(entry['adapter'], entry['args'][:2]) for entry in calls(env)] == [
        ('toto2', ['-m', 'pytest']), ('cpu', ['-m', 'benchmarks.accuracy_table'])]
