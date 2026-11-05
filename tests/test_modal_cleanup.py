"""Cleanup may target only the fresh, validated ID of this run's ephemeral app."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


APP_ID = 'ap-1234567890abcdefghijkl'


def test_app_id_handoff_is_private_and_cannot_overwrite_an_older_run(tmp_path, monkeypatch):
    from ci.modal_cleanup import record_app_id

    monkeypatch.chdir(tmp_path)
    record_app_id(APP_ID, '.app-id')
    path = Path('.app-id')
    assert path.read_text() == APP_ID + '\n'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        record_app_id('ap-abcdefghijkl1234567890', '.app-id')
    assert path.read_text() == APP_ID + '\n'


@pytest.mark.parametrize('filename', ['', '.', '..', '../foreign', 'nested/app', '/tmp/app'])
def test_app_id_handoff_rejects_paths_outside_a_fresh_local_file(filename):
    from ci.modal_cleanup import record_app_id

    with pytest.raises(ValueError):
        record_app_id(APP_ID, filename)


@pytest.mark.parametrize('app_id', [None, '', 'vtsfm-gpu-tests', 'ap-short', APP_ID + '\nother'])
def test_handoff_rejects_names_and_invalid_ids_before_writing(app_id, tmp_path, monkeypatch):
    from ci.modal_cleanup import record_app_id

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        record_app_id(app_id, '.app-id')
    assert not Path('.app-id').exists()


def test_cleanup_without_an_app_needs_no_modal_client(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'ci' / 'modal_cleanup.py'
    result = subprocess.run([sys.executable, '-I', str(script)], cwd=tmp_path,
                            env={**os.environ, 'MODAL_APP_ID_FILE': '.app-id'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'No app ID recorded' in result.stdout


@pytest.mark.parametrize('contents', ['vtsfm-gpu-tests', 'ap-short', APP_ID + '\nother'])
def test_cleanup_refuses_invalid_ids_before_contacting_modal(contents, tmp_path, monkeypatch):
    from ci.modal_cleanup import stop_recorded_app

    monkeypatch.chdir(tmp_path)
    Path('.app-id').write_text(contents)
    with pytest.raises(ValueError):
        stop_recorded_app('.app-id')


def test_cleanup_rejects_symlinks_without_touching_their_target(tmp_path, monkeypatch):
    from ci.modal_cleanup import stop_recorded_app

    monkeypatch.chdir(tmp_path)
    Path('foreign').write_text(APP_ID)
    Path('.app-id').symlink_to('foreign')
    with pytest.raises(ValueError, match='symlink'):
        stop_recorded_app('.app-id')
    assert Path('foreign').read_text() == APP_ID


def test_cleanup_stops_the_recorded_id_without_name_lookup(tmp_path, monkeypatch):
    from ci.modal_cleanup import record_app_id, stop_recorded_app

    monkeypatch.chdir(tmp_path)
    record_app_id(APP_ID, '.app-id')

    def modal_command(command, **kwargs):
        assert command == [sys.executable, '-m', 'modal', 'app', 'stop', '--yes', APP_ID]
        return subprocess.CompletedProcess(command, 0, stdout='')

    monkeypatch.setattr(subprocess, 'run', modal_command)
    stop_recorded_app('.app-id')


@pytest.mark.parametrize('reported_id,state,allowed', [
    (APP_ID, 'stopped', True),
    (APP_ID, 'ephemeral', False),
    ('ap-abcdefghijkl1234567890', 'stopped', False),
])
def test_cleanup_accepts_failed_stop_only_when_the_exact_app_is_already_stopped(
        reported_id, state, allowed, tmp_path, monkeypatch):
    from ci.modal_cleanup import record_app_id, stop_recorded_app

    monkeypatch.chdir(tmp_path)
    record_app_id(APP_ID, '.app-id')

    # The only substituted boundary is the external Modal CLI; no GPU is launched.
    def modal_command(command, **kwargs):
        assert command[1:4] == ['-m', 'modal', 'app']
        if command[4] == 'stop':
            assert command[5:] == ['--yes', APP_ID]
            raise subprocess.CalledProcessError(1, command, stderr='stop failed')
        assert command[4:] == ['info', APP_ID, '--json', '--no-color']
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
            'app_id': reported_id, 'lifecycle': {'state': state}}))

    monkeypatch.setattr(subprocess, 'run', modal_command)
    if allowed:
        stop_recorded_app('.app-id')
    else:
        with pytest.raises(subprocess.CalledProcessError):
            stop_recorded_app('.app-id')
