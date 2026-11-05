"""Record and stop only this run's ephemeral app; importing this needs no Modal."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys


def _local_file(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or len(path.parts) != 1 or path.name == '..':
        raise ValueError('App ID file must be a filename relative to the working directory')
    return path


def _app_id(value: str | None) -> str:
    # Modal otherwise accepts deployment names, which could target another app.
    if value is None or not re.fullmatch(r'ap-[A-Za-z0-9]{22}', value):
        raise ValueError('Invalid Modal app ID; refusing cleanup by name')
    return value


def record_app_id(app_id: str | None, filename: str) -> None:
    """Write before GPU submission; exclusive creation refuses stale files/symlinks."""
    app_id = _app_id(app_id)
    path = _local_file(filename)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='ascii') as output:
        output.write(app_id + '\n')


def stop_recorded_app(filename: str) -> None:
    path = _local_file(filename)
    if path.is_symlink():
        raise ValueError('App ID file must not be a symlink')
    if not path.exists():
        print('No app ID recorded; no GPU function was submitted by the entrypoint.')
        return
    app_id = _app_id(path.read_text(encoding='ascii').strip())
    command = [sys.executable, '-m', 'modal', 'app']
    try:
        subprocess.run([*command, 'stop', '--yes', app_id], check=True,
                       capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as error:
        # Normal ephemeral shutdown may win this race. Do not hide other failures.
        result = subprocess.run([*command, 'info', app_id, '--json', '--no-color'],
                                check=True, capture_output=True, text=True, timeout=60)
        info = json.loads(result.stdout)
        if info.get('app_id') != app_id or info.get('lifecycle', {}).get('state') != 'stopped':
            print(error.stderr, file=sys.stderr)
            raise
    print(f'Confirmed cleanup of this run\'s app: {app_id}')


if __name__ == '__main__':
    stop_recorded_app(os.environ['MODAL_APP_ID_FILE'])
