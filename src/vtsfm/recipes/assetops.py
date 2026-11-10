"""Check or apply the optional conformal hook to an AssetOpsBench checkout."""

import argparse
from importlib import resources
import os
from pathlib import Path
import subprocess


PATCH_NAME = 'assetops-conformal-factory.patch'
COMPOSITION = Path('src/servers/tsfm/engine/composition.py')


def _git(root, *arguments, patch=None):
    environment = os.environ.copy()
    # The selected checkout must own both Git metadata and working-tree writes.
    for name in ['GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR']:
        environment.pop(name, None)
    return subprocess.run(['git', '-C', str(root), *arguments], input=patch,
                          capture_output=True, text=True, env=environment)


def patch_assetops(checkout, *, apply=False):
    """Return the hook's status, changing only the chosen checkout on ``apply``.

    Compatible local changes are retained. An already applied hook is a no-op;
    unsupported or partially modified hook locations fail before any write.
    """
    root = Path(checkout).expanduser().resolve()
    repository = _git(root, 'rev-parse', '--show-toplevel')
    if repository.returncode or Path(repository.stdout.strip()).resolve() != root:
        raise ValueError('Choose the AssetOpsBench Git checkout root, not a subdirectory')
    target = root / COMPOSITION
    if not target.is_file() or target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError(f'Expected a regular checkout file: {COMPOSITION}')

    patch = resources.files('vtsfm.recipes').joinpath(PATCH_NAME).read_text(encoding='utf-8')
    forward = _git(root, 'apply', '--check', '-', patch=patch)
    if forward.returncode == 0:
        if not apply:
            return 'Patch can be applied; no files changed'
        result = _git(root, 'apply', '-', patch=patch)
        if result.returncode:
            raise ValueError(f'Could not apply the patch: {result.stderr.strip()}')
        return 'Applied the optional conformal factory patch'

    reverse = _git(root, 'apply', '--reverse', '--check', '-', patch=patch)
    if reverse.returncode == 0:
        return 'Patch already applied; no files changed'
    raise ValueError('Unsupported or overlapping AssetOpsBench source; no files changed. '
                     + forward.stderr.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path, help='AssetOpsBench Git checkout root')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Check compatibility without writes (default)')
    mode.add_argument('--apply', action='store_true', help='Apply the optional hook to this checkout')
    arguments = parser.parse_args(argv)
    try:
        print(patch_assetops(arguments.checkout, apply=arguments.apply))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
