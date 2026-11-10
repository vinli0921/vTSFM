"""Checkout-only integration checks, also runnable against an installed wheel."""

import hashlib
from importlib import resources
import subprocess
import sys

import pytest

from vtsfm.recipes.assetops import COMPOSITION, PATCH_NAME, patch_assetops


def git(root, *arguments):
    return subprocess.run(['git', '-C', str(root), *arguments],
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'AssetOpsBench'
    root.mkdir()
    git(root, 'init', '-q')
    target = root / COMPOSITION
    target.parent.mkdir(parents=True)
    # Only the pinned hook contexts, not a copied recipe implementation.
    lines = ['# fixture context\n'] * 730
    sections = {
        260: ['    scenario_id: Optional[str] = None,', ') -> dict:'],
        272: ['            scenario_id=scenario_id,', '        )'],
        613: ['def _conformal_ad(',
              '    store, y, recipe: dict, *, asset_id, parent_run_id, scenario_id',
              ') -> dict:'],
        645: ['    ci = ConformalIntervals(R.resolve(merged))', '    ci.fit(y_train, fh=fh)'],
        697: ['    scenario_id: Optional[str] = None,', ') -> dict:'],
        709: ['            scenario_id=scenario_id,', '        )'],
    }
    for start, values in sections.items():
        lines[start - 1:start - 1 + len(values)] = [value + '\n' for value in values]
    target.write_text(''.join(lines))
    unrelated = root / 'unrelated.txt'
    unrelated.write_text('original\n')
    git(root, 'add', '.')
    unrelated.write_text('staged teammate change\n')
    git(root, 'add', 'unrelated.txt')
    unrelated.write_text('unstaged teammate change\n')
    return root


def test_patch_resource_remains_byte_identical():
    patch = resources.files('vtsfm.recipes').joinpath(PATCH_NAME).read_bytes()
    assert hashlib.sha256(patch).hexdigest() == 'd8b660e25dbcae436087fd1d4a0e919924fad94153d3a5e93d3d3b325863fc76'


def test_check_apply_and_idempotency_preserve_unrelated_work_and_index(checkout):
    target = checkout / COMPOSITION
    target.write_text(target.read_text() + '# unrelated source edit\n')
    before = target.read_bytes()
    index = git(checkout, 'diff', '--cached')
    unrelated = (checkout / 'unrelated.txt').read_bytes()
    assert 'can be applied' in patch_assetops(checkout)
    assert target.read_bytes() == before
    assert 'Applied' in patch_assetops(checkout, apply=True)
    after = target.read_bytes()
    assert after != before
    assert after.endswith(b'# unrelated source edit\n')
    for apply in [False, True]:
        assert 'already applied' in patch_assetops(checkout, apply=apply)
        assert target.read_bytes() == after
    assert (checkout / 'unrelated.txt').read_bytes() == unrelated
    assert git(checkout, 'diff', '--cached') == index
    assert set(git(checkout, 'diff', '--name-only').splitlines()) == {str(COMPOSITION), 'unrelated.txt'}


@pytest.mark.parametrize('apply', [False, True])
def test_subdirectory_is_rejected_instead_of_silently_skipping(checkout, apply):
    before = (checkout / COMPOSITION).read_bytes()
    with pytest.raises(ValueError, match='checkout root'):
        patch_assetops(checkout / 'src', apply=apply)
    assert (checkout / COMPOSITION).read_bytes() == before


def test_non_checkout_and_missing_source_fail(tmp_path, checkout):
    with pytest.raises(ValueError, match='checkout root'):
        patch_assetops(tmp_path)
    (checkout / COMPOSITION).unlink()
    with pytest.raises(ValueError, match='regular checkout file'):
        patch_assetops(checkout, apply=True)


@pytest.mark.parametrize('partial', [False, True])
def test_overlap_and_partial_patch_fail_without_writes(checkout, partial):
    target = checkout / COMPOSITION
    if partial:
        patch_assetops(checkout, apply=True)
        value = target.read_text().replace('    ci = factory(R.resolve(merged))',
                                          '    ci = ExistingWrapper(R.resolve(merged))')
    else:
        value = target.read_text().replace('    ci = ConformalIntervals(R.resolve(merged))',
                                          '    ci = ExistingWrapper(R.resolve(merged))')
    target.write_text(value)
    before = target.read_bytes()
    with pytest.raises(ValueError, match='Unsupported or overlapping'):
        patch_assetops(checkout, apply=True)
    assert target.read_bytes() == before


def test_symlink_cannot_redirect_patch_outside_checkout(checkout, tmp_path):
    target = checkout / COMPOSITION
    outside = tmp_path / 'installed-composition.py'
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    before = outside.read_bytes()
    with pytest.raises(ValueError, match='regular checkout file'):
        patch_assetops(checkout, apply=True)
    assert outside.read_bytes() == before


def test_git_environment_cannot_redirect_selected_checkout(checkout, tmp_path, monkeypatch):
    other = tmp_path / 'another-checkout'
    other.mkdir()
    git(other, 'init', '-q')
    monkeypatch.setenv('GIT_DIR', str(other / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(other))
    assert 'Applied' in patch_assetops(checkout, apply=True)
    assert not (other / COMPOSITION).exists()


def test_cli_defaults_to_check_and_requires_explicit_apply(checkout):
    target = checkout / COMPOSITION
    before = target.read_bytes()
    command = [sys.executable, '-m', 'vtsfm.recipes.assetops', str(checkout)]
    checked = subprocess.run(command, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert target.read_bytes() == before
    applied = subprocess.run(command + ['--apply'], capture_output=True, text=True)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    assert target.read_bytes() != before
    repeated = subprocess.run(command + ['--apply'], capture_output=True, text=True)
    assert repeated.returncode == 0 and 'already applied' in repeated.stdout
    invalid = subprocess.run(command + ['--apply', '--check'], capture_output=True, text=True)
    assert invalid.returncode == 2
