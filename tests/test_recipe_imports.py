"""The optional recipe boundary must remain importable without model packages."""

import os
import subprocess
import sys


def test_optional_recipe_package_does_not_import_model_or_recipe_dependencies():
    code = '''
import sys
import vtsfm.recipes
import vtsfm.recipes.assetops
assert not {'torch', 'vllm', 'numpy', 'pandas', 'sktime', 'servers'}.intersection(sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], env=os.environ.copy(),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_assetops_help_does_not_load_recipe_or_model_dependencies():
    result = subprocess.run([sys.executable, '-m', 'vtsfm.recipes.assetops', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--apply' in result.stdout and '--check' in result.stdout
