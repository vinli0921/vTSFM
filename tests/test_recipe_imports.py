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
