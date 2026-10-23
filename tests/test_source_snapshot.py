"""Benchmark archives must retain the implementation behind compatibility shims."""
from pathlib import Path


def test_snapshot_preserves_adapter_implementation_and_excludes_bytecode(tmp_path, monkeypatch):
    from benchmarks import run
    package = tmp_path / 'checkout' / 'src' / 'vtsfm'
    adapter = package / 'adapters' / 'example'
    adapter.mkdir(parents=True)
    implementation = adapter / 'model.py'
    implementation.write_text('original model implementation\n')
    (package / 'native_server.py').write_text('compatibility shim\n')
    cache = adapter / '__pycache__'
    cache.mkdir()
    (cache / 'model.pyc').write_bytes(b'cached code')
    output = tmp_path / 'run'
    output.mkdir()
    monkeypatch.setattr(run, 'REPO_ROOT', tmp_path / 'checkout')
    run.snapshot_sources(output)
    implementation.write_text('later edit\n')
    saved = output / 'harness' / 'vtsfm'
    assert (saved / 'adapters' / 'example' / 'model.py').read_text() == 'original model implementation\n'
    assert (saved / 'native_server.py').read_text() == 'compatibility shim\n'
    assert not list(saved.rglob('*.pyc'))
