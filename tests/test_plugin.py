"""The package must be discoverable without importing the GPU runtime."""
import json
import subprocess
import sys
from importlib.metadata import entry_points
from types import SimpleNamespace


def test_entry_points_are_lazy_and_discoverable():
    for group, target in [
        ('vllm.general_plugins', 'register_models'),
        ('vllm.io_processor_plugins', 'get_io_processor'),
    ]:
        found = [e for e in entry_points(group=group) if e.name == 'vtsfm']
        assert len(found) == 1
        assert found[0].load().__name__ == target
    process = subprocess.run([sys.executable, '-c',
        'import sys,json,vtsfm.plugin; print(json.dumps({"torch": "torch" in sys.modules, "vllm": "vllm" in sys.modules}))'],
        check=True, capture_output=True, text=True)
    assert json.loads(process.stdout) == {'torch': False, 'vllm': False}


def test_registration_uses_our_model_without_eager_cuda_import(monkeypatch):
    from vtsfm.plugin import get_io_processor, register_models
    registrations = []
    registry = SimpleNamespace(register_model=lambda *args: registrations.append(args))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(ModelRegistry=registry))
    register_models()
    assert registrations[0] == ('Chronos2Model', 'vtsfm.adapters.chronos2.model:VTSFMChronos2ForForecasting')
    assert len(registrations) == 4
    assert get_io_processor() == 'vtsfm.adapters.chronos2.io_processor.VTSFMChronos2IOProcessor'
