"""A GPU benchmark must reject occupants that arrive after its startup gate."""

import os
from types import SimpleNamespace

import pytest

from benchmarks import timesfm_native_kv


class FakeNVML:
    class NVMLError_NotSupported(Exception):
        pass

    def __init__(self, compute=(), graphics=(), graphics_error=None):
        self.compute = list(compute)
        self.graphics = list(graphics)
        self.graphics_error = graphics_error

    def nvmlDeviceGetComputeRunningProcesses(self, gpu):
        return [SimpleNamespace(pid=pid) for pid in self.compute]

    def nvmlDeviceGetGraphicsRunningProcesses(self, gpu):
        if self.graphics_error is not None:
            raise self.graphics_error
        return [SimpleNamespace(pid=pid) for pid in self.graphics]


def test_gpu_guard_allows_only_its_process_and_retains_observations():
    owner = os.getpid()
    nvml = FakeNVML()
    monitor = SimpleNamespace(nvml=nvml, gpu=object())
    record = {}
    timesfm_native_kv._assert_gpu_owned(monitor, record)
    nvml.compute = nvml.graphics = [owner]
    timesfm_native_kv._assert_gpu_owned(monitor, record)
    ownership = record['gpu_ownership']
    assert ownership['owner_pid'] == owner
    assert ownership['checks'] == len(ownership['observations']) == 2
    assert ownership['foreign_pids'] == []
    assert ownership['observations'][1]['compute_pids'] == [owner]
    assert ownership['observations'][1]['graphics_pids'] == [owner]


@pytest.mark.parametrize('source', ['compute', 'graphics'])
def test_gpu_guard_rejects_new_foreign_occupant_and_keeps_evidence(source):
    owner = os.getpid()
    nvml = FakeNVML(compute=[owner])
    monitor = SimpleNamespace(nvml=nvml, gpu=object())
    record = {}
    timesfm_native_kv._assert_gpu_owned(monitor, record)
    getattr(nvml, source).append(owner + 1)
    with pytest.raises(RuntimeError, match='foreign GPU processes'):
        timesfm_native_kv._assert_gpu_owned(monitor, record)
    ownership = record['gpu_ownership']
    assert ownership['checks'] == len(ownership['observations']) == 2
    assert ownership['foreign_pids'] == [owner + 1]
    assert ownership['observations'][-1]['foreign_pids'] == [owner + 1]


def test_gpu_guard_records_unsupported_graphics_query():
    nvml = FakeNVML(compute=[os.getpid()], graphics_error=FakeNVML.NVMLError_NotSupported())
    record = {}
    timesfm_native_kv._assert_gpu_owned(SimpleNamespace(nvml=nvml, gpu=object()), record)
    assert record['gpu_ownership']['observations'][0]['graphics_supported'] is False


def test_gpu_guard_does_not_hide_unexpected_nvml_errors():
    nvml = FakeNVML(graphics_error=RuntimeError('driver query failed'))
    record = {}
    with pytest.raises(RuntimeError, match='driver query failed'):
        timesfm_native_kv._assert_gpu_owned(SimpleNamespace(nvml=nvml, gpu=object()), record)
    assert record['gpu_ownership']['checks'] == 1
    assert 'driver query failed' in record['gpu_ownership']['observations'][0]['query_error']
