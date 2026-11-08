"""Numerical capacity and fairness of the pinned scheduler hook, without vLLM."""

from dataclasses import replace
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vtsfm.adapters.timesfm25.admission import DEFAULT_LIMITS, forecast_cost
from vtsfm.adapters.transport import encode_metadata


class _ByteVector:
    device = 'cpu'
    dtype = 'uint8'

    def __init__(self, value):
        self.value = value
        self.shape = (len(value),)

    def tolist(self):
        return list(self.value)


def _request(identity, rows=1, context=32, horizon=1024):
    metadata = encode_metadata(identity, [f'{identity}-{row}' for row in range(rows)],
                               [context] * rows, horizon, [0.5])
    return SimpleNamespace(request_id=identity, mm_features=[SimpleNamespace(
        modality='timeseries', data={'request_metadata': SimpleNamespace(data=_ByteVector(metadata))})])


@pytest.fixture
def scheduler_module(monkeypatch):
    class BaseScheduler:
        def __init__(self, config):
            self.vllm_config = config
            self.scheduler_config = config.scheduler_config
            self.running = []
            self.requests = {}

        def add_request(self, request):
            self.requests[request.request_id] = request

        def schedule(self, throttle_prefills=False):
            return self._forecast_blocked_waiter, throttle_prefills

    names = ('vllm', 'vllm.v1', 'vllm.v1.core', 'vllm.v1.core.sched', 'vllm.v1.core.sched.scheduler')
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['vllm'].__version__ = '0.33.0'
    sys.modules[names[-1]].Scheduler = BaseScheduler
    name = 'vtsfm.continuous_scheduler'
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    # Do not leave a class inheriting the test stub for subsequent CPU tests.
    sys.modules.pop(name, None)


def _config(**overrides):
    config = SimpleNamespace(use_v2_model_runner=False,
        scheduler_config=SimpleNamespace(async_scheduling=False, enable_chunked_prefill=True,
                                         policy='fcfs', max_num_seqs=4),
        model_config=SimpleNamespace(enforce_eager=True, runner_type='pooling'),
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                        data_parallel_size=1))
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_single_request_cannot_consume_all_remaining_blocks(scheduler_module):
    scheduler = scheduler_module.ForecastBlockScheduler(_config())
    request = _request('long')
    assert scheduler._reserve_prefill_lookahead(request, 0, 8) == 1
    scheduler.running = [request]
    assert scheduler._reserve_prefill_lookahead(request, 1, 7) == 1
    assert scheduler._reserve_prefill_lookahead(request, 8, 0) == 0


def test_new_and_active_requests_share_a_step_only_within_total_capacity(scheduler_module):
    scheduler = scheduler_module.ForecastBlockScheduler(_config())
    first, second, third = _request('a', 2), _request('b', 2), _request('c', 1)
    scheduler.forecast_limits = replace(DEFAULT_LIMITS, max_execution_rows=4)
    scheduler.running = [first]
    assert scheduler._reserve_prefill_lookahead(second, 0, 8) == 1
    # The upstream scheduler appends each admitted request to running.
    scheduler.running.append(second)
    assert scheduler._reserve_prefill_lookahead(third, 0, 8) == 0
    assert scheduler._reserve_prefill_lookahead(first, 2, 6) == 1


def test_active_work_budget_counts_horizon_and_context(scheduler_module):
    scheduler = scheduler_module.ForecastBlockScheduler(_config())
    first, second = _request('a', 2, 33, 129), _request('b', 2, 33, 129)
    scheduler.forecast_limits = replace(DEFAULT_LIMITS,
        max_execution_work=forecast_cost(3, 33, 129).work_units)
    scheduler.running = [first]
    assert scheduler._reserve_prefill_lookahead(second, 0, 2) == 0


def test_large_oldest_waiter_reserves_next_capacity_instead_of_starving(scheduler_module):
    scheduler = scheduler_module.ForecastBlockScheduler(_config())
    scheduler.forecast_limits = replace(DEFAULT_LIMITS, max_execution_rows=4)
    active, oldest, small = _request('active', 2), _request('oldest', 4), _request('small', 1)
    scheduler.running = [active]
    assert scheduler._reserve_prefill_lookahead(oldest, 0, 8) == 0
    assert scheduler._reserve_prefill_lookahead(small, 0, 8) == 0
    # Active work still progresses while admissions wait.
    assert scheduler._reserve_prefill_lookahead(active, 7, 1) == 1
    scheduler.running.clear()  # Completion/cancellation releases its capacity.
    assert scheduler.schedule() == (None, False)
    assert scheduler._reserve_prefill_lookahead(oldest, 0, 8) == 1


def test_cancelled_oldest_waiter_does_not_leave_a_persistent_reservation(scheduler_module):
    scheduler = scheduler_module.ForecastBlockScheduler(_config())
    scheduler.forecast_limits = replace(DEFAULT_LIMITS, max_execution_rows=4)
    scheduler.running = [_request('active', 2)]
    assert scheduler._reserve_prefill_lookahead(_request('oldest', 4), 0, 8) == 0
    scheduler.schedule()  # Queue no longer contains the cancelled waiter.
    assert scheduler._reserve_prefill_lookahead(_request('small'), 0, 8) == 1


@pytest.mark.parametrize('field,value', [
    ('use_v2_model_runner', True), ('speculative_config', object()),
])
def test_unsupported_engine_modes_fail_explicitly(scheduler_module, field, value):
    with pytest.raises(ValueError, match='synchronous V1'):
        scheduler_module.ForecastBlockScheduler(_config(**{field: value}))


def test_launch_requires_bounded_slots_and_fcfs(scheduler_module):
    config = _config()
    config.scheduler_config.max_num_seqs = 5
    with pytest.raises(ValueError, match='scheduled requests'):
        scheduler_module.ForecastBlockScheduler(config)
    config = _config()
    config.scheduler_config.policy = 'priority'
    with pytest.raises(ValueError, match='FCFS'):
        scheduler_module.ForecastBlockScheduler(config)


def test_engine_core_no_kv_override_is_restored_before_scheduler_init(scheduler_module, monkeypatch):
    config = _config()
    config.scheduler_config.enable_chunked_prefill = False  # Exact EngineCore override.
    base = scheduler_module.ForecastBlockScheduler.__mro__[2]
    original = base.__init__
    observed = []

    def initialize(self, supplied):
        observed.append(supplied.scheduler_config.enable_chunked_prefill)
        original(self, supplied)

    monkeypatch.setattr(base, '__init__', initialize)
    scheduler = scheduler_module.ForecastBlockScheduler(config)
    assert observed == [True]
    assert scheduler.scheduler_config.enable_chunked_prefill is True


def test_vllm_owned_kv_groups_cannot_use_forecast_marker_override(scheduler_module):
    config = _config()
    config.scheduler_config.enable_chunked_prefill = False
    with pytest.raises(ValueError, match='adapter-owned KV'):
        scheduler_module.ForecastBlockScheduler(config,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]))
    assert config.scheduler_config.enable_chunked_prefill is False


def test_trace_disabled_does_not_read_metadata_or_clock(scheduler_module, monkeypatch):
    monkeypatch.delenv('VTSFM_CONTINUOUS_TRACE', raising=False)
    scheduler = scheduler_module.ForecastTracingScheduler(_config())

    def forbidden(*args, **kwargs):
        pytest.fail('Disabled tracing performed diagnostic work')

    monkeypatch.setattr(scheduler_module.time, 'monotonic_ns', forbidden)
    monkeypatch.setattr(scheduler_module, '_request_metadata', forbidden)
    request = SimpleNamespace(request_id='no-metadata-required')
    scheduler.add_request(request)
    scheduler._forecast_blocked_waiter = None  # Stub's unrelated return value.
    assert scheduler.schedule() == (None, False)


def test_whole_forecast_diagnostic_trace_preserves_native_schedule(scheduler_module, monkeypatch, tmp_path):
    path = tmp_path / 'scheduler.jsonl'
    monkeypatch.setenv('VTSFM_CONTINUOUS_TRACE', str(path))
    scheduler = scheduler_module.ForecastTracingScheduler(_config())
    request = _request('wire-a')
    request.request_id = 'engine-a-suffix'
    base = scheduler_module.ForecastTracingScheduler.__mro__[2]
    expected = SimpleNamespace(num_scheduled_tokens={request.request_id: 8})
    monkeypatch.setattr(base, 'schedule', lambda self, throttle_prefills=False: expected)
    clocks = iter((100, 200))
    monkeypatch.setattr(scheduler_module.time, 'monotonic_ns', lambda: next(clocks))

    scheduler.add_request(request)
    assert scheduler.schedule() is expected
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event['event'] for event in events] == ['enqueue', 'schedule']
    assert [event['timestamp_ns'] for event in events] == [100, 200]
    assert all(event['request_ids'] == ['wire-a'] for event in events)
    assert all(event['engine_request_ids'] == ['engine-a-suffix'] for event in events)
    assert all(not event['truncated'] for event in events)
    assert expected.num_scheduled_tokens == {'engine-a-suffix': 8}
