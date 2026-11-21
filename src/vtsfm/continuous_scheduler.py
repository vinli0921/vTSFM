"""Pinned V1 scheduling bridge: one numerical forecast block per prompt marker.

The upstream scheduler still owns queues, slots, completion and cancellation.
Its existing prefill lookahead hook limits each request to one block and gates
new requests using numerical capacity. No vLLM source method is patched.
"""

import json
from itertools import islice
import os
import time

import vllm
from vllm.v1.core.sched.scheduler import Scheduler

from vtsfm.admission import FORECAST_LIMITS_ENV, forward_resident_bytes, resolve_limits
from vtsfm.adapters.timesfm25.admission import DEFAULT_LIMITS, forecast_cost
from vtsfm.adapters.transport import decode_metadata
from vtsfm.config import FORECAST_BATCH_ROWS_ENV, MAX_FORECAST_BATCH_ROWS, VLLM_VERSION
from vtsfm.stepping import CONTINUOUS_TRACE_ENV, PREFILL_GROUP_STEPS_ENV


def _request_metadata(request):
    features = [feature for feature in request.mm_features if feature.modality == 'timeseries']
    if len(features) != 1 or features[0].data is None:
        raise ValueError('Each forecast must retain one numerical payload')
    return decode_metadata(features[0].data['request_metadata'].data)


def request_cost(request):
    """Read immutable numerical metadata on CPU; never stage a context here."""
    metadata = _request_metadata(request)
    return forecast_cost(len(metadata['item_ids']), max(metadata['context_lengths']), metadata['horizon'])


class ForecastTraceMixin:
    """Optional CPU-only enqueue/schedule spans for diagnostic benchmark runs.

    Disabled tracing does not read request metadata or sample a clock. Each
    enabled event uses one append write, with bounded IDs/request count, and
    retains no request dictionary beyond the upstream scheduler's own state.
    """

    def __init__(self, *args, **kwargs):
        if vllm.__version__ != VLLM_VERSION:
            raise ValueError(f'Forecast scheduler hooks require pinned vLLM {VLLM_VERSION}')
        self._forecast_trace_path = os.environ.get(CONTINUOUS_TRACE_ENV)
        super().__init__(*args, **kwargs)

    def _forecast_trace(self, event, timestamp_ns, requests, **details):
        requests = tuple(requests)
        selected = requests[:64]
        wire_ids = [_request_metadata(request)['request_id'] for request in selected]
        engine_ids = [request.request_id for request in selected]
        # A project's valid batch has at most four slots. Also bound diagnostic
        # output if this tracing-only class is used outside that configuration.
        truncated = len(selected) != len(requests) or any(
            len(identity) > 256 for identity in (*wire_ids, *engine_ids))
        payload = (json.dumps({
            'event': event, 'timestamp_ns': timestamp_ns,
            'request_ids': [identity[:256] for identity in wire_ids],
            'engine_request_ids': [identity[:256] for identity in engine_ids],
            'request_count': len(requests), 'truncated': truncated,
            **details,
        }, separators=(',', ':'), ensure_ascii=True) + '\n').encode('utf-8')
        descriptor = os.open(self._forecast_trace_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if os.write(descriptor, payload) != len(payload):
                raise OSError('Incomplete forecast scheduler trace write')
        finally:
            os.close(descriptor)

    def add_request(self, request):
        if not self._forecast_trace_path:
            return super().add_request(request)
        started = time.monotonic_ns()
        result = super().add_request(request)
        self._forecast_trace('enqueue', started, (request,))
        return result

    def schedule(self, throttle_prefills=False):
        if not self._forecast_trace_path:
            return super().schedule(throttle_prefills)
        started = time.monotonic_ns()
        result = super().schedule(throttle_prefills)
        self._forecast_trace('schedule', started,
            (self.requests[request_id] for request_id in result.num_scheduled_tokens))
        return result


class ForecastTracingScheduler(ForecastTraceMixin, Scheduler):
    """Unmodified whole-forecast scheduler with opt-in diagnostic timestamps."""


class ForecastBlockScheduler(ForecastTraceMixin, Scheduler):
    """FCFS block scheduling with conservative capacity reserved for active state.

    A virtual prompt marker represents one model-specific forecast block, not
    a time-series value or text token. Returning zero for a waiting request
    uses upstream deferral; returning one schedules exactly its next block.
    """

    def __init__(self, *args, **kwargs):
        if vllm.__version__ != VLLM_VERSION:
            raise ValueError(f'Forecast blocks require pinned vLLM {VLLM_VERSION}')
        config = kwargs.get('vllm_config', args[0] if args else None)
        if config is None:
            raise ValueError('Forecast block scheduling requires a vLLM configuration')
        scheduler_config = config.scheduler_config
        parallel = config.parallel_config
        if (config.use_v2_model_runner or scheduler_config.async_scheduling
                or not config.model_config.enforce_eager
                or config.model_config.runner_type != 'pooling'
                or config.speculative_config is not None
                or parallel.tensor_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1):
            raise ValueError('Forecast blocks require synchronous V1 pooling, chunked prefill, '
                             'eager execution and one GPU without speculation')
        if scheduler_config.policy != 'fcfs':
            raise ValueError('Forecast block admission currently requires FCFS scheduling')
        kv_cache_config = kwargs.get('kv_cache_config', args[1] if len(args) > 1 else None)
        if kv_cache_config is not None and kv_cache_config.kv_cache_groups:
            raise ValueError('Forecast blocks require adapter-owned KV state, without vLLM KV groups')
        # vLLM 0.33 EngineCore disables chunked prefill for every no-KV model
        # immediately before constructing the scheduler. Here prompt markers
        # are resumable forecast blocks, and the adapter owns attention/state;
        # restore the flag before Scheduler caches/uses this configuration.
        # GPU workers were initialized with the requested flag before that
        # EngineCore mutation; restoring it also keeps both sides consistent.
        scheduler_config.enable_chunked_prefill = True
        super().__init__(*args, **kwargs)
        self.forecast_limits = resolve_limits(DEFAULT_LIMITS, os.environ.get(FORECAST_LIMITS_ENV))
        limits = self.forecast_limits
        limits.validate_launch(self.scheduler_config.max_num_seqs, forecast_cost(
            limits.max_request_rows, limits.max_context, limits.max_horizon))
        self._forecast_blocked_waiter = None
        try:
            self.prefill_group_steps = int(os.environ.get(PREFILL_GROUP_STEPS_ENV, '0'))
        except ValueError as error:
            raise ValueError(f'{PREFILL_GROUP_STEPS_ENV} must be 0, 1 or 2') from error
        if self.prefill_group_steps not in (0, 1, 2):
            raise ValueError(f'{PREFILL_GROUP_STEPS_ENV} must be 0, 1 or 2')
        self._forecast_group_waiter = None
        self._forecast_group_deadline = None
        self._forecast_group_released = False
        self._forecast_group_step = 0
        self._forecast_group_decodes = {}
        self.prefill_group_rows = limits.max_execution_rows
        if self.prefill_group_steps:
            fusion_rows = int(os.environ.get(FORECAST_BATCH_ROWS_ENV, '0'))
            if fusion_rows and not 2 <= fusion_rows <= MAX_FORECAST_BATCH_ROWS:
                raise ValueError('Forecast fusion rows must be between 2 and 16')
            self.prefill_group_rows = min(fusion_rows or limits.max_execution_rows,
                                          limits.max_execution_rows)

    def schedule(self, throttle_prefills=False):
        # Upstream FCFS preserves skipped waiting order. Once the oldest waiter
        # cannot fit, hold later admissions for this iteration, allowing the
        # finite active forecasts to drain instead of starving a large waiter.
        self._forecast_blocked_waiter = None
        if self.prefill_group_steps:
            self._forecast_group_step += 1
            self._forecast_group_decodes = {}
            oldest = next(iter(self.waiting), None)
            identity = oldest.request_id if oldest is not None else None
            if identity != self._forecast_group_waiter:
                if (self._forecast_group_deadline is not None and not self._forecast_group_released
                        and self._forecast_trace_path):
                    self._forecast_trace('prefill_group_release', time.monotonic_ns(), (),
                        waiter_engine_id=self._forecast_group_waiter,
                        step=self._forecast_group_step, deadline_step=self._forecast_group_deadline,
                        reason='cancelled_or_replaced')
                self._forecast_group_waiter = identity
                self._forecast_group_deadline = None
                self._forecast_group_released = False
        return super().schedule(throttle_prefills)

    def _fits_forecast_costs(self, costs):
        limits = self.forecast_limits
        return (sum(cost.rows for cost in costs) <= limits.max_execution_rows
                and sum(cost.work_units for cost in costs) <= limits.max_execution_work
                and forward_resident_bytes(costs) <= limits.max_forward_bytes)

    def _prefill_cohort_count(self, active_costs, queued_costs):
        """Count a FIFO prefix under the unchanged slot/numerical ceilings."""
        costs, count, rows, work_per_row = list(active_costs), 0, 0, 0
        slots = self.scheduler_config.max_num_seqs - len(costs)
        for candidate in queued_costs[:slots]:
            if not self._fits_forecast_costs((*costs, candidate)):
                break
            combined_rows = rows + candidate.rows
            combined_work = max(work_per_row, candidate.work_units // candidate.rows)
            if count and (combined_rows > self.prefill_group_rows
                          or combined_rows * combined_work > self.forecast_limits.max_execution_work):
                break
            costs.append(candidate)
            count += 1
            rows, work_per_row = combined_rows, combined_work
            # The worker may execute a request above the optional fusion cap
            # alone; such a request cannot justify waiting for a larger call.
            if rows > self.prefill_group_rows:
                break
        return count

    def _group_prefill(self, request, candidate, active_costs):
        """Briefly wait for a larger compatible cohort while decode progresses.

        The oldest waiter's deadline is absolute across schedule iterations; a
        changing cohort cannot renew it. No request/queue is removed or reordered.
        Only already-runnable decode work can justify a hold. One-block requests,
        lone arrivals and queues without a larger feasible cohort bypass it.
        """
        if request.request_id != self._forecast_group_waiter:
            return False
        step, deadline = self._forecast_group_step, self._forecast_group_deadline
        remaining = self._forecast_group_decodes
        finishing = {identity for identity, blocks in remaining.items()
                     if 0 < blocks <= self.prefill_group_steps}
        queued = []
        if deadline is None or step < deadline:
            if candidate.horizon > 128 and finishing:
                # Bound metadata work by the finite scheduled slot count. An
                # incompatible entry ends the prefix; FCFS is never bypassed.
                for waiter in islice(self.waiting, self.scheduler_config.max_num_seqs):
                    cost = request_cost(waiter)
                    if cost.horizon <= 128:
                        queued.clear()  # Do not add queueing ahead of a short job.
                        break
                    if (cost.context + 31) // 32 != (candidate.context + 31) // 32:
                        break
                    queued.append(cost)
        current = self._prefill_cohort_count(active_costs, queued)
        future_costs = [cost for active, cost in zip(self.running, active_costs, strict=True)
                        if active.request_id not in finishing]
        future = self._prefill_cohort_count(future_costs, queued)
        if future > max(current, 1):
            if deadline is None:
                deadline = self._forecast_group_deadline = step + self.prefill_group_steps
            if self._forecast_trace_path:
                self._forecast_trace('prefill_group_hold', time.monotonic_ns(), (request,),
                    step=step, deadline_step=deadline, budget_steps=self.prefill_group_steps,
                    remaining_steps={identity[:256]: blocks for identity, blocks in remaining.items()},
                    finishing_request_ids=[identity[:256] for identity in sorted(finishing)],
                    current_cohort=current, future_cohort=future, reason='near_completion')
            return True
        if deadline is not None:
            # Admission/early release also consumes this oldest waiter's budget.
            self._forecast_group_deadline = min(step, deadline)
            if self._forecast_trace_path and not self._forecast_group_released:
                self._forecast_trace('prefill_group_release', time.monotonic_ns(), (request,),
                    step=step, deadline_step=deadline,
                    reason='deadline' if step >= deadline else 'no_group_benefit')
            self._forecast_group_released = True
        return False

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        if num_new_tokens <= 0:
            return 0
        if any(active.request_id == request.request_id for active in self.running):
            if self.prefill_group_steps and num_computed_tokens > 0:
                self._forecast_group_decodes[request.request_id] = request.num_tokens - num_computed_tokens
            return 1
        if self._forecast_blocked_waiter is not None:
            return 0

        limits = self.forecast_limits
        candidate = request_cost(request)
        # The frontend already checks this. Retain an assertion for direct
        # internal engine callers, rather than executing an invalid request.
        limits.validate_request(candidate)
        active_costs = [request_cost(active) for active in self.running]
        costs = (*active_costs, candidate)
        if not self._fits_forecast_costs(costs):
            self._forecast_blocked_waiter = request.request_id
            return 0
        if self.prefill_group_steps and self._group_prefill(request, candidate, active_costs):
            self._forecast_blocked_waiter = request.request_id
            return 0
        return 1
