"""Adapter-owned block state behind vLLM V1's chunked pooling lifecycle."""

from dataclasses import dataclass
import json
import os
import time

import torch
from vllm.model_executor.layers.pooler import Pooler
from vllm.utils import length_from_prompt_token_ids_or_embeds

from vtsfm.admission import forward_resident_bytes
from vtsfm.stepping import BlockRequest, CONTINUOUS_TRACE_ENV, plan_block_calls
from vtsfm.adapters.result_transport import encode_result_metadata, pack_result
from vtsfm.adapters.transport import decode_requests


@dataclass(frozen=True)
class BlockSlot:
    request_id: str
    wire_id: str
    progress: int
    blocks: int


@dataclass(frozen=True)
class _ResponseMetadata:
    wire_id: str
    item_ids: tuple
    horizon: int
    levels: tuple
    cost: object


class BlockExecutor:
    """Keep only active request state; completed/cancelled tensors are released.

    vLLM schedules one virtual prompt marker per forecast block. New contexts
    arrive through raw multimodal transport once. Later iterations read actual
    cached slot identities/progress from the pinned worker callback.
    """

    def __init__(self, stepper, limits, cost_for_shape, quantiles, fusion_rows):
        self.stepper, self.limits = stepper, limits
        self.cost_for_shape, self.quantiles = cost_for_shape, quantiles
        self.fusion_rows = fusion_rows or limits.max_execution_rows
        self.states, self.metadata = {}, {}
        self.slot_source = None
        self.outputs = None
        self.profiling = True
        self.trace_path = os.environ.get(CONTINUOUS_TRACE_ENV)

    def set_slots(self, callback):
        self.slot_source = callback

    def trace(self, event, **fields):
        if self.trace_path:
            with open(self.trace_path, 'a') as stream:
                stamp = fields.pop('timestamp_ns', time.monotonic_ns())
                stream.write(json.dumps(dict(event=event, timestamp_ns=stamp, **fields)) + '\n')

    def forget(self, request_ids, reason='cancel'):
        for identity in request_ids:
            state = self.states.pop(identity, None)
            metadata = self.metadata.pop(identity, None)
            if state is not None:
                self.stepper.release(state)
                self.trace('release', request_ids=[metadata.wire_id],
                           engine_request_ids=[identity], reason=reason)

    def snapshot(self):
        return {'state_count': len(self.states),
                'active_rows': sum(state.rows for state in self.states.values()),
                'active_work': sum(record.cost.work_units for record in self.metadata.values()),
                'cache_bytes': sum(state.cache_bytes for state in self.states.values()),
                'tensor_bytes': sum(state.tensor_bytes for state in self.states.values()),
                'pending_outputs': self.outputs is not None}

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        length = length_from_prompt_token_ids_or_embeds(input_ids, inputs_embeds)
        self.outputs = None
        slots = self.slot_source() if self.slot_source is not None else None
        self.profiling = not slots
        if self.profiling:
            return torch.zeros(length, 0, device=positions.device)
        if length != len(slots) or len({slot.request_id for slot in slots}) != len(slots):
            raise ValueError('Block execution requires one scheduled marker per unique slot')

        transported = {}
        if kwargs.get('context') is not None:
            transported = {request.request_id: request for request in decode_requests(
                kwargs['context'], kwargs['request_metadata'])}
        new = []
        for slot in slots:
            if slot.request_id not in self.states:
                if slot.progress != 0 or slot.wire_id not in transported:
                    raise ValueError('A new block request must start with its numerical context')
                request = transported.pop(slot.wire_id)
                cost = self.cost_for_shape(len(request.item_ids), request.context.shape[1], request.horizon)
                self.limits.validate_request(cost)
                if slot.blocks != (request.horizon + 127) // 128:
                    raise ValueError('Block marker count does not match forecast horizon')
                new.append((slot, request, cost))
            else:
                state = self.states[slot.request_id]
                if state.blocks_done != slot.progress or self.metadata[slot.request_id].wire_id != slot.wire_id:
                    raise ValueError('Worker block progress or identity does not match adapter state')
        if transported:
            raise ValueError('Numerical payloads do not match new forecast block slots')
        costs = [record.cost for record in self.metadata.values()] + [cost for _, _, cost in new]
        if (sum(cost.rows for cost in costs) > self.limits.max_execution_rows
                or sum(cost.work_units for cost in costs) > self.limits.max_execution_work
                or forward_resident_bytes(costs) > self.limits.max_forward_bytes):
            raise ValueError('Block scheduler exceeded the active numerical capacity')

        for slot, request, cost in new:
            self.states[slot.request_id] = self.stepper.initialize(request.context, request.horizon)
            self.metadata[slot.request_id] = _ResponseMetadata(request.request_id, request.item_ids,
                                                              request.horizon, request.quantile_levels, cost)
        calls = plan_block_calls([
            BlockRequest(slot.request_id, self.states[slot.request_id].rows,
                         self.states[slot.request_id].bucket,
                         self.stepper.model_call_work([self.states[slot.request_id]])
                         // self.states[slot.request_id].rows)
            for slot in slots], max_rows=self.limits.max_execution_rows,
            max_work=self.limits.max_execution_work, fusion_rows=self.fusion_rows)
        completed = {}
        for identities in calls:
            states = [self.states[identity] for identity in identities]
            stage = 'prefill' if states[0].blocks_done == 0 else 'decode'
            before = [state.blocks_done for state in states] if self.trace_path else None
            cache_positions = [state.cache_position for state in states] if self.trace_path else None
            cache_position = cache_positions[0] if cache_positions and len(set(cache_positions)) == 1 else None
            records = [self.metadata[identity] for identity in identities]
            start = time.monotonic_ns() if self.trace_path else 0
            if stage == 'prefill':
                self.stepper.prefill(states)
            else:
                self.stepper.advance(states)
            finished = []
            for identity, state, record in zip(identities, states, records, strict=True):
                if self.stepper.complete(state):
                    output = self.stepper.finalize(state)
                    if (output.shape != (len(record.item_ids), len(self.quantiles), record.horizon)
                            or output.dtype != torch.float32 or output.device != positions.device):
                        raise ValueError('Completed block forecast has an invalid shape, dtype or device')
                    completed[identity] = pack_result(output, encode_result_metadata(
                        record.item_ids, record.horizon, record.levels))
                    finished.append(record.wire_id)
            if self.trace_path:
                self.trace('step', request_ids=[record.wire_id for record in records],
                           engine_request_ids=list(identities), stage=stage,
                           block_indices=before, rows=sum(state.rows for state in states),
                           cache_position=cache_position, cache_positions=cache_positions,
                           finished_ids=finished,
                           timestamp_ns=start, start_ns=start, end_ns=time.monotonic_ns(),
                           active_rows=sum(state.rows for state in self.states.values()),
                           cache_bytes=sum(state.cache_bytes for state in self.states.values()),
                           tensor_bytes=sum(state.tensor_bytes for state in self.states.values()),
                           execution_work=self.stepper.model_call_work(states))
            self.forget([identity for identity in identities if identity in completed], reason='complete')
        self.outputs = [completed.get(slot.request_id) for slot in slots]
        return torch.empty((len(slots), 0), device=positions.device)


class BlockForecastPooler(Pooler):
    def __init__(self, executor):
        super().__init__()
        self.executor = executor

    def get_supported_tasks(self):
        return {'plugin'}

    def forward(self, hidden_states, pooling_metadata):
        outputs, self.executor.outputs = self.executor.outputs, None
        if self.executor.profiling:
            return [hidden_states.new_zeros((1, 9, 1)) for _ in pooling_metadata.pooling_params]
        cursor = pooling_metadata.get_pooling_cursor()
        if outputs is None or len(outputs) != len(pooling_metadata.pooling_params):
            raise ValueError('Block pooler has no matching unconsumed results')
        if cursor.num_scheduled_tokens_cpu.tolist() != [1] * len(outputs):
            raise ValueError('Each pooling iteration must execute one forecast block')
        if [output is not None for output in outputs] != cursor.get_finished_mask():
            raise ValueError('Forecast completion does not match vLLM block progress')
        return outputs
