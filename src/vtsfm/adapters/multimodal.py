"""Carry prepared numerical requests through vLLM's multimodal interface."""

import uuid

import torch
from vllm.multimodal.inputs import (
    MultiModalFieldConfig, MultiModalFieldElem, MultiModalInputs,
    MultiModalKwargsItem, MultiModalKwargsItems, PlaceholderRange,
)
from vllm.multimodal.parse import DictEmbeddingItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import BaseDummyInputsBuilder, BaseMultiModalProcessor, BaseProcessingInfo

from .transport import encode_metadata

MODALITY = 'timeseries'


def fields():
    # Shared fields split distinct requests into separate vLLM groups; the raw
    # runner's dict.update then retains only the final group's values. Batched
    # fields keep every request, stacking equal shapes and retaining ragged lists.
    return {'context': MultiModalFieldConfig.batched(MODALITY),
            'request_metadata': MultiModalFieldConfig.batched(MODALITY, keep_on_cpu=True)}


class ForecastDataParser(MultiModalDataParser):
    def _parse_timeseries(self, data):
        return DictEmbeddingItems(data, modality=MODALITY,
                                  required_fields=frozenset(fields()), fields_factory=lambda _: fields())

    def _get_subparsers(self):
        return {MODALITY: self._parse_timeseries}

    def parse_mm_data(self, mm_data, **kwargs):
        return MultiModalDataItems({MODALITY: self._parse_timeseries(mm_data[MODALITY])})


class ForecastProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {MODALITY: None}

    def get_data_parser(self):
        return ForecastDataParser()


class ForecastInputBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return ''

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        # Minimal warmup input; this does not represent peak request memory.
        metadata = encode_metadata('vtsfm-dummy', [None], [32], 1, [0.5])
        return {MODALITY: {'context': [torch.ones(1, 32)],
                          'request_metadata': [torch.tensor(list(metadata), dtype=torch.uint8)]}}


class ForecastMultiModalProcessor(BaseMultiModalProcessor):
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return fields()

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(self, prompt, mm_items, hf_processor_mm_kwargs, tokenization_kwargs=None, **kwargs):
        if mm_items[MODALITY].get_count() != 1:
            raise ValueError('Expected one numerical payload per forecast request')
        data = mm_items[MODALITY].get(0)
        values = {name: MultiModalFieldElem(data=data[name], field=config.field)
                  for name, config in fields().items()}
        # One placeholder token carries a whole forecast, not a count of time steps.
        return MultiModalInputs(
            type='multimodal', prompt_token_ids=[1],
            mm_kwargs=MultiModalKwargsItems({MODALITY: [MultiModalKwargsItem(values)]}),
            mm_hashes={MODALITY: [uuid.uuid4().hex]},
            mm_placeholders={MODALITY: [PlaceholderRange(offset=0, length=0)]},
        )
