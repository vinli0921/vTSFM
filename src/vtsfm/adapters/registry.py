"""Explicit supported adapters, with no model imports or runtime probing."""

from vtsfm.adapters.base import AdapterSpec
from vtsfm.adapters.chronos2.spec import CHRONOS2
from vtsfm.adapters.timesfm25.spec import TIMESFM25
from vtsfm.adapters.toto2.spec import TOTO2
from vtsfm.adapters.moirai2.spec import MOIRAI2

_ADAPTERS = (CHRONOS2, TIMESFM25, TOTO2, MOIRAI2)


def list_adapters() -> tuple[AdapterSpec, ...]:
    return _ADAPTERS


def get_adapter(name: str = 'chronos2') -> AdapterSpec:
    for adapter in _ADAPTERS:
        if adapter.name == name:
            return adapter
    supported = ', '.join(adapter.name for adapter in _ADAPTERS)
    raise ValueError(f'Unknown adapter {name!r}. Available adapters: {supported}')
