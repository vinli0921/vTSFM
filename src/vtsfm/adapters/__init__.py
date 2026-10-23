"""CPU-safe adapter descriptions; model runtimes are imported only when serving."""

from vtsfm.adapters.base import AdapterSpec
from vtsfm.adapters.registry import get_adapter, list_adapters

__all__ = ['AdapterSpec', 'get_adapter', 'list_adapters']
