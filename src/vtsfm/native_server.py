"""Compatibility ASGI entry point for the first adapter's native comparator."""

from vtsfm.adapters.chronos2.native_server import app, main

__all__ = ['app']


if __name__ == '__main__':
    main()
