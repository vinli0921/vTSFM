"""Inference-only fast path for Toto's unit-scaling wrappers (issue #74).

Toto's layers call ``scale_fwd(x, s)`` / ``scale_bwd(x, s)`` about 720 times per
forecast. Both run through a ``torch.autograd.Function`` whose ``apply`` binds
arguments with ``inspect`` on every call, which dominates batch-1 host time.
Without autograd those functions only compute ``s * x`` (forward scale) or
``x`` (backward scale), so inside :func:`fast_unit_scaling` with grad disabled
the shims below return exactly that. Everywhere else, including any official
Toto model sharing the process, the original ``Function`` still runs.

One patch point per library covers every importer of ``scale_fwd`` /
``scale_bwd``:

- ``dd_unit_scaling.scale._ScaledGrad`` (Toto's compile-friendly copy), looked
  up by its ``scale_fwd`` / ``scale_bwd`` at call time;
- ``unit_scaling.scale._scale`` (Graphcore upstream, used by ``U.silu`` and
  ``uu.Dropout``), which upstream's own ``scale_fwd`` / ``scale_bwd`` call.

Each library is patched only when its ``scale`` module source matches the
digest recorded here; otherwise it stays native and a warning names it.

Numerics: a scale of exactly 1.0 returns the input tensor itself rather than
``1.0 * x``. Values are identical, but the result keeps the input's strides
where the multiply could have produced a contiguous copy, so a later kernel may
round differently (observed max abs difference 1.9e-6 at 16 rows, #74).
"""

import contextlib
import hashlib
import importlib
import inspect
import logging

import torch

logger = logging.getLogger(__name__)

# sha256 of inspect.getsource(<module>) for dd_unit_scaling at DataDog/toto
# c079fd7e32424c5d33f9e267c5d32a5630242eb5 and unit-scaling 0.3.5.
PINNED_SCALE_SOURCES = {
    'dd_unit_scaling.scale': '29ab85245f4bd10df19a6a25cafef1a7817f5f4544da2673f872b2eef40687c1',
    'unit_scaling.scale': 'f7fd58748a2a7fdaf379f066b8c96c23f86543ac89e25aaf61747f283c8b8b09',
}

_active = 0  # Nesting depth of fast_unit_scaling(); shims read it at call time.
_installed = {}  # module name -> (attribute, original object)


def _fast(input, fwd_scale, bwd_scale):
    if _active and not torch.is_grad_enabled():
        return input if fwd_scale == 1.0 else fwd_scale * input
    return None


class _InferenceScaledGrad:
    """Stands in for ``dd_unit_scaling.scale._ScaledGrad``; only ``apply`` is used."""

    original = None

    @classmethod
    def apply(cls, input, fwd_scale, bwd_scale):
        output = _fast(input, fwd_scale, bwd_scale)
        return cls.original.apply(input, fwd_scale, bwd_scale) if output is None else output


def _make_scale(original):
    def _scale(t, fwd_scale=1.0, bwd_scale=1.0):
        output = _fast(t, fwd_scale, bwd_scale)
        return original(t, fwd_scale=fwd_scale, bwd_scale=bwd_scale) if output is None else output

    _scale.__wrapped__ = original
    return _scale


def source_digest(module):
    return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()


def install():
    """Patch each pinned library once; returns the module names now patched."""
    for name, attribute in (('dd_unit_scaling.scale', '_ScaledGrad'), ('unit_scaling.scale', '_scale')):
        if name in _installed:
            continue
        module = importlib.import_module(name)
        digest = source_digest(module)
        if digest != PINNED_SCALE_SOURCES[name]:
            logger.warning('%s source changed (sha256 %s); Toto keeps its native unit-scaling '
                           'wrappers there. Revalidate the #74 fast path before updating the pin.',
                           name, digest)
            continue
        original = getattr(module, attribute)
        if attribute == '_ScaledGrad':
            _InferenceScaledGrad.original = original
            replacement = _InferenceScaledGrad
        else:
            replacement = _make_scale(original)
        setattr(module, attribute, replacement)
        _installed[name] = (attribute, original)
    return tuple(_installed)


def uninstall():
    """Restore the original objects (tests and A/B measurements)."""
    for name, (attribute, original) in _installed.items():
        setattr(importlib.import_module(name), attribute, original)
    _installed.clear()


@contextlib.contextmanager
def fast_unit_scaling(enabled=True):
    """Use the shims for calls made in this block when grad is disabled."""
    global _active
    if not enabled:
        yield
        return
    _active += 1
    try:
        yield
    finally:
        _active -= 1
