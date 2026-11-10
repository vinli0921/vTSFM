"""Opt-in univariate attention for the pinned Moirai 2.0 encoder (issue #24).

Pinned uni2ts calls ``F.scaled_dot_product_attention`` with 5-D tensors
``[*batch, group, heads_per_group, L, dim]`` and a dense float bias
``[*batch, heads, 1, L, L]``, rebuilt in every layer from ``BinaryAttentionBias``.
PyTorch's fused SDPA kernels accept only 4-D inputs, so that call takes the math
path and materializes several ``O(batch * heads * L^2)`` tensors per layer.

For univariate rows every token has the same variate ID. The binary variate bias
is then one constant per head for every (query, key) pair, and softmax is
invariant to a constant shift, so the bias cannot change the output. This module
replaces each layer's attention on one model instance with a 4-D call that keeps
only the boolean packed-causal mask, or uses ``is_causal`` when that mask is a
plain lower triangle (no padding).

Only use this for univariate inputs (``target_dim == 1``). vTSFM's Moirai runtime
rejects grouped/multivariate rows, so it meets that condition.
"""

import hashlib
import inspect
import types
import weakref

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# sha256 of inspect.getsource(GroupedQueryAttention.forward) at uni2ts
# cfd46d4510ed8896f263116f32928eede05b0a75. Revalidate before changing the pin.
PINNED_FORWARD_SHA256 = '0cbd7ac8bcde724a399390d4e4cf48f877dc341e5eede35561e2412c2a33d5c3'


def upstream_forward_digest():
    from uni2ts.module.attention import GroupedQueryAttention

    return hashlib.sha256(inspect.getsource(GroupedQueryAttention.forward).encode()).hexdigest()


class _MaskState:
    """Remember whether the current packed mask is a plain causal triangle.

    The encoder passes the same mask object to every layer, so the check runs
    once per module call. A weak reference avoids keeping the mask alive and
    cannot match a different tensor that reuses freed memory.
    """

    def __init__(self):
        self.ref = None
        self.causal = False

    def is_causal(self, mask):
        if self.ref is not None and self.ref() is mask:
            return self.causal
        length = mask.shape[-1]
        tril = torch.ones(length, length, dtype=torch.bool, device=mask.device).tril()
        self.causal = mask.shape[-2] == length and bool(torch.equal(mask, tril.expand_as(mask)))
        self.ref = weakref.ref(mask)
        return self.causal


def _univariate_forward(self, query, key, value, attn_mask=None, query_var_id=None, kv_var_id=None,
                        query_time_id=None, kv_time_id=None, *, state, verify):
    groups, per_group = self.num_groups, self.heads_per_group
    query = self.q_norm(rearrange(self.q_proj(query), '... l (g h d) -> ... g h l d', g=groups, h=per_group))
    key = self.k_norm(repeat(self.k_proj(key), '... l (g d) -> ... g h l d', g=groups, h=per_group))
    value = repeat(self.v_proj(value), '... l (g d) -> ... g h l d', g=groups, h=per_group)
    query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
    query_time_id, kv_time_id = self._get_time_id(query, key, query_time_id, kv_time_id)
    if verify and query_var_id is not None:
        first = query_var_id[..., :1]
        if not (bool((query_var_id == first).all()) and bool((kv_var_id == first).all())):
            raise ValueError('Univariate Moirai attention received more than one variate ID per row')
    query, key = self._qk_proj(query, key, query_var_id=query_var_id, kv_var_id=kv_var_id,
                               query_time_id=query_time_id, kv_time_id=kv_time_id)

    lead, (_, _, length, dim) = query.shape[:-4], query.shape[-4:]
    heads = groups * per_group
    kv_length = key.shape[-2]
    mask, causal = None, False
    if attn_mask is not None:
        if state.is_causal(attn_mask):
            causal = True
        else:
            mask = attn_mask.reshape(-1, 1, length, kv_length)  # bool, broadcast over heads
    out = F.scaled_dot_product_attention(
        query.reshape(-1, heads, length, dim), key.reshape(-1, heads, kv_length, dim),
        value.reshape(-1, heads, kv_length, dim), attn_mask=mask, is_causal=causal,
        dropout_p=self.attn_dropout_p if self.training else 0.0, scale=self.softmax_scale)
    out = out.reshape(*lead, groups, per_group, length, dim)
    return self.out_proj(rearrange(out, '... g h l d -> ... l (g h d)'))


def install_univariate_attention(module, *, verify=False, check_source=True):
    """Patch one ``Moirai2Module`` instance in place; return a function that undoes it.

    ``verify=True`` checks variate IDs on every call (one host sync per layer).
    """
    if check_source:
        digest = upstream_forward_digest()
        if digest != PINNED_FORWARD_SHA256:
            raise RuntimeError(f'uni2ts GroupedQueryAttention.forward changed (sha256 {digest}); '
                               'revalidate the univariate attention patch before using it')
    state = _MaskState()
    patched = []
    for layer in module.encoder.layers:
        attention = layer.self_attn
        if 'forward' in vars(attention):
            raise RuntimeError('Moirai attention already has an instance-level forward')

        def forward(self, *args, **kwargs):
            return _univariate_forward(self, *args, state=state, verify=verify, **kwargs)

        attention.forward = types.MethodType(forward, attention)
        patched.append(attention)

    def uninstall():
        for attention in patched:
            vars(attention).pop('forward', None)

    return uninstall
