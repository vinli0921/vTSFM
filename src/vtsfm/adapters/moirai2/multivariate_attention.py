"""Experimental native Moirai attention preserving the binary variate bias.

This is an explicit installer for a pinned, evaluation-mode ``Moirai2Module``;
the serving adapter still accepts independent univariate histories only. The
upstream module constructs its packed causal boolean mask without modification.
One head-shared FlexAttention BlockMask represents that mask, while score_mod
adds the learned same/cross-variate bias without a dense per-head float tensor.

CUDA execution uses compiled FlexAttention with IEEE FP32 dot products. CPU
execution is an unfused semantic reference, not evidence of a fused kernel or a
speedup. The upstream shared boolean mask and packed scaler remain quadratic.

Projection ordering below follows Salesforce uni2ts (Apache-2.0), commit
cfd46d4510ed8896f263116f32928eede05b0a75, module/attention.py.
"""

import hashlib
import inspect
import types
import weakref


PINNED_SOURCE_SHA256 = {
    'GroupedQueryAttention.forward': '0cbd7ac8bcde724a399390d4e4cf48f877dc341e5eede35561e2412c2a33d5c3',
    'BinaryAttentionBias.forward': '23ff2b8b5a1c535d48cab207814750e0a3f0dbdb5b8baf08901dd9615d9fd311',
    'Moirai2Module.forward': '0b8ff86b0ff43a681ff7975ba42ad7b6d191c4f4c41f346b7290da7563d8a37f',
    'packed_causal_attention_mask': '5dd6f8b54c9c3411e5f9fd628f62629d139b8cc2749ddae719bcb51b1cdb7440',
}


def upstream_source_digests():
    """Read the guarded native implementation; imports runtime packages lazily."""
    from uni2ts.common.torch_util import packed_causal_attention_mask
    from uni2ts.model.moirai2 import Moirai2Module
    from uni2ts.module.attention import GroupedQueryAttention
    from uni2ts.module.position import BinaryAttentionBias

    functions = (GroupedQueryAttention.forward, BinaryAttentionBias.forward,
                 Moirai2Module.forward, packed_causal_attention_mask)
    return {name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            for name, fn in zip(PINNED_SOURCE_SHA256, functions)}


class _MaskState:
    """Share block metadata across layers, never across native module calls."""

    def __init__(self):
        self.reset()

    def reset(self, *_):
        self.ref = None
        self.block_mask = None

    def get(self, mask, batch, query_length, key_length):
        from torch.nn.attention.flex_attention import create_block_mask

        if self.ref is not None and self.ref() is mask:
            return self.block_mask
        flat_mask = mask.reshape(batch, query_length, key_length)

        def packed_mask(b, h, q, k):
            # Flex's boundary tiles can contain indices past the real length.
            valid = (q < query_length) & (k < key_length)
            return valid & flat_mask[b, q.clamp(max=query_length - 1), k.clamp(max=key_length - 1)]

        self.block_mask = create_block_mask(
            packed_mask, B=batch, H=None, Q_LEN=query_length, KV_LEN=key_length,
            device=mask.device,
        )
        self.ref = weakref.ref(mask)
        return self.block_mask


def _multivariate_forward(self, query, key, value, attn_mask=None, query_var_id=None,
                          kv_var_id=None, query_time_id=None, kv_time_id=None, *, state, calls):
    import torch
    from einops import rearrange, repeat
    from torch.nn.attention.flex_attention import flex_attention

    if self.training or self.attn_dropout_p != 0:
        raise ValueError('Native multivariate FlexAttention requires eval mode and zero attention dropout')
    if torch.is_grad_enabled():
        raise ValueError('Native multivariate FlexAttention is inference-only; use torch.inference_mode()')
    if query.device.type not in ('cpu', 'cuda'):
        raise ValueError('Native multivariate FlexAttention supports CPU semantic checks or CUDA execution')
    if torch.is_autocast_enabled(query.device.type):
        raise ValueError('Native multivariate FlexAttention requires FP32 execution with autocast disabled')
    if query.dtype != torch.float32 or key.dtype != torch.float32 or value.dtype != torch.float32:
        raise ValueError('Native multivariate FlexAttention currently requires FP32 inputs')
    if attn_mask is None or attn_mask.dtype != torch.bool:
        raise ValueError('Native multivariate FlexAttention requires the upstream packed boolean mask')
    lead, length, width = query.shape[:-2], query.shape[-2], query.shape[-1]
    key_length = key.shape[-2]
    if (key.shape[:-2] != lead or value.shape[:-2] != lead or value.shape[-2] != key_length
            or key.shape[-1] != width or value.shape[-1] != width
            or attn_mask.shape != (*lead, length, key_length) or length == 0 or key_length == 0):
        raise ValueError('Native attention inputs and packed mask must have matching batch/sequence axes')

    groups, per_group = self.num_groups, self.heads_per_group
    query = self.q_norm(rearrange(self.q_proj(query), '... l (g h d) -> ... g h l d', g=groups, h=per_group))
    key = self.k_norm(repeat(self.k_proj(key), '... l (g d) -> ... g h l d', g=groups, h=per_group))
    value = repeat(self.v_proj(value), '... l (g d) -> ... g h l d', g=groups, h=per_group)
    query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
    query_time_id, kv_time_id = self._get_time_id(query, key, query_time_id, kv_time_id)
    query, key = self._qk_proj(query, key, query_var_id=query_var_id, kv_var_id=kv_var_id,
                               query_time_id=query_time_id, kv_time_id=kv_time_id)

    heads, dim = groups * per_group, self.head_dim
    query = query.reshape(-1, heads, length, dim)
    key = key.reshape(-1, heads, key_length, dim)
    value = value.reshape(-1, heads, key_length, dim)
    q_ids, k_ids = query_var_id.reshape(-1, length), kv_var_id.reshape(-1, key_length)
    weight = self.var_attn_bias.emb.weight
    if weight.dtype != torch.float32:
        raise ValueError('Native multivariate FlexAttention currently requires FP32 bias weights')

    def variate_bias(score, b, h, q, k):
        same = q_ids[b, q.clamp(max=length - 1)] == k_ids[b, k.clamp(max=key_length - 1)]
        # Literal BinaryAttentionBias order: row 0 cross, row 1 same.
        return score + ((~same) * weight[0, h] + same * weight[1, h])

    block_mask = state.get(attn_mask, query.shape[0], length, key_length)
    if query.device.type == 'cuda':
        if calls.get('cuda') is None:
            calls['cuda'] = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        attention = calls['cuda']
    elif query.device.type == 'cpu':
        attention = flex_attention
    # Pinned Torch 2.14 gen_defines inserts this option as a source literal;
    # its own get_float32_precision() likewise returns "'ieee'", with quotes.
    out = attention(query, key, value, block_mask=block_mask, score_mod=variate_bias,
                    scale=self.softmax_scale, kernel_options={'FLOAT32_PRECISION': "'ieee'"})
    out = out.reshape(*lead, groups, per_group, length, dim)
    return self.out_proj(rearrange(out, '... g h l d -> ... l (g h d)'))


def install_multivariate_attention(module):
    """Install on one pinned native module and return an idempotent undo callable.

    No classes, serving defaults, request schemas, model weights, or upstream
    forecasting code are modified. Unknown configurations are rejected before
    any instance changes. Invoke native forecasts inside inference_mode().
    """
    from uni2ts.model.moirai2 import Moirai2Module
    from uni2ts.module.attention import GroupedQueryAttention
    from uni2ts.module.position import BinaryAttentionBias

    if type(module) is not Moirai2Module:
        raise TypeError('Expected the pinned native Moirai2Module')
    actual = upstream_source_digests()
    if actual != PINNED_SOURCE_SHA256:
        raise RuntimeError(f'Pinned uni2ts source changed; revalidate multivariate attention: {actual}')
    if module.training or 'forward' in vars(module):
        raise ValueError('Install on an unpatched evaluation-mode native Moirai2Module')
    attentions = [layer.self_attn for layer in module.encoder.layers]
    if not attentions:
        raise ValueError('Expected at least one native Moirai attention layer')
    for attention in attentions:
        bias = attention.var_attn_bias
        if (type(attention) is not GroupedQueryAttention or 'forward' in vars(attention)
                or type(bias) is not BinaryAttentionBias or 'forward' in vars(bias)
                or attention.training or attention.attn_dropout_p != 0
                or attention.time_attn_bias is not None or attention.var_qk_proj is not None
                or bias.emb.weight.shape != (2, attention.num_heads)):
            raise ValueError('Unsupported or already patched native Moirai attention configuration')

    state, calls, installed = _MaskState(), {}, []
    pre_hook = module.register_forward_pre_hook(state.reset)
    post_hook = module.register_forward_hook(state.reset, always_call=True)
    try:
        for attention in attentions:
            def forward(self, *args, **kwargs):
                return _multivariate_forward(self, *args, state=state, calls=calls, **kwargs)

            bound = types.MethodType(forward, attention)
            attention.forward = bound
            installed.append((attention, bound))
    except BaseException:
        pre_hook.remove()
        post_hook.remove()
        for attention, bound in installed:
            if vars(attention).get('forward') is bound:
                del attention.forward
        raise

    def uninstall():
        if any('forward' in vars(attention) and vars(attention)['forward'] is not bound
               for attention, bound in installed):
            raise RuntimeError('An installed attention forward was replaced; refusing to remove another patch')
        pre_hook.remove()
        post_hook.remove()
        for attention, bound in installed:
            if vars(attention).get('forward') is bound:
                del attention.forward
        state.reset()
        calls.clear()

    return uninstall
