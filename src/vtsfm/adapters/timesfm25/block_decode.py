# Copyright 2026 vTSFM contributors
#
# The forward functions below are derived from Google LLC's TimesFM 2.5 Torch
# implementation (Copyright 2025 Google LLC), licensed under Apache-2.0.
# https://github.com/google-research/timesfm/tree/e51928e27119cb17bebc005be2696b75e0a9e688
# See the repository LICENSE.
"""Private row-position-aware forward for TimesFM forecast-block decoding.

Inputs are FP32 ``[rows, patches, 32]`` with matching boolean masks. Each layer's
DecodeCache owns ``[rows, capacity, heads, head_dim]`` K/V tensors and per-row
counters. ``next_index`` stays int32; masked counts retain native sum promotion.
The caller owns capacity validation, inference mode and cache
packing/writeback. Full cache capacity and the native attention function remain
unchanged. The only mathematical-path change is replacing the pinned attention
forward's common-position K/V write with independent per-row writes.

This helper neither changes module methods/parameters nor replaces the original
whole-forecast or uniform-position forward. Call ``validate_block_decode_source``
once when constructing its stepper, rather than inspecting source in every call.
"""

import hashlib
import inspect
import textwrap

import torch
from timesfm.torch.transformer import make_attn_mask


_PINNED_FORWARD_SHA256 = {
    'model': '357dc38ec0baac7984d0607f632ba68de0cf9d66c30adee2856e2d0efdae1abc',
    'transformer': '6f1cdfc64dd87b553003570edcce91e4c38348e12677856c9e390052820e9a67',
    'attention': 'd254502e9ad38240496e2915172865be8c5db45eaaa5366e3a6a5ebf6b2c39e3',
}


def validate_block_decode_source(model):
    """Reject changed native forward algorithms before enabling the copied path.

    Check class methods deliberately: an execution policy may wrap an instance's
    forward without changing the pinned algorithm used by this private helper.
    """
    methods = [('model', type(model).forward)]
    for layer in model.stacked_xf:
        methods.extend((('transformer', type(layer).forward),
                        ('attention', type(layer.attn).forward)))
    checked = set()
    for name, method in methods:
        if (name, method) in checked:
            continue
        checked.add((name, method))
        try:
            source = textwrap.dedent(inspect.getsource(method)).strip()
        except (OSError, TypeError) as error:
            raise ValueError(f'Cannot verify pinned TimesFM {name}.forward source') from error
        if hashlib.sha256(source.encode()).hexdigest() != _PINNED_FORWARD_SHA256[name]:
            raise ValueError(f'Mixed-position decoding requires the pinned TimesFM {name}.forward source')


def _attention_forward(attention, inputs_q, *, decode_cache=None, patch_mask=None):
    # Match MultiHeadAttention.forward at the pinned revision, including order.
    b, n_patches, _ = inputs_q.shape
    if patch_mask is None:
        patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)

    if attention.fuse_qkv:
        qkv = attention.qkv_proj(inputs_q)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = query.view(b, n_patches, attention.num_heads, attention.head_dim)
        key = key.view(b, n_patches, attention.num_heads, attention.head_dim)
        value = value.view(b, n_patches, attention.num_heads, attention.head_dim)
    else:
        query = attention.query(inputs_q).view(b, n_patches, attention.num_heads, attention.head_dim)
        key = attention.key(inputs_q).view(b, n_patches, attention.num_heads, attention.head_dim)
        value = attention.value(inputs_q).view(b, n_patches, attention.num_heads, attention.head_dim)

    if decode_cache is None:
        num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
        next_index = torch.zeros_like(num_masked, dtype=torch.int32)
    else:
        num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1) + decode_cache.num_masked
        next_index = decode_cache.next_index.clone()

    if attention.use_rotary_position_embeddings:
        position = (torch.arange(n_patches, device=inputs_q.device)[None, :]
                    + next_index[:, None] - num_masked[:, None])
        query = attention.rotary_position_embedding(query, position)
        key = attention.rotary_position_embedding(key, position)

    query = attention.query_ln(query)
    key = attention.key_ln(key)

    if attention.use_per_dim_scale:
        query = attention.per_dim_scale(query)

    if decode_cache is not None:
        _, decode_cache_size, _, _ = decode_cache.value.shape

        # Unlike upstream's next_index[0] slice, each row writes its own prefix
        # extension. Unique row/column pairs avoid any scatter accumulation.
        rows = torch.arange(b, dtype=torch.int64, device=inputs_q.device)[:, None]
        columns = (next_index.to(torch.int64)[:, None]
                   + torch.arange(n_patches, dtype=torch.int64, device=inputs_q.device)[None, :])
        decode_cache.key[rows, columns] = key
        decode_cache.value[rows, columns] = value

        key = decode_cache.key
        value = decode_cache.value
        decode_cache.next_index += n_patches
        decode_cache.num_masked = num_masked
        attn_mask = make_attn_mask(
            query_length=n_patches,
            num_all_masked_kv=num_masked,
            query_index_offset=next_index,
            kv_length=decode_cache_size,
        )
    else:
        attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)

    x = attention.attention_fn(query, key, value, mask=attn_mask)
    x = x.reshape(b, n_patches, attention.in_features)
    out = attention.out(x)
    return out, decode_cache


def _transformer_forward(layer, input_embeddings, patch_mask, decode_cache=None):
    # Preserve Transformer.forward's norm, residual and feedforward expressions.
    attn_output, decode_cache = _attention_forward(
        layer.attn, inputs_q=layer.pre_attn_ln(input_embeddings),
        decode_cache=decode_cache, patch_mask=patch_mask,
    )
    attn_output = layer.post_attn_ln(attn_output) + input_embeddings
    output_embeddings = (
        layer.post_ff_ln(layer.ff1(layer.activation(layer.ff0(layer.pre_ff_ln(attn_output)))))
        + attn_output
    )
    return output_embeddings, decode_cache


def block_decode_forward(model, inputs, masks, decode_caches=None):
    """Run native model-forward math with per-row cache positions, in place.

    Return the native ``((input_embeddings, output_embeddings, point, spread),
    caches)`` structure. Only the supplied caches are mutated; the model and its
    instance/class forward methods remain untouched.
    """
    tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
    input_embeddings = model.tokenizer(tokenizer_inputs)

    if decode_caches is None:
        decode_caches = [None] * model.x

    output_embeddings = input_embeddings
    new_decode_caches = []
    for i, layer in enumerate(model.stacked_xf):
        output_embeddings, new_cache = _transformer_forward(
            layer, output_embeddings, masks[..., -1], decode_caches[i],
        )
        new_decode_caches.append(new_cache)
    output_ts = model.output_projection_point(output_embeddings)
    output_quantile_spread = model.output_projection_quantiles(output_embeddings)

    return (input_embeddings, output_embeddings, output_ts, output_quantile_spread), new_decode_caches
