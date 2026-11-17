# Copyright 2026 vTSFM contributors
#
# The forward functions below are derived from Google LLC's TimesFM 2.5 Torch
# implementation (Copyright 2025 Google LLC), licensed under Apache-2.0.
# https://github.com/google-research/timesfm/tree/e51928e27119cb17bebc005be2696b75e0a9e688
# See the repository LICENSE.
"""Batched neural layers with request-local native attention for block decoding.

Inputs are FP32 ``[sum(request_rows), 4, 32]`` with matching boolean masks.
``cache_groups[request][layer]`` owns that request's original full-width K/V;
``positions[request]`` is its CPU-known physical write position. Each request's
layer caches share their geometry and counters, maintained by the stepper's
initialization and identical per-layer updates. No counter is read back to CPU.

Tokenizer, QKV, RoPE, normalization, output projection, MLP and forecast heads
stay batched. Native masks and integer position/count metadata are built once
per forward and reused across layers. Each request updates its own K/V in place.
Equal-width caches use temporary per-layer K/V concatenation for one batched
native attention call; different widths keep separate native calls. Temporary
packs never replace the owned caches or require cache copyback.

The caller owns inference mode and request lifetimes. This helper never changes
model methods or weights. Prefill and whole-forecast calls keep the original
native path; singleton AR also batches its two flip groups through this helper.
Call ``validate_block_decode_source`` once in the stepper constructor.
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
            raise ValueError(f'Block decoding requires the pinned TimesFM {name}.forward source')


def _decode_groups(model, inputs, masks, cache_groups, positions):
    """Validate CPU geometry before any cache can be changed."""
    if inputs.ndim != 3 or inputs.shape[1:] != (model.m, model.p) or masks.shape != inputs.shape:
        raise ValueError('Block decode requires [rows,4,32] inputs and matching masks')
    if not cache_groups or len(cache_groups) != len(positions):
        raise ValueError('Block decode needs one CPU position per nonempty cache group')
    groups, cursor = [], 0
    for caches, position in zip(cache_groups, positions, strict=True):
        if type(position) is not int or position < 0 or len(caches) != model.x:
            raise ValueError('Block cache groups need nonnegative integer positions and every layer')
        # The stepper constructs every layer with identical request geometry;
        # dtype, device and independent ownership do not change between blocks.
        rows, capacity = caches[0].key.shape[:2]
        if rows < 1 or position + model.m > capacity:
            raise ValueError('Block cache request rows or write capacity are invalid')
        if any(cache.key.shape[:2] != (rows, capacity) for cache in caches[1:]):
            raise ValueError('Block layer caches must share request rows and capacity')
        groups.append((slice(cursor, cursor + rows), caches, position))
        cursor += rows
    if cursor != inputs.shape[0]:
        raise ValueError('Block cache group rows must exactly partition the neural batch')
    return groups


def _decode_positions(inputs, masks, groups):
    """Layer-invariant native positions/counters, shared by attention backends.

    All layers start with identical counters, see the same patch masks and
    advance once. Reading layer zero therefore gives the native values for
    every layer; each layer still updates its own independent counters below.
    """
    n_patches = inputs.shape[1]
    next_index = torch.cat([caches[0].next_index for _, caches, _ in groups]).clone()
    num_masked = (torch.sum(masks[..., -1].to(torch.int32), dim=-1)
                  + torch.cat([caches[0].num_masked for _, caches, _ in groups]))
    position = (torch.arange(n_patches, device=inputs.device)[None, :]
                + next_index[:, None] - num_masked[:, None])
    return position, num_masked, next_index


def _decode_metadata(inputs, masks, groups):
    """Build request-local dense causal masks once per neural forward."""
    n_patches = inputs.shape[1]
    position, num_masked, next_index = _decode_positions(inputs, masks, groups)
    attn_masks = [make_attn_mask(
        query_length=n_patches,
        num_all_masked_kv=num_masked[section],
        query_index_offset=next_index[section],
        kv_length=caches[0].value.shape[1],
    ) for section, caches, _ in groups]
    return position, num_masked, attn_masks


def _attention_forward(attention, inputs_q, groups, layer_index, position, num_masked, attn_masks,
                       batched_mask, attention_backend=None, attention_state=None):
    # Follow pinned MultiHeadAttention.forward math through the scaled Q/K/V.
    b, n_patches, _ = inputs_q.shape

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

    if attention.use_rotary_position_embeddings:
        query = attention.rotary_position_embedding(query, position)
        key = attention.rotary_position_embedding(key, position)

    query = attention.query_ln(query)
    key = attention.key_ln(key)

    if attention.use_per_dim_scale:
        query = attention.per_dim_scale(query)

    if attention_backend is not None:
        # The experimental boundary sees already transformed Q/K/V. It owns
        # cache append and attention only; every neural operation is shared.
        x = attention_backend.attend(layer_index, query, key, value, attention_state)
        for section, caches, _ in groups:
            caches[layer_index].next_index += n_patches
            caches[layer_index].num_masked = num_masked[section].clone()
    else:
        for section, caches, start in groups:
            cache = caches[layer_index]
            stop = start + n_patches
            # Python integer slices use known request geometry, avoiding the native
            # CUDA-scalar slice-bound synchronization as well as full-cache copies.
            cache.key[:, start:stop].copy_(key[section])
            cache.value[:, start:stop].copy_(value[section])
            cache.next_index += n_patches
            cache.num_masked = num_masked[section].clone()

        if batched_mask is not None:
            # Full native widths and per-row masks are unchanged. Only these
            # temporary gathers combine independent owners; no K/V is scattered back.
            key = torch.cat([caches[layer_index].key for _, caches, _ in groups])
            value = torch.cat([caches[layer_index].value for _, caches, _ in groups])
            x = attention.attention_fn(query, key, value, mask=batched_mask)
        else:
            attended = [attention.attention_fn(
                query[section], caches[layer_index].key, caches[layer_index].value, mask=attn_mask,
            ) for (section, caches, _), attn_mask in zip(groups, attn_masks, strict=True)]
            x = torch.cat(attended, dim=0)
    x = x.reshape(b, n_patches, attention.in_features)
    out = attention.out(x)
    return out


def _transformer_forward(layer, input_embeddings, groups, layer_index, position, num_masked, attn_masks,
                         batched_mask, attention_backend=None, attention_state=None):
    # Preserve Transformer.forward's norm, residual and feedforward expressions.
    attn_output = _attention_forward(
        layer.attn, layer.pre_attn_ln(input_embeddings), groups, layer_index,
        position, num_masked, attn_masks, batched_mask, attention_backend, attention_state,
    )
    attn_output = layer.post_attn_ln(attn_output) + input_embeddings
    output_embeddings = (
        layer.post_ff_ln(layer.ff1(layer.activation(layer.ff0(layer.pre_ff_ln(attn_output)))))
        + attn_output
    )
    return output_embeddings


def block_decode_forward(model, inputs, masks, cache_groups, positions, *, attention_backend=None):
    """Batch neural layers while native attention updates each owned cache.

    Return the native ``((input_embeddings, output_embeddings, point, spread),
    cache_groups)`` structure, keeping the supplied request-major cache objects.
    This function advances four input patches; prefill uses the native model.

    ``attention_backend`` is an experimental cache/attention boundary. Its
    ``prepare(model, inputs, masks, cache_groups, positions)`` validates geometry
    and returns ``(groups, position, num_masked, opaque_state)``. Groups contain
    ``(row_slice, layer_caches, physical_position)``; counters retain native
    semantics even when their dense K/V payloads are empty. Preparation runs
    once, before model calls. ``attend(layer, q, k, v, state)`` receives scaled,
    normalized/rotated Q/K and V, all ``[rows,4,heads,head_dim]``, and returns an
    attended tensor of the same shape. It must not mutate the supplied counters;
    this shared helper advances them. Normal serving omits the argument.
    """
    if attention_backend is None:
        groups = _decode_groups(model, inputs, masks, cache_groups, positions)
        position, num_masked, attn_masks = _decode_metadata(inputs, masks, groups)
        capacity = groups[0][1][0].key.shape[1]
        batched_mask = (torch.cat(attn_masks) if all(
            caches[0].key.shape[1] == capacity for _, caches, _ in groups
        ) else None)
        attention_state = None
    else:
        groups, position, num_masked, attention_state = attention_backend.prepare(
            model, inputs, masks, cache_groups, positions)
        attn_masks = batched_mask = None
    tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
    input_embeddings = model.tokenizer(tokenizer_inputs)

    output_embeddings = input_embeddings
    for i, layer in enumerate(model.stacked_xf):
        output_embeddings = _transformer_forward(
            layer, output_embeddings, groups, i, position, num_masked, attn_masks, batched_mask,
            attention_backend, attention_state,
        )
    output_ts = model.output_projection_point(output_embeddings)
    output_quantile_spread = model.output_projection_quantiles(output_embeddings)

    return (input_embeddings, output_embeddings, output_ts, output_quantile_spread), cache_groups
