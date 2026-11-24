"""Copy new K/V to owned caches and gather attention inputs in one dispatch.

Imported only for eligible CUDA decode calls. There is no floating-point math:
normalization, RoPE, query scaling, masks and native attention stay upstream.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['capacity'])
def _append_pack(Pointers, Key, Value, PackedKey, PackedValue,
                 capacity,
                 HEADS: tl.constexpr, DIM: tl.constexpr, PATCHES: tl.constexpr,
                 K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
                 V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < capacity * HEADS * DIM
    time = offsets // (HEADS * DIM)
    head, dim = offsets // DIM % HEADS, offsets % DIM
    route = Pointers + row * 3
    owned_key = tl.load(route).to(tl.pointer_type(tl.float32))
    owned_value = tl.load(route + 1).to(tl.pointer_type(tl.float32))
    start = tl.load(route + 2)
    new = valid & (time >= start) & (time < start + PATCHES)
    # Never read the owned append span: other blocks may already be writing it.
    # Each destination element is written once; history/tail bytes stay untouched.
    old_key = tl.load(owned_key + offsets, mask=valid & ~new, other=0)
    old_value = tl.load(owned_value + offsets, mask=valid & ~new, other=0)
    new_key = tl.load(Key + row * K0 + (time - start) * K1 + head * K2 + dim * K3,
                      mask=new, other=0)
    new_value = tl.load(Value + row * V0 + (time - start) * V1 + head * V2 + dim * V3,
                        mask=new, other=0)
    key = tl.where(new, new_key, old_key)
    value = tl.where(new, new_value, old_value)
    tl.store(owned_key + offsets, new_key, mask=new)
    tl.store(owned_value + offsets, new_value, mask=new)
    packed_offset = row * capacity * HEADS * DIM + offsets
    tl.store(PackedKey + packed_offset, key, mask=valid)
    tl.store(PackedValue + packed_offset, value, mask=valid)


def append_and_pack(pointers, layer, key, value, packed_key, packed_value):
    rows, capacity, heads, dim = packed_key.shape
    _append_pack[(triton.cdiv(capacity * heads * dim, 1024), rows)](
        pointers[layer], key, value, packed_key, packed_value,
        capacity, heads, dim, key.shape[1],
        *key.stride(), *value.stride(), BLOCK=1024,
    )
