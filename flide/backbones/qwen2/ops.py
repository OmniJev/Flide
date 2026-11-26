"""Fused Qwen pointwise operations, rotary embedding and KV arena copies."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms(X, D, W, H, Y, N, sx, sd, eps, ADD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(X + row * sx + col, mask=valid, other=0).to(tl.float32)
    if ADD:
        d = tl.load(D + row * sd + col, mask=valid, other=0).to(tl.float32)
        x = (x + d).to(X.dtype.element_ty).to(tl.float32)
        tl.store(H + row * N + col, x, mask=valid)
    inv = tl.math.rsqrt(tl.sum(x * x, 0) / N + eps)
    weight = tl.load(W + col, mask=valid, other=0).to(tl.float32)
    # Qwen2RMSNorm casts the normalized hidden state before multiplying by the weight.
    normed = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * N + col, normed * weight, mask=valid)


def rms_norm(x, weight, eps: float, residual=None):
    rows, width = x.shape
    y = x.new_empty(x.shape)
    hidden = x.new_empty(x.shape) if residual is not None else x
    if rows:
        d = x if residual is None else residual
        _rms[(rows,)](x, d, weight, hidden, y, width, x.stride(0), d.stride(0), eps,
                      ADD=residual is not None, BLOCK=triton.next_power_of_2(width), num_warps=8 if width > 2048 else 4)
    return (hidden, y) if residual is not None else y


@triton.jit
def _swiglu(GU, OUT, N, S, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    gate = tl.load(GU + row * S + col, mask=valid, other=0).to(tl.float32)
    up = tl.load(GU + row * S + N + col, mask=valid, other=0).to(tl.float32)
    # The SiLU activation is stored at the projection dtype before the gate multiplies the up projection.
    activated = (gate / (1.0 + tl.exp(-gate))).to(GU.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * N + col, activated * up, mask=valid)


def swiglu(gate_up):
    rows, width = gate_up.shape
    n = width // 2
    out = gate_up.new_empty((rows, n))
    if rows:
        _swiglu[(rows, triton.cdiv(n, 1024))](gate_up, out, n, gate_up.stride(0), BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _rotary(X, COS, SIN, QW, KW, OUT, T, HEADS, HD: tl.constexpr, sx, sh, sc, eps, Q_HEADS: tl.constexpr,
            NORM: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    mask = (row[:, None] < T) & (col[None, :] < HD)
    other = tl.where(col < HD // 2, col + HD // 2, col - HD // 2)
    base = X + row[:, None] * sx + head * sh
    x = tl.load(base + col[None, :], mask=mask, other=0)
    y = tl.load(base + other[None, :], mask=mask, other=0)
    if NORM:
        # Qwen3 rounds after normalization and after multiplying by the head's weight.
        weight = tl.where(head < Q_HEADS, QW, KW)
        wx = tl.load(weight + col, mask=col < HD, other=0).to(tl.float32)
        wy = tl.load(weight + other, mask=col < HD, other=0).to(tl.float32)
        xf, yf = x.to(tl.float32), y.to(tl.float32)
        inv = tl.math.rsqrt(tl.sum(xf * xf, 1) / HD + eps)[:, None]
        x = ((xf * inv).to(x.dtype).to(tl.float32) * wx[None, :]).to(x.dtype)
        y = ((yf * inv).to(y.dtype).to(tl.float32) * wy[None, :]).to(y.dtype)
    c = tl.load(COS + row[:, None] * sc + col[None, :], mask=mask, other=0)
    s = tl.load(SIN + row[:, None] * sc + col[None, :], mask=mask, other=0)
    # Match the two products' rounding in Transformers before adding them.
    a = (x.to(tl.float32) * c.to(tl.float32)).to(x.dtype).to(tl.float32)
    b = (y.to(tl.float32) * s.to(tl.float32)).to(y.dtype).to(tl.float32)
    value = a + tl.where(col[None, :] < HD // 2, -b, b)
    tl.store(OUT + (row[:, None] * HEADS + head) * HD + col[None, :], value, mask=mask)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, norm=None, k_norm=None, q_heads=0) -> torch.Tensor:
    """Rotate Q/K, optionally normalizing each head before RoPE. With packed Q/K the weights change at `q_heads`."""
    tokens, heads, head_dim = x.shape
    out = torch.empty((tokens, heads, head_dim), dtype=x.dtype, device=x.device)
    if tokens:
        block = triton.next_power_of_2(head_dim)
        rows = max(1, 1024 // block)
        q_weight = x if norm is None else norm.weight
        k_weight = q_weight if k_norm is None else k_norm.weight
        eps = 0.0 if norm is None else norm.variance_epsilon
        _rotary[(triton.cdiv(tokens, rows), heads)](x, cos, sin, q_weight, k_weight, out, tokens, heads, head_dim,
                                                  x.stride(0), x.stride(1), cos.stride(0), eps, q_heads,
                                                  norm is not None, block, rows, enable_fp_fusion=False)
    return out


@triton.jit
def _cached_kv(K, V, ARENA_K, ARENA_V, CK, CV, SRC, DST, C, WIDTH, sk, sv,
               ROWS: tl.constexpr, COLS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    valid = (row[:, None] < C) & (col[None, :] < WIDTH)
    src = tl.load(SRC + row, mask=row < C, other=-1)
    dst = tl.load(DST + row, mask=row < C, other=-1)
    new = valid & (src[:, None] >= 0)
    old = valid & (src[:, None] < -1)
    stream_k = tl.load(K + src[:, None] * sk + col[None, :], mask=new, other=0)
    stream_v = tl.load(V + src[:, None] * sv + col[None, :], mask=new, other=0)
    entry = (-src[:, None].to(tl.int64) - 2) * WIDTH + col[None, :]
    cached_k = tl.load(ARENA_K + entry, mask=old, other=0)
    cached_v = tl.load(ARENA_V + entry, mask=old, other=0)
    key = tl.where(new, stream_k, cached_k)
    value = tl.where(new, stream_v, cached_v)
    context = row[:, None] * WIDTH + col[None, :]
    tl.store(CK + context, key, mask=valid)
    tl.store(CV + context, value, mask=valid)
    save = dst[:, None].to(tl.int64) * WIDTH + col[None, :]
    write = valid & (dst[:, None] >= 0)
    tl.store(ARENA_K + save, key, mask=write)
    tl.store(ARENA_V + save, value, mask=write)


def cached_kv(k, v, arena_k, arena_v, source, destination):
    """Assemble the context and write save points in one kernel per layer.

    Source rows start at zero for the stream and at -2 for the arena. -1 marks padding. Destination rows are arena
    indices or -1. The cache reserves disjoint destination rows before a forward that reads older entries.
    """
    count = source.numel()
    width = k.shape[1] * k.shape[2]
    keys = k.new_empty((count, *k.shape[1:]))
    values = v.new_empty((count, *v.shape[1:]))
    _cached_kv[(triton.cdiv(count, 4), triton.cdiv(width, 128))](
        k, v, arena_k, arena_v, keys, values, source, destination, count, width, k.stride(0), v.stride(0), 4, 128,
    )
    return keys, values


class ArenaContext:
    """KV rows for the first attention source, assembled from the stream and Flide's session arena."""

    def __init__(self, store, source, destination):
        self.store, self.source, self.destination = store, source, destination
        self.meta = (source, source, source)   # exact ranges do not read context token metadata

    def keys(self, layer, k, v):
        return cached_kv(k, v, self.store.K[layer], self.store.V[layer], self.source, self.destination)
