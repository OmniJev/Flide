"""Row-wise pieces between the matrix products, in Triton: RMSNorm (with an optional residual add), SwiGLU,
gated RMSNorm, per-head q/k RMSNorm with the partial rotary embedding, and the attention output gate; for Gemma 4
the per-head norm with its rotary embedding, the post-norm with residual add, layer scalar and next pre-norm, and
GELU-tanh times a second input.

Arithmetic in float32 from load to store, rounded once to the activation dtype at the output (the eager
modules round some intermediate steps too; dropping those roundings brings bfloat16 runs closer to exact). Each kernel handles one row, or one head of a block of rows, per program, so the
token count is a runtime value and one compiled kernel serves every length.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _p2(n: int) -> int:
    return triton.next_power_of_2(n)


@triton.jit
def _rms_kernel(X, D, W, H_OUT, Y, N, sx, sd, sh, sy, eps, ADD: tl.constexpr, OFFSET: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(X + row * sx + cols, mask=m, other=0.0)
    if ADD:
        x = (x.to(tl.float32) + tl.load(D + row * sd + cols, mask=m, other=0.0).to(tl.float32)).to(X.dtype.element_ty)
        tl.store(H_OUT + row * sh + cols, x, mask=m)
    xf = x.to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(xf * xf, 0) / N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    if OFFSET:
        w = 1.0 + w
    tl.store(Y + row * sy + cols, (xf * inv * w).to(Y.dtype.element_ty), mask=m)


def rms_norm(x, w, eps: float, residual=None, offset: bool = True):
    """y = RMSNorm(x [+ residual]) with weight (1 + w) when `offset` (Qwen3.5's zero-centred norm), else w.
    With a residual, returns (x + residual, y)."""
    T, N = x.shape
    y = torch.empty_like(x)
    h = torch.empty_like(x) if residual is not None else x
    if T:
        d = residual if residual is not None else x
        _rms_kernel[(T,)](x, d, w, h, y, N, x.stride(0), d.stride(0), h.stride(0), y.stride(0), eps,
                          ADD=residual is not None, OFFSET=offset, BLOCK=_p2(N), num_warps=8 if N > 2048 else 4)
    return (h, y) if residual is not None else y


@triton.jit
def _swiglu_kernel(GU, OUT, N, sg, so, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < N
    a = tl.load(GU + row * sg + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(GU + row * sg + N + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + row * so + cols, (a / (1.0 + tl.exp(-a)) * b).to(OUT.dtype.element_ty), mask=m)


def swiglu(gu, n: int):
    """silu(gu[:, :n]) * gu[:, n:]."""
    T = gu.shape[0]
    out = torch.empty(T, n, device=gu.device, dtype=gu.dtype)
    if T:
        B = 1024
        _swiglu_kernel[(T, triton.cdiv(n, B))](gu, out, n, gu.stride(0), out.stride(0), BLOCK=B, num_warps=4)
    return out


@triton.jit
def _gated_rms_kernel(O, Z, W, OUT, T, DV, so_t, so_h, sz_t, sout_t, eps, BT: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mc = cols < DV
    m = (t < T)[:, None] & mc[None, :]
    o = tl.load(O + t[:, None] * so_t + h * so_h + cols[None, :], mask=m, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(o * o, 1) / DV + eps)
    w = tl.load(W + cols, mask=mc, other=0.0).to(tl.float32)
    y = o * inv[:, None] * w[None, :]
    z = tl.load(Z + t[:, None] * sz_t + h * DV + cols[None, :], mask=m, other=0.0).to(tl.float32)
    out = y * (z / (1.0 + tl.exp(-z)))
    tl.store(OUT + t[:, None] * sout_t + h * DV + cols[None, :], out.to(OUT.dtype.element_ty), mask=m)


def gated_rms(o, z, w, eps: float):
    """fla output o [t, heads, dv] and the gate z [t, heads * dv] (row-strided) -> [t, heads * dv]:
    RMSNorm over dv with weight w, times silu(z)."""
    T, H, DV = o.shape
    out = torch.empty(T, H * DV, device=o.device, dtype=o.dtype)
    if T:
        BT = max(1, 2048 // _p2(DV))
        _gated_rms_kernel[(triton.cdiv(T, BT), H)](o, z, w, out, T, DV, o.stride(0), o.stride(1), z.stride(0),
                                                   out.stride(0), eps, BT=BT, BLOCK=_p2(DV), num_warps=4)
    return out


@triton.jit
def _qk_kernel(X, COS, SIN, W, OUT, T, HD, R, sx_t, sx_h, sc_t, so_t, so_h, eps, BT: tl.constexpr,
               BLOCK: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    mt = t < T
    md = d < HD
    m = mt[:, None] & md[None, :]
    base = X + t[:, None] * sx_t + h * sx_h
    x = tl.load(base + d[None, :], mask=m, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, 1) / HD + eps)
    half = R // 2
    rot = d < R
    partner = tl.where(d < half, d + half, d - half)
    mr = mt[:, None] & (md & rot)[None, :]
    xp = tl.load(base + partner[None, :], mask=mr, other=0.0).to(tl.float32)
    w = 1.0 + tl.load(W + d, mask=md, other=0.0).to(tl.float32)
    wp = 1.0 + tl.load(W + partner, mask=md & rot, other=0.0).to(tl.float32)
    y = x * inv[:, None] * w[None, :]
    yp = xp * inv[:, None] * wp[None, :]
    c = tl.load(COS + t[:, None] * sc_t + d[None, :], mask=mr, other=1.0).to(tl.float32)
    s = tl.load(SIN + t[:, None] * sc_t + d[None, :], mask=mr, other=0.0).to(tl.float32)
    sign = tl.where(d < half, -1.0, 1.0)
    out = tl.where(rot[None, :], y * c + sign[None, :] * yp * s, y)
    tl.store(OUT + t[:, None] * so_t + h * so_h + d[None, :], out.to(OUT.dtype.element_ty), mask=m)


def qk_norm_rope(q, k, cos, sin, wq, wk, eps: float):
    """Per-head RMSNorm (weight 1 + w) of q [t, nh, hd] and k [t, nkv, hd] (any strides with a contiguous last
    dimension), then the rotary embedding on the first r = cos.shape[-1] dims. Returns contiguous q, k."""
    return norm_rope(q, cos, sin, wq, eps), norm_rope(k, cos, sin, wk, eps)


def norm_rope(x, cos, sin, w, eps: float):
    """`qk_norm_rope` for one of the two: x [t, heads, hd], cos and sin [t, r]."""
    T, N, HD = x.shape
    out = torch.empty(T, N, HD, device=x.device, dtype=x.dtype)
    if T:
        BT = max(1, 2048 // _p2(HD))
        _qk_kernel[(triton.cdiv(T, BT), N)](x, cos, sin, w, out, T, HD, cos.shape[-1], x.stride(0), x.stride(1),
                                            cos.stride(0), out.stride(0), out.stride(1), eps, BT=BT, BLOCK=_p2(HD),
                                            num_warps=4)
    return out


@triton.jit
def _gate_kernel(O, G, OUT, T, HD, so_t, so_h, sg_t, sg_h, sout_t, BT: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    m = (t < T)[:, None] & (d < HD)[None, :]
    o = tl.load(O + t[:, None] * so_t + h * so_h + d[None, :], mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + t[:, None] * sg_t + h * sg_h + d[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + t[:, None] * sout_t + h * HD + d[None, :], (o / (1.0 + tl.exp(-g))).to(OUT.dtype.element_ty), mask=m)


def gate(o, g):
    """o [t, heads, hd] times sigmoid(g [t, heads, hd]) -> [t, heads * hd]; both may be strided views with a
    contiguous last dimension (the gate is interleaved with q in the projection output)."""
    T, H, HD = o.shape
    out = torch.empty(T, H * HD, device=o.device, dtype=g.dtype)
    if T:
        BT = max(1, 2048 // _p2(HD))
        _gate_kernel[(triton.cdiv(T, BT), H)](o, g, out, T, HD, o.stride(0), o.stride(1), g.stride(0), g.stride(1),
                                              out.stride(0), BT=BT, BLOCK=_p2(HD), num_warps=4)
    return out


@triton.jit
def _gnorm_rope_kernel(X, COS, SIN, W, OUT, T, HD, sx_t, sx_h, sc_t, so_t, so_h, eps, HAS_W: tl.constexpr,
                       ROPE: tl.constexpr, BT: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    mt = t < T
    md = d < HD
    m = mt[:, None] & md[None, :]
    base = X + t[:, None] * sx_t + h * sx_h
    x = tl.load(base + d[None, :], mask=m, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, 1) / HD + eps)
    y = x * inv[:, None]
    if HAS_W:
        y = y * tl.load(W + d, mask=md, other=0.0).to(tl.float32)[None, :]
    if ROPE:
        half = HD // 2
        partner = tl.where(d < half, d + half, d - half)
        yp = tl.load(base + partner[None, :], mask=m, other=0.0).to(tl.float32) * inv[:, None]
        if HAS_W:
            yp = yp * tl.load(W + partner, mask=md, other=0.0).to(tl.float32)[None, :]
        c = tl.load(COS + t[:, None] * sc_t + d[None, :], mask=m, other=1.0).to(tl.float32)
        s = tl.load(SIN + t[:, None] * sc_t + d[None, :], mask=m, other=0.0).to(tl.float32)
        y = y * c + tl.where(d < half, -1.0, 1.0)[None, :] * yp * s
    tl.store(OUT + t[:, None] * so_t + h * so_h + d[None, :], y.to(OUT.dtype.element_ty), mask=m)


def gemma_norm_rope(x, cos, sin, w, eps: float):
    """Gemma 4's per-head RMSNorm of x [t, heads, hd] (weight w, none when w is None), then the rotary embedding
    over the whole head (rotate-half pairs d and d + hd/2; cos and sin [t, hd], None for no rotation). Any strides
    with a contiguous last dimension; returns a contiguous [t, heads, hd]."""
    T, N, HD = x.shape
    out = torch.empty(T, N, HD, device=x.device, dtype=x.dtype)
    if T:
        BT = max(1, 2048 // _p2(HD))
        rope = cos is not None
        c, s = (cos, sin) if rope else (x, x)
        ww = w if w is not None else x
        _gnorm_rope_kernel[(triton.cdiv(T, BT), N)](x, c, s, ww, out, T, HD, x.stride(0), x.stride(1),
                                                    cos.stride(0) if rope else 0, out.stride(0), out.stride(1), eps,
                                                    HAS_W=w is not None, ROPE=rope, BT=BT, BLOCK=_p2(HD), num_warps=4)
    return out


@triton.jit
def _sandwich_kernel(D, H, W1, S, W2, H_OUT, Y, N, sd, sh, sho, sy, eps, HAS_S: tl.constexpr, NEXT: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    d = tl.load(D + row * sd + cols, mask=m, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(d * d, 0) / N + eps)
    h = tl.load(H + row * sh + cols, mask=m, other=0.0).to(tl.float32)
    h = h + d * inv * tl.load(W1 + cols, mask=m, other=0.0).to(tl.float32)
    if HAS_S:
        h = h * tl.load(S).to(tl.float32)
    hb = h.to(H_OUT.dtype.element_ty)
    tl.store(H_OUT + row * sho + cols, hb, mask=m)
    if NEXT:
        hf = hb.to(tl.float32)
        inv2 = tl.math.rsqrt(tl.sum(hf * hf, 0) / N + eps)
        tl.store(Y + row * sy + cols, (hf * inv2 * tl.load(W2 + cols, mask=m, other=0.0).to(tl.float32))
                 .to(Y.dtype.element_ty), mask=m)


def sandwich(d, h, w1, eps: float, scale=None, w2=None):
    """Gemma 4's post-norm step: h + RMSNorm(d) * w1 (times the layer scalar `scale`, a one-element tensor), and,
    with w2, the next pre-norm RMSNorm(h) * w2 of the rounded result. Returns (h, y or None)."""
    T, N = d.shape
    ho = torch.empty_like(h)
    y = torch.empty_like(h) if w2 is not None else None
    if T:
        yy = y if y is not None else ho
        _sandwich_kernel[(T,)](d, h, w1, scale if scale is not None else w1, w2 if w2 is not None else w1, ho, yy, N,
                               d.stride(0), h.stride(0), ho.stride(0), yy.stride(0), eps, HAS_S=scale is not None,
                               NEXT=w2 is not None, BLOCK=_p2(N), num_warps=8 if N > 2048 else 4)
    return ho, y


@triton.jit
def _geglu_kernel(A, B, OUT, N, sa, sb, so, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < N
    a = tl.load(A + row * sa + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(B + row * sb + cols, mask=m, other=0.0).to(tl.float32)
    z = 0.7978845608028654 * (a + 0.044715 * a * a * a)
    g = 0.5 * a * (2.0 - 2.0 / (1.0 + tl.exp(2.0 * z)))
    tl.store(OUT + row * so + cols, (g * b).to(OUT.dtype.element_ty), mask=m)


def geglu(a, b):
    """GELU-tanh(a) * b for a, b [t, n] (row-strided views, contiguous last dimension) -> [t, n]."""
    T, n = a.shape
    out = torch.empty(T, n, device=a.device, dtype=a.dtype)
    if T:
        B = 1024
        _geglu_kernel[(T, triton.cdiv(n, B))](a, b, out, n, a.stride(0), b.stride(0), out.stride(0), BLOCK=B,
                                              num_warps=4)
    return out
