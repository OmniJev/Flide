"""NVFP4 activations and grouped expert products on SM120, using Triton only."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _encode_groups(x, global_scale, Q, S, group, GROUPS, BG: tl.constexpr):
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.minimum(tl.div_rn(amax, (6.0 * global_scale)), 448.0).to(tl.float8e4nv)
    tl.store(S + group, scale, group < GROUPS)
    inv = tl.where(scale.to(tl.float32) > 0, tl.div_rn(1.0, (scale.to(tl.float32) * global_scale)), 0.0)
    v = x * inv[:, None]
    lo, hi = tl.split(v.reshape(BG, 8, 2))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, $2, $1; mov.b32 $0, {r, 0, 0, 0}; }",
        constraints="=r,f,f", args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)
    tl.store(Q + group[:, None] * 8 + tl.arange(0, 8)[None, :], packed, group[:, None] < GROUPS)


@triton.jit
def _quantize(X, GLOBAL, Q, S, GROUPS, BG: tl.constexpr):
    group = tl.program_id(0) * BG + tl.arange(0, BG)
    c = tl.arange(0, 16)
    x = tl.load(X + group[:, None] * 16 + c[None, :], group[:, None] < GROUPS, 0.0).to(tl.float32)
    _encode_groups(x, tl.load(GLOBAL), Q, S, group, GROUPS, BG)


@triton.jit
def _swiglu_quantize(GU, GLOBAL, Q, S, GROUPS, N: tl.constexpr, STRIDE, BG: tl.constexpr):
    group = tl.program_id(0) * BG + tl.arange(0, BG)
    row = (group // (N // 16)).to(tl.int64)
    col = group % (N // 16) * 16
    offsets = row[:, None] * STRIDE + col[:, None] + tl.arange(0, 16)[None, :]
    valid = group[:, None] < GROUPS
    gate = tl.load(GU + offsets, valid, 0.0).to(tl.float32)
    up = tl.load(GU + offsets + N, valid, 0.0).to(tl.float32)
    # Keep the BF16 boundary of the separate SwiGLU before computing its block scale.
    x = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16).to(tl.float32)
    _encode_groups(x, tl.load(GLOBAL), Q, S, group, GROUPS, BG)


def quantize(x, global_scale):
    if x.ndim != 2 or x.shape[1] % 16 or x.dtype != torch.bfloat16:
        raise ValueError("NVFP4 activation must be BF16 [rows, K] with K divisible by 16")
    x = x.contiguous()
    q = torch.empty((x.shape[0], x.shape[1] // 2), dtype=torch.uint8, device=x.device)
    s = torch.empty((x.shape[0], x.shape[1] // 16), dtype=torch.float8_e4m3fn, device=x.device)
    groups = x.numel() // 16
    if groups:
        _quantize[(triton.cdiv(groups, 32),)](x, global_scale, q, s, groups, 32)
    return q, s


def swiglu_quantize(gu, n, global_scale):
    """SwiGLU rounded to BF16, then NVFP4, without an intermediate activation tensor."""
    if gu.ndim != 2 or gu.shape[1] != 2 * n or n % 16 or gu.dtype != torch.bfloat16 or gu.stride(1) != 1:
        raise ValueError("NVFP4 SwiGLU input must be BF16 [rows, 2*N] with contiguous columns and N divisible by 16")
    rows = gu.shape[0]
    q = torch.empty((rows, n // 2), dtype=torch.uint8, device=gu.device)
    s = torch.empty((rows, n // 16), dtype=torch.float8_e4m3fn, device=gu.device)
    groups = rows * (n // 16)
    if groups:
        _swiglu_quantize[(triton.cdiv(groups, 128),)](gu, global_scale, q, s, groups, n, gu.stride(0), 128)
    return q, s


@triton.jit
def _grouped(A, AS, B, BS, GLOBAL, INPUT_SCALE, C, WT, ROWS, BLOCK_E, USED,
             PAIRS, N: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr,
             BY_TOKEN: tl.constexpr, HALVES: tl.constexpr, MUL_W: tl.constexpr,
             BM: tl.constexpr = 32, BN: tl.constexpr = 64, BK: tl.constexpr = 128,
             GM: tl.constexpr = 0, STORE: tl.constexpr = "", NBM=0):
    if GM:
        pid = tl.program_id(0)
        width = GM * tl.cdiv(N, BN)
        first = pid // width * GM
        size = tl.minimum(NBM - first, GM)
        pm = first + pid % width % size
        pn = pid % width // size
    else:
        pm, pn = tl.program_id(0), tl.program_id(1)
    if pm * BM >= tl.load(USED):
        return
    e = tl.load(BLOCK_E + pm).to(tl.int64)
    pairs = tl.load(ROWS + pm * BM + tl.arange(0, BM))
    valid = pairs < PAIRS
    ar = pairs // TOPK if BY_TOKEN else pairs
    cols = pn * BN + tl.arange(0, BN)
    kp, ks = tl.arange(0, BK // 2), tl.arange(0, BK // 16)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k, s = block * (BK // 2) + kp, block * (BK // 16) + ks
        a = tl.load(A + ar[:, None].to(tl.int64) * (K // 2) + k[None, :], valid[:, None] & (k[None, :] < K // 2), 0)
        b = tl.load(B + (e * N + cols[None, :]) * (K // 2) + k[:, None],
                    (cols[None, :] < N) & (k[:, None] < K // 2), 0)
        sa = tl.load(AS + ar[:, None].to(tl.int64) * (K // 16) + s[None, :],
                     valid[:, None] & (s[None, :] < K // 16), 0.0)
        sb = tl.load(BS + (e * N + cols[:, None]) * (K // 16) + s[None, :],
                     (cols[:, None] < N) & (s[None, :] < K // 16), 0.0)
        acc = tl.dot_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    if HALVES:
        alpha = tl.load(GLOBAL + e * 2 + (cols >= N // 2).to(tl.int32))
    else:
        alpha = tl.load(GLOBAL + e)
    acc *= alpha * tl.load(INPUT_SCALE)
    if MUL_W:
        acc *= tl.load(WT + pairs, valid, 0.0)[:, None]
    tl.store(C + pairs[:, None].to(tl.int64) * N + cols[None, :], acc, valid[:, None] & (cols[None, :] < N), cache_modifier=STORE)


@triton.jit
def _gate_up_swiglu_quantize(A, AS, B, BS, GLOBAL, INPUT_SCALE, DOWN_SCALE, Q, S,
                             ROWS, BLOCK_E, USED, PAIRS, I: tl.constexpr, K: tl.constexpr,
                             TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    if pm * BM >= tl.load(USED):
        return
    expert = tl.load(BLOCK_E + pm).to(tl.int64)
    pairs = tl.load(ROWS + pm * BM + tl.arange(0, BM))
    valid = pairs < PAIRS
    tokens = (pairs // TOPK).to(tl.int64)

    # Pair the same BN channels from the gate and up halves of the packed weight.
    j = tl.arange(0, 2 * BN)
    cols = pn * BN + j % BN + (j // BN) * I
    weight_rows = expert * (2 * I) + cols
    kp, ks = tl.arange(0, BK // 2), tl.arange(0, BK // 16)
    acc = tl.full((BM, 2 * BN), 0, tl.float32)
    for block in range(K // BK):
        k, sc = block * (BK // 2) + kp, block * (BK // 16) + ks
        a = tl.load(A + tokens[:, None] * (K // 2) + k[None, :], valid[:, None], 0)
        b = tl.load(B + weight_rows[None, :] * (K // 2) + k[:, None])
        sa = tl.load(AS + tokens[:, None] * (K // 16) + sc[None, :], valid[:, None], 0.)
        sb = tl.load(BS + weight_rows[:, None] * (K // 16) + sc[None, :])
        acc = tl.dot_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)

    # Retain the BF16 projection and activation boundaries of the two-kernel path.
    alpha = tl.load(GLOBAL + expert * 2 + j // BN)
    rounded = (acc * (alpha * tl.load(INPUT_SCALE))).to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(rounded.reshape(BM, 2, BN).permute(0, 2, 1))
    value = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16).to(tl.float32)
    groups = pairs[:, None].to(tl.int64) * (I // 16) + pn * (BN // 16) + tl.arange(0, BN // 16)[None, :]
    count: tl.constexpr = BM * BN // 16
    _encode_groups(value.reshape(count, 16), tl.load(DOWN_SCALE), Q, S, groups.reshape(count),
                   PAIRS * (I // 16), count)


def expert_tiles(pairs):
    # Large prefills share one BM=64 alignment between both expert products.
    if pairs >= 16384:
        return (64, 128, 4), (64, 256, 8)
    return (32, 64, 4), (32, 64, 4)


def down_tuning(n, k, pairs, bm):
    # Group nearby row tiles through all columns; stream the large FP32 output.
    if n == 2048 and k == 512 and pairs >= 16384 and bm == 64:
        return 256, 3, 16, ".cs"
    return 128, 3, 0, ""


def grouped(a, sa, w, sw, global_scale, input_scale, out, routing, topk, by_token, halves=False, top=None,
            tile=(32, 64, 4), tuning=None):
    rows, blocks, used = routing
    n, k = w.shape[1], w.shape[2] * 2
    bm, bn, warps = tile
    if tuning is None:
        tuning = down_tuning(n, k, out.shape[0], bm) if not by_token else (128, 3, 0, "")
    bk, stages, gm, store = tuning
    nbm = blocks.numel()
    grid = (nbm * triton.cdiv(n, bn),) if gm else (nbm, triton.cdiv(n, bn))
    _grouped[grid](
        a, sa, w, sw, global_scale, input_scale, out, top if top is not None else out,
        rows, blocks, used, out.shape[0], n, k, topk, by_token, halves, top is not None,
        BM=bm, BN=bn, BK=bk, GM=gm, STORE=store, NBM=nbm, num_warps=warps, num_stages=stages)
    return out


@dataclass
class NvFp4Experts:
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor
    s2: torch.Tensor
    g13: torch.Tensor
    g2: torch.Tensor
    a13: torch.Tensor
    a2: torch.Tensor
    down_dtype: torch.dtype = torch.float32

    def __call__(self, x, top, ids, act):
        from .moe import align

        t, h = x.shape
        k, i = ids.shape[1], self.w2.shape[2] * 2
        # A missing weight tensor defers weighting to the slot-order combine.
        down_dtype = torch.bfloat16 if top is None else self.down_dtype
        if not t:
            return x.new_empty((0, h), dtype=down_dtype)
        gate_tile, down_tile = expert_tiles(t * k)
        routing = align(ids, self.w13.shape[0], bm=gate_tile[0])
        aq, sa = quantize(x, self.a13)
        if (h, i, k) == (2048, 512, 8):
            bm = gate_tile[0]
            bn, bk = (128, 128) if bm == 64 else (32, 256)
            vq = x.new_empty((t * k, i // 2), dtype=torch.uint8)
            sv = x.new_empty((t * k, i // 16), dtype=torch.float8_e4m3fn)
            _gate_up_swiglu_quantize[(routing[1].numel(), i // bn)](
                aq, sa, self.w13, self.s13, self.g13, self.a13, self.a2, vq, sv, *routing,
                t * k, i, h, k, bm, bn, bk, num_warps=4, num_stages=3)
        else:
            gu = grouped(aq, sa, self.w13, self.s13, self.g13, self.a13,
                         x.new_empty((t * k, i * 2)), routing, k, True, halves=True, tile=gate_tile)
            vq, sv = swiglu_quantize(gu, i, self.a2)
        return grouped(vq, sv, self.w2, self.s2, self.g2, self.a2,
                       x.new_empty((t * k, h), dtype=down_dtype), routing, k, False,
                       top=None if top is None else top.reshape(-1), tile=down_tile)
