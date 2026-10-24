"""The Triton pointwise kernels against the eager PyTorch functions they replace (backbones/qwen35/ops.py run
without torch.compile), on the shapes and strides of the Qwen3.5 decoder. The kernels round once at the
output where eager rounds some intermediate steps as well, so bfloat16 results may differ by a few units
of the last place; the line is 4.

    python tests/test_pointwise.py
"""
from __future__ import annotations

import sys
import torch.nn.functional as F
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels import pointwise as pw


def _rms(x, w, eps: float):
    """Qwen3.5 RMSNorm: zero-centred weight, statistics in float32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)

def _add_rms(h, d, w, eps: float):
    h = h + d
    return h, _rms(h, w, eps)

def _swiglu(gu, n: int):
    return F.silu(gu[:, :n]) * gu[:, n:]

def _gated_rms(o, z, w, eps: float):
    """fla output [t, heads, dv] and the strided gate z [t, heads * dv] -> [t, heads * dv]."""
    t, nh, dv = o.shape
    of = o.float()
    y = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)).to(o.dtype) * w
    return (y.float() * F.silu(z.view(t, nh, dv).float())).to(o.dtype).view(t, nh * dv)

def _norm_rope(x, cos, sin, w, eps: float):
    """Per-head RMSNorm of x [t, heads, hd] and the partial rotary embedding, in the input dtype."""
    r = cos.shape[-1]
    c, s = cos[:, None, :].float(), sin[:, None, :].float()
    y = _rms(x, w, eps).float()
    x1, x2, rest = y[..., : r // 2], y[..., r // 2: r], y[..., r:]
    rot = torch.cat([x1 * c[..., : r // 2] - x2 * s[..., : r // 2], x2 * c[..., r // 2:] + x1 * s[..., r // 2:]], -1)
    return torch.cat([rot, rest], -1).to(x.dtype)

def _qk_norm_rope(q, k, cos, sin, wq, wk, eps: float):
    """Per-head RMSNorm of q and k and the partial rotary embedding, in the input dtype."""
    return _norm_rope(q, cos, sin, wq, eps), _norm_rope(k, cos, sin, wk, eps)

def _gate(o, gate):
    """Attention output [t, heads, hd] times the sigmoid of the strided gate [t, heads, hd] -> [t, heads * hd]."""
    t = o.shape[0]
    return (o.float() * torch.sigmoid(gate.float())).to(gate.dtype).reshape(t, -1)


def ulps(a, b):
    """Largest |a - b| divided by the dtype spacing at |b| (0 means identical)."""
    a, b = a.float(), b.float()
    eps = torch.finfo(torch.bfloat16 if a.dtype == torch.bfloat16 else b.dtype).eps
    spacing = (b.abs() * eps).clamp_min(1e-30)
    return float(((a - b).abs() / spacing).max())


def check(name, got, want, ok_ulps):
    got, want = (got if isinstance(got, tuple) else (got,)), (want if isinstance(want, tuple) else (want,))
    worst, same = 0.0, True
    for g, w in zip(got, want):
        same &= bool(torch.equal(g, w))
        worst = max(worst, float((g.float() - w.float()).abs().max() / (w.float().abs().max() + 1e-30)))
    ok = same or worst <= ok_ulps * torch.finfo(want[0].dtype).eps
    print(f"  {name:14} {str(want[0].dtype):15} identical {same}  max rel diff {worst:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


def main():
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(0)
    ok = True
    for dt in (torch.float32, torch.bfloat16):
        T, H, I = 777, 1024, 3584
        r = lambda *s: torch.randn(*s, device=dev, generator=g).to(dt)
        x, d, w = r(T, H), r(T, H), r(H) * 0.1
        ok &= check("rms_norm", pw.rms_norm(x, w, 1e-6), _rms(x, w, 1e-6), 4)
        ok &= check("add_rms_norm", pw.rms_norm(x, w, 1e-6, residual=d), _add_rms(x, d, w, 1e-6), 4)
        gu = r(T, 2 * I)
        ok &= check("swiglu", pw.swiglu(gu, I), _swiglu(gu, I), 4)
        nv, dv = 16, 128
        o, zbig, wn = r(T, nv, dv), r(T, 3 * nv * dv), r(dv)
        z = zbig[:, nv * dv: 2 * nv * dv]
        ok &= check("gated_rms", pw.gated_rms(o, z, wn, 1e-6), _gated_rms(o, z, wn, 1e-6), 4)
        nh, nkv, hd, rot = 8, 2, 256, 64
        p = r(T, nh * 2 * hd + 2 * nkv * hd)
        qg = p[:, : nh * 2 * hd].view(T, nh, 2 * hd)
        q, gt = qg[..., :hd], qg[..., hd:]
        k = p[:, nh * 2 * hd: nh * 2 * hd + nkv * hd].view(T, nkv, hd)
        ang = torch.rand(T, rot // 2, device=dev, generator=g) * 6.28
        cos = torch.cat([ang.cos(), ang.cos()], -1).to(dt)
        sin = torch.cat([ang.sin(), ang.sin()], -1).to(dt)
        wq, wk = r(hd) * 0.1, r(hd) * 0.1
        ok &= check("qk_norm_rope", pw.qk_norm_rope(q, k, cos, sin, wq, wk, 1e-6),
                    _qk_norm_rope(q, k, cos, sin, wq, wk, 1e-6), 4)
        oa = r(T, nh, hd)
        ok &= check("gate", pw.gate(oa, gt), _gate(oa, gt), 4)
    print("->", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
