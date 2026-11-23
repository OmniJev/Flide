"""Check native NVFP4 packing and routed GEMMs against independent decoded references."""
import torch

from flide.kernels.nvfp4 import quantize, grouped
from flide.kernels.moe import align


def decode(q, s):
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64, device=q.device)
    codes = torch.stack([q & 15, q >> 4], -1).flatten(-2).long()
    return values[codes & 7] * torch.where((codes & 8) != 0, -1., 1.) * s.double().repeat_interleave(16, -1)


def reference_quantize(x, global_scale):
    blocks = x.float().reshape(-1, 16)
    scales = (blocks.abs().amax(-1) / (6 * global_scale)).clamp(max=448).to(torch.float8_e4m3fn)
    scale = scales.float() * global_scale
    normalized = blocks * torch.where(scale > 0, 1.0 / scale, 0.)[:, None]
    bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device)
    codes = (normalized.abs()[..., None] > bounds).sum(-1)
    codes += ((normalized.abs()[..., None] == bounds[[1, 3, 5]]).any(-1)).long()
    codes |= torch.signbit(normalized).long() << 3
    packed = (codes[:, 0::2] | codes[:, 1::2] << 4).byte()
    return packed.reshape(x.shape[0], -1), scales.reshape(x.shape[0], -1)


def main():
    torch.manual_seed(42)
    for scale in [.007, .0027, 1., 1e-5]:
        x = torch.randn(17, 256, dtype=torch.bfloat16, device='cuda')
        x[0] = 0
        q, s = quantize(x, torch.tensor(scale, device='cuda'))
        rq, rs = reference_quantize(x, scale)
        assert torch.equal(s.view(torch.uint8), rs.view(torch.uint8)), scale
        assert torch.equal(q, rq), (scale, (q != rq).sum().item())
    print('PASS activation packing, E4M3 scales, zero blocks and saturation')
    e, n, k, t, topk = 4, 128, 256, 17, 2
    w = torch.randint(0, 256, (e, n, k // 2), device='cuda', dtype=torch.uint8)
    ws = (torch.rand(e, n, k // 16, device='cuda') + .1).to(torch.float8_e4m3fn)
    wg = torch.rand(e, 2, device='cuda') * .02
    input_scale = torch.tensor(.007, device='cuda')
    x = torch.randn(t, k, device='cuda', dtype=torch.bfloat16)
    aq, sa = quantize(x, input_scale)
    ids = torch.stack([torch.randperm(e, device='cuda')[:topk] for _ in range(t)]).int()
    top = torch.softmax(torch.randn(t, topk, device='cuda'), -1)
    routing = align(ids, e, bm=32)
    out = torch.empty(t * topk, n, device='cuda', dtype=torch.bfloat16)
    grouped(aq, sa, w, ws, wg, input_scale, out, routing, topk, True, halves=True)
    dx = decode(aq, sa)
    ref = torch.stack([dx[row] @ decode(w[int(ids[row, slot])], ws[int(ids[row, slot])]).T *
                       wg[int(ids[row, slot])].repeat_interleave(n // 2).double() * input_scale.double()
                       for row in range(t) for slot in range(topk)]).bfloat16()
    torch.testing.assert_close(out, ref, atol=.004, rtol=.008)
    # Second GEMM reads one activation per routed pair and applies each pair's weight once.
    pair_x = torch.randn(t * topk, k, device='cuda', dtype=torch.bfloat16)
    pq, ps = quantize(pair_x, input_scale)
    out2 = torch.empty(t * topk, n, device='cuda', dtype=torch.float32)
    grouped(pq, ps, w, ws, wg[:, 0].contiguous(), input_scale, out2, routing, topk, False, top=top.flatten())
    dp = decode(pq, ps)
    ref2 = torch.stack([dp[row * topk + slot] @ decode(w[int(ids[row, slot])], ws[int(ids[row, slot])]).T *
                        wg[int(ids[row, slot]), 0].double() * input_scale.double() * top[row, slot].double()
                        for row in range(t) for slot in range(topk)]).float()
    torch.testing.assert_close(out2, ref2, atol=1e-6, rtol=2e-5)
    print('PASS grouped native FP4 GEMMs, split gate/up scales, routing and final weighting')


if __name__ == '__main__':
    main()
