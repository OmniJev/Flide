"""Verify BF16 Down storage on both sides of the routing-weight multiply."""
import torch
from flide.kernels import nvfp4 as nv, moe
from flide.kernels.pointwise import swiglu


def bits_equal(a, b):
    assert a.dtype == b.dtype
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (a.shape, int((a != b).sum()))


@torch.inference_mode()
def main():
    torch.manual_seed(97)
    for t, h, i in [(0, 256, 160), (1, 256, 160), (137, 256, 160), (2051, 2048, 512)]:
        e, k = 16, 8
        w13 = torch.randint(0, 256, (e, 2 * i, h // 2), device='cuda', dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, h, i // 2), device='cuda', dtype=torch.uint8)
        s13 = (torch.rand(e, 2 * i, h // 16, device='cuda') + .01).to(torch.float8_e4m3fn)
        s2 = (torch.rand(e, h, i // 16, device='cuda') + .01).to(torch.float8_e4m3fn)
        args = (w13, w2, s13, s2, torch.rand(e, 2, device='cuda') * .002,
                torch.rand(e, device='cuda') * .002, torch.tensor(.0027, device='cuda'),
                torch.tensor(.0027, device='cuda'))
        reference = nv.NvFp4Experts(*args, down_dtype=torch.float32)
        candidate = nv.NvFp4Experts(*args, down_dtype=torch.bfloat16)
        x = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
        ids = torch.randint(0, e // 2, (t, k), device='cuda', dtype=torch.int32)
        top = torch.softmax(torch.randn(t, k, device='cuda'), -1)
        shared = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
        gate = torch.randn(t, 1, device='cuda')
        residual = torch.randn_like(shared)
        weight = torch.randn(h, device='cuda', dtype=torch.float32) * .1

        def expected_output():
            return reference(x, top, ids, swiglu).bfloat16()

        def check():
            out = candidate(x, top, ids, swiglu)
            bits_equal(out, expected_output())
            summed = out.float().reshape(t, k, h).sum(dim=1).bfloat16()
            bits_equal(moe.combine(out, shared, gate, k, round_experts=True),
                       moe.combine(summed, shared, gate, 1))
            actual = moe.combine_add_rms(out, shared, gate, k, residual, weight, 1e-6, round_experts=True)
            expected = moe.combine_add_rms(summed, shared, gate, 1, residual, weight, 1e-6)
            for a, b in zip(actual, expected):
                bits_equal(a, b)

        check()
        raw = candidate(x, None, ids, swiglu)
        bits_equal(raw, reference(x, torch.ones_like(top), ids, swiglu).bfloat16())
        routed = torch.zeros_like(x, dtype=torch.float32)
        for slot in range(k):
            routed = torch.addcmul(routed, top[:, slot, None], raw[slot::k].float())
        bits_equal(moe.combine(raw, shared, gate, k, round_experts=True, top=top),
                   moe.combine(routed.bfloat16(), shared, gate, 1))
        if t:
            for _ in range(2):
                candidate(x, top, ids, swiglu)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = candidate(x, top, ids, swiglu)
                combined = moe.combine_add_rms(out, shared, gate, k, residual, weight, 1e-6,
                                               round_experts=True)
            x.normal_()
            ids.copy_(torch.randint(0, e, ids.shape, device='cuda', dtype=torch.int32))
            top.copy_(torch.softmax(torch.randn_like(top), -1))
            shared.normal_()
            gate.normal_()
            residual.normal_()
            weight.normal_()
            graph.replay()
            torch.cuda.synchronize()
            bits_equal(out, expected_output())
            summed = expected_output().float().reshape(t, k, h).sum(dim=1).bfloat16()
            expected = moe.combine_add_rms(summed, shared, gate, 1, residual, weight, 1e-6)
            for a, b in zip(combined, expected):
                bits_equal(a, b)
        print('PASS BF16 Down storage rounding, FP32 reduction and graph replay', t, h, i)


if __name__ == '__main__':
    main()
