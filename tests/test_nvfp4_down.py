"""Down scheduling and pipelining retain the original GEMM's FP32 output bits."""
import torch
from flide.kernels import nvfp4 as nv
from flide.kernels.moe import align


@torch.inference_mode()
def main():
    torch.manual_seed(91)
    assert nv.down_tuning(2048, 512, 16384, 64) == (256, 3, 16, '.cs')
    assert nv.down_tuning(2048, 512, 16383, 64) == (128, 3, 0, '')
    assert nv.down_tuning(2048, 256, 16384, 64) == (128, 3, 0, '')
    for t, e, topk, n, k in [(1, 4, 2, 160, 160), (67, 17, 4, 320, 512),
                            (2051, 17, 8, 2048, 512)]:
        pairs = t * topk
        ids = torch.randint(0, max(e // 2, 1), (t, topk), device='cuda', dtype=torch.int32)
        top = torch.softmax(torch.randn(t, topk, device='cuda'), -1).flatten()
        x = torch.randn(pairs, k, device='cuda', dtype=torch.bfloat16)
        x *= torch.pow(2., torch.randint(-8, 9, (pairs, 1), device='cuda')).bfloat16()
        x[0] = 0
        g = torch.tensor(.0027, device='cuda')
        a, sa = nv.quantize(x, g)
        w = torch.randint(0, 256, (e, n, k // 2), device='cuda', dtype=torch.uint8)
        scales = torch.pow(2., torch.randint(-8, 9, (e, n, k // 16), device='cuda'))
        sw = (scales * (torch.rand_like(scales) + .125)).to(torch.float8_e4m3fn)
        global_scale = torch.rand(e, device='cuda') * .02
        routing = align(ids, e, 64)
        old = torch.empty(pairs, n, device='cuda', dtype=torch.float32)
        out = torch.empty_like(old)
        nv.grouped(a, sa, w, sw, global_scale, g, old, routing, topk, False,
                   top=top, tile=(64, 256, 8), tuning=(128, 3, 0, ''))
        for gm in (1, 4, 16, 64):
            nv.grouped(a, sa, w, sw, global_scale, g, out, routing, topk, False,
                       top=top, tile=(64, 256, 8), tuning=(256, 3, gm, '.cs'))
            assert torch.equal(old.view(torch.uint8), out.view(torch.uint8)), (t, n, k, gm)
        print('PASS Down bits, exponent variation, K/N tails, unused experts and final group', t, n, k)
    print('PASS Down tuning dispatch and grouped schedules')


if __name__ == '__main__':
    main()
