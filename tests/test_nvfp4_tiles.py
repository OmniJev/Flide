"""Expert tile changes preserve bits across routing, tails and graph replay."""
import torch

from flide.kernels import nvfp4 as nv
from flide.kernels.moe import align
from flide.kernels.pointwise import swiglu
from test_nvfp4 import decode


def bits_equal(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), float((a.float() - b.float()).abs().max())


def packed_weight(e, n, k):
    q = torch.randint(0, 256, (e, n, k // 2), device='cuda', dtype=torch.uint8)
    scale = (torch.rand(e, n, k // 16, device='cuda') + .01).to(torch.float8_e4m3fn)
    return q, scale


@torch.inference_mode()
def main():
    torch.manual_seed(73)
    assert nv.expert_tiles(16383) == ((32, 64, 4), (32, 64, 4))
    assert nv.expert_tiles(16384) == ((64, 128, 4), (64, 256, 8))
    for t, e, topk, h, i in [(1, 8, 2, 160, 160), (67, 16, 4, 256, 160),
                             (2051, 16, 8, 2048, 512)]:
        x = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
        if t > 1:
            x[0] = 0
        # Skewed routing leaves half the experts unused and exercises block padding.
        ids = torch.randint(0, e // 2, (t, topk), device='cuda', dtype=torch.int32)
        ids[:, 0] = 0
        top = torch.softmax(torch.randn(t, topk, device='cuda'), -1)
        if t > 1:
            top[0] = 0
        scale = torch.tensor(.0027, device='cuda')
        aq, sa = nv.quantize(x, scale)
        pair_x = torch.randn(t * topk, i, device='cuda', dtype=torch.bfloat16)
        pq, ps = nv.quantize(pair_x, scale)
        old_routing, new_routing = align(ids, e, 32), align(ids, e, 64)
        for by_token, n, k, a, s, dtype, tile in [
                (True, 2 * i, h, aq, sa, torch.bfloat16, (64, 128, 4)),
                (False, h, i, pq, ps, torch.float32, (64, 256, 8))]:
            w, ws = packed_weight(e, n, k)
            g = torch.rand(e, 2, device='cuda') * .02 if by_token else torch.rand(e, device='cuda') * .02
            old = torch.empty(t * topk, n, device='cuda', dtype=dtype)
            new = torch.empty_like(old)
            kwargs = dict(halves=by_token, top=None if by_token else top.flatten())
            nv.grouped(a, s, w, ws, g, scale, old, old_routing, topk, by_token, **kwargs)
            nv.grouped(a, s, w, ws, g, scale, new, new_routing, topk, by_token, tile=tile, **kwargs)
            bits_equal(old, new)
            # Independent decoded FP64 reference includes an N tail and K tail.
            if t == 1:
                da = decode(a, s)
                refs = []
                for slot in range(topk):
                    expert = int(ids[0, slot])
                    alpha = g[expert].repeat_interleave(n // 2) if by_token else g[expert]
                    value = da[0 if by_token else slot] @ decode(w[expert], ws[expert]).T
                    value *= alpha.double() * scale.double()
                    if not by_token:
                        value *= top[0, slot].double()
                    refs.append(value)
                torch.testing.assert_close(new, torch.stack(refs).to(dtype), atol=.004 if by_token else 1e-6,
                                           rtol=.008 if by_token else 2e-5)
        print('PASS GEMM tile bits, skewed/empty experts, padding and zero rows', t, h, i)

    # End-to-end experts: the large-prefill threshold takes tuned tiles in the graph.
    t, e, topk, h, i = 2051, 16, 8, 2048, 512
    w13, s13 = packed_weight(e, 2 * i, h)
    w2, s2 = packed_weight(e, h, i)
    experts = nv.NvFp4Experts(w13, w2, s13, s2, torch.rand(e, 2, device='cuda') * .002,
                             torch.rand(e, device='cuda') * .002, scale, scale)
    x = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
    ids = torch.randint(e, (t, topk), device='cuda', dtype=torch.int32)
    top = torch.softmax(torch.randn(t, topk, device='cuda'), -1)
    def reference(x, top, ids):
        # Materialize BF16 Gate/Up with the original tiles to cover both tile selection and fusion.
        pairs = x.shape[0] * topk
        tile = (32, 64, 4)
        routing = align(ids, e, tile[0])
        aq, sa = nv.quantize(x, scale)
        gu = nv.grouped(aq, sa, w13, s13, experts.g13, scale, x.new_empty((pairs, 2 * i)),
                        routing, topk, True, halves=True, tile=tile)
        q, s = nv.swiglu_quantize(gu, i, scale)
        return nv.grouped(q, s, w2, s2, experts.g2, scale, x.new_empty((pairs, h), dtype=torch.float32),
                          routing, topk, False, top=top.flatten(), tile=tile)

    bits_equal(experts(x[:37], top[:37], ids[:37], swiglu), reference(x[:37], top[:37], ids[:37]))
    expected = reference(x, top, ids)
    actual = experts(x, top, ids, swiglu)
    bits_equal(actual, expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = experts(x, top, ids, swiglu)
    graph.replay()
    torch.cuda.synchronize()
    bits_equal(captured, expected)
    x.normal_()
    ids.copy_(torch.randint(e // 2, ids.shape, device='cuda', dtype=torch.int32))
    top.copy_(torch.softmax(torch.randn_like(top), -1))
    expected = reference(x, top, ids)
    graph.replay()
    torch.cuda.synchronize()
    bits_equal(captured, expected)
    print('PASS full experts byte equality, tuned CUDA graph and replay with changed inputs/routing')


if __name__ == '__main__':
    main()
