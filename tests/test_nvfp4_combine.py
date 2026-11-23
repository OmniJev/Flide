"""NVFP4 sum/combine fusion preserves PyTorch reduction and all BF16 boundaries."""
import torch
from flide.kernels import moe


def bits_equal(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (a.shape, int((a != b).sum()))


def check(t, n, strided=False):
    width = n * 2 if strided else n
    y = torch.randn(t * 8, width, device='cuda', dtype=torch.float32)[:, :n]
    shared = torch.randn(t, width, device='cuda', dtype=torch.bfloat16)[:, :n]
    gate = torch.linspace(-100, 100, t, device='cuda')[:, None]
    residual = torch.randn(t, width, device='cuda', dtype=torch.bfloat16)[:, :n]
    weight = torch.randn(n, device='cuda', dtype=torch.float32) * .1

    def reference():
        return y.reshape(t, 8, n).sum(dim=1).bfloat16()

    def verify():
        bits_equal(moe.combine(y, shared, gate, 8, round_experts=True),
                   moe.combine(reference(), shared, gate, 1))
        for offset in (False, True):
            actual = moe.combine_add_rms(y, shared, gate, 8, residual, weight, 1e-6,
                                          offset=offset, round_experts=True)
            expected = moe.combine_add_rms(reference(), shared, gate, 1, residual, weight, 1e-6,
                                            offset=offset)
            for a, b in zip(actual, expected):
                bits_equal(a, b)

    if t:
        # Four-accumulator order differs from a sequential or tree top-8 sum.
        v = y.reshape(t, 8, n)
        v[0, 0] = 1e20
        v[0, 1] = 1
        v[0, 2:4] = 0
        v[0, 4] = -1e20
        v[0, 5:] = 0
        if t > 1:
            v[1] = -0.0
        if t > 2:
            v[2].mul_(torch.pow(2., torch.arange(8, device='cuda')[:, None] * 4 - 16))
    verify()
    if t:
        # BF16 underflow must retain the following combine's +0 initialization.
        y.fill_(-1e-42)
        shared.fill_(-0.0)
        residual.fill_(-0.0)
        verify()
        y.normal_()
        shared.normal_()
        residual.normal_()
        for _ in range(2):
            moe.combine_add_rms(y, shared, gate, 8, residual, weight, 1e-6, round_experts=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            combined = moe.combine(y, shared, gate, 8, round_experts=True)
            actual = moe.combine_add_rms(y, shared, gate, 8, residual, weight, 1e-6, round_experts=True)
        y.normal_()
        shared.normal_()
        gate.normal_()
        residual.normal_()
        weight.normal_()
        graph.replay()
        torch.cuda.synchronize()
        bits_equal(combined, moe.combine(reference(), shared, gate, 1))
        expected = moe.combine_add_rms(reference(), shared, gate, 1, residual, weight, 1e-6)
        for a, b in zip(actual, expected):
            bits_equal(a, b)
    print('PASS fused NVFP4 combine bits, rounding, offsets and graph replay', t, n, strided)


@torch.inference_mode()
def main():
    torch.manual_seed(73)
    for t, n in [(0, 2048), (1, 4), (1, 2048), (137, 128), (137, 768),
                 (137, 2048), (137, 4096), (2051, 2048)]:
        check(t, n)
    check(137, 2048, strided=True)
    y = torch.zeros(8, 128, device='cuda', dtype=torch.float16)
    try:
        moe.combine(y, y[:1], torch.zeros(1, 1, device='cuda'), 8, round_experts=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted an unsupported fused expert dtype')


if __name__ == '__main__':
    main()
