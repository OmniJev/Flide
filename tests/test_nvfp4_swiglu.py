"""Fusion must preserve the separate BF16 SwiGLU and NVFP4 bytes, including graph replay."""
import torch

from flide.kernels.nvfp4 import quantize, swiglu_quantize
from flide.kernels.pointwise import swiglu


def check(gu, n, global_scale):
    expected_q, expected_s = quantize(swiglu(gu, n), global_scale)
    actual_q, actual_s = swiglu_quantize(gu, n, global_scale)
    assert torch.equal(actual_q, expected_q), (gu.shape, 'packed FP4', (actual_q != expected_q).sum().item())
    assert torch.equal(actual_s.view(torch.uint8), expected_s.view(torch.uint8)), (gu.shape, 'E4M3 scales')


@torch.inference_mode()
def main():
    torch.manual_seed(29)
    for rows, n, padding in [(0, 512, 0), (1, 16, 0), (3, 32, 0), (17, 320, 0),
                             (129, 512, 0), (4097, 512, 0), (19, 1024, 32)]:
        gu = torch.randn(rows, 2 * n + padding, dtype=torch.bfloat16, device='cuda')[:, :2 * n]
        if rows:
            gu[0, :n] = -0.0
            gu[0, n:] = 0.0
        if rows > 1:
            values = torch.tensor([-100., -90., -10., -1., -0., 0., .5, 1., 10., 20., 100.],
                                  device='cuda', dtype=torch.bfloat16)
            gu[1, :n] = values.repeat((n + len(values)-1)//len(values))[:n]
            gu[1, n:] = gu[1, :n].roll(5) * 8
        for scale in [.007, .0027, 1e-5]:
            check(gu, n, torch.tensor(scale, device='cuda'))
    print('PASS byte-identical FP4 packing and E4M3 scales: zero/sign, saturation, partial tiles and row strides')
    gu = torch.randn(33, 1024, dtype=torch.bfloat16, device='cuda')
    global_scale = torch.tensor(.0027, device='cuda')
    swiglu_quantize(gu, 512, global_scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = swiglu_quantize(gu, 512, global_scale)
    for factor in [1., .75]:
        gu.mul_(factor)
        expected_q, expected_s = quantize(swiglu(gu, 512), global_scale)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(q, expected_q)
        assert torch.equal(s.view(torch.uint8), expected_s.view(torch.uint8))
    print('PASS fused CUDA graph replay consumes changed inputs with identical bytes')


if __name__ == '__main__':
    main()
