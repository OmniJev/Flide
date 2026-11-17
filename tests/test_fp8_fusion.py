"""CuTe FP8 producer fusion, prequantized grouped products, and safe fallbacks."""
import sys
import os
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import torch
from flide import settings
from flide.kernels import fp8_fusion as fusion
from flide.kernels import moe
from flide.kernels.fp8 import quantize_tokens
from flide.kernels.pointwise import swiglu

@torch.inference_mode()
def main():
    old, invariant = settings.fp8_fusion, settings.batch_invariant
    try:
        env = dict(os.environ)
        env.pop('FLIDE_FP8_FUSION', None)
        default = subprocess.check_output(
            [sys.executable, '-c', 'from flide import settings; print(settings.fp8_fusion)'],
            env=env, cwd=Path(__file__).resolve().parents[1], text=True).strip()
        assert default == 'auto'
        settings.fp8_fusion = 'auto'
        with patch.object(fusion, '_load', return_value=(None, ImportError('missing runtime'))):
            assert fusion._backend() is None
            settings.fp8_fusion = 'swiglu'
            try:
                fusion._backend()
            except RuntimeError as exc:
                assert 'dependencies are unavailable' in str(exc)
            else:
                raise AssertionError('explicit selection must report missing dependencies')
        settings.fp8_fusion = 'auto'
        print('PASS default auto and missing dependency fallback')
        assert fusion.swiglu_quant(torch.empty(2, 1024), 512) is None
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            print('SKIP GPU requires SM120');return
        torch.manual_seed(721)
        gu = torch.randn(137, 1024, device='cuda', dtype=torch.bfloat16)
        q, s = fusion.swiglu_quant(gu, 512)
        settings.fp8_fusion = 'swiglu'
        explicit_q, explicit_s = fusion.swiglu_quant(gu, 512)
        assert torch.equal(q.view(torch.uint8), explicit_q.view(torch.uint8)) and torch.equal(s,explicit_s)
        settings.fp8_fusion = 'auto'
        rq, rs = quantize_tokens(swiglu(gu, 512))
        assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8)) and torch.equal(s, rs)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = fusion.swiglu_quant(gu, 512)
        gu.normal_();g.replay();torch.cuda.synchronize()
        rq, rs = quantize_tokens(swiglu(gu, 512))
        assert torch.equal(out[0].view(torch.uint8), rq.view(torch.uint8)) and torch.equal(out[1],rs)
        print('PASS SwiGLU and graph replay')
        # Real-model rounding boundaries, reduced to synthetic inputs.
        edge = torch.zeros(3,1024,device='cuda',dtype=torch.bfloat16)
        edge[:,0] = 16
        edge[:,512] = torch.tensor([.5,.2158203125,.462890625],device='cuda') / 16
        edge[:,1] = .0004444122314453125
        edge[:,513] = torch.tensor([.12255859375,.2451171875,-.490234375],device='cuda')
        eq, es = fusion.swiglu_quant(edge,512)
        rq, rs = quantize_tokens(swiglu(edge,512))
        assert torch.equal(eq.view(torch.uint8),rq.view(torch.uint8)) and torch.equal(es,rs)
        print('PASS BF16/FP8 halfway-point regression')
        assert fusion.swiglu_quant(gu[:,:510],255) is None
        assert fusion.swiglu_quant(gu,256) is None
        assert fusion._threads_per_row(0) is None
        assert fusion._threads_per_row(32800) is None
        assert [fusion._threads_per_row(n) for n in (512,544,4096,4128,16384,16416)] == [32,128,128,256,256,512]
        # Includes each thread-layout transition and masked tails. Padded storage
        # and changing row counts exercise reuse of width-specialized kernels.
        for n in (32, 96, 256, 512, 544, 768, 1024, 1056, 1536, 2048, 4096,
                  4128, 8192, 14336, 16384, 16416, 28672, 32768):
            for rows in (1, 5, 137):
                storage = torch.randn(rows, 2*n+16, device='cuda', dtype=torch.bfloat16)
                x = storage[:, :2*n]
                x[0].zero_()
                if rows > 1:
                    x[1].mul_(100)
                    x[2].mul_(1e-6)
                    x[3].zero_()
                    x[3,0], x[3,n] = 16, .5/16
                    x[3,1], x[3,n+1] = .0004444122314453125, .12255859375
                aq, asc = fusion.swiglu_quant(x,n)
                rq, rs = quantize_tokens(swiglu(x,n))
                assert torch.equal(aq.view(torch.uint8),rq.view(torch.uint8)), (n,rows,'fp8')
                assert torch.equal(asc,rs), (n,rows,'scale')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): captured = fusion.swiglu_quant(x,n)
            x.normal_()
            graph.replay()
            rq, rs = quantize_tokens(swiglu(x,n))
            assert torch.equal(captured[0].view(torch.uint8),rq.view(torch.uint8)), (n,'graph')
            assert torch.equal(captured[1],rs), (n,'graph scale')
            print(f'PASS width {n}: rows 1/5/137, padded stride, edges, graph replay', flush=True)
        assert fusion.swiglu_quant(gu.float(),512) is None
        settings.batch_invariant = True
        assert fusion.swiglu_quant(gu,512) is None
        settings.batch_invariant = False
        # Same grouped matrix product, using a prequantized input must skip quantization.
        a=torch.randn(137,512,device='cuda',dtype=torch.bfloat16)
        weights=torch.randn(2,64,512,device='cuda',dtype=a.dtype)*.02
        qw,sw=moe.quantize_experts(weights)
        ids=torch.zeros(137,1,device='cuda',dtype=torch.int32)
        tile=moe.tile(137,True,a.device)
        rows,blocks,used=moe.align(ids,2,tile[0])
        ref=moe.grouped_fp8(a,qw,sw,a.new_empty(137,64),rows,blocks,used,1,True,tile=tile)
        aq=quantize_tokens(a)
        with patch('flide.kernels.fp8.quantize_tokens', side_effect=AssertionError('requantized')):
            out=moe.grouped_fp8(None,qw,sw,a.new_empty(137,64),rows,blocks,used,1,True,tile=tile,quantized=aq)
        assert torch.equal(out,ref)
        with patch.object(fusion, '_load', return_value=(None, ImportError('missing runtime'))):
            assert fusion.swiglu_quant(gu,512) is None
        def failed_launch(*args, **kwargs):
            raise RuntimeError('simulated CUDA failure')
        with patch.object(fusion, '_load', return_value=(failed_launch, None)):
            try:
                fusion.swiglu_quant(gu,512)
            except RuntimeError as exc:
                assert str(exc) == 'simulated CUDA failure'
            else:
                raise AssertionError('execution errors must propagate')
        settings.fp8_fusion='off'
        with patch.object(fusion, '_load', side_effect=AssertionError('loaded while disabled')):
            assert fusion.swiglu_quant(gu,512) is None
        print('PASS prequantized GEMM, dtype/shape/invariant and disabled fallbacks')
    finally:settings.fp8_fusion,settings.batch_invariant=old,invariant
if __name__=='__main__':main()
