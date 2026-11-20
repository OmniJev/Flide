"""Unified optional backend: dependency failures, fallbacks and session-context inputs."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from flide import settings
from flide.formats.base import Read
from flide.runtime.pack import Job, pack
from flide.kernels import attention as attn
from flide.kernels.attention_backends import configure, prepare_attention, optimized_attention, _cute


@torch.inference_mode()
def main():
    saved, invariant = settings.attention_backend, settings.batch_invariant
    try:
        with patch('flide.kernels.cute_attention.load', side_effect=ImportError('missing optional runtime')):
            configure('auto')
            assert _cute() is None
            try:
                configure('cute')
            except RuntimeError as exc:
                assert 'dependencies are unavailable' in str(exc)
            else:
                raise AssertionError('explicit cute must fail without dependencies')
        print('PASS optional dependency: auto fallback and explicit failure', flush=True)
        configure('auto')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0) or _cute() is None:
            print('SKIP GPU checks: requires SM120 and optional CuTe runtime', flush=True)
            return
        configure('cute'); settings.batch_invariant = False
        torch.manual_seed(101)
        # Split session save-point levels use context KV addresses, not fresh-stream offsets.
        b = pack([Job([1]*4200, [[2]*7, [3]*17], [(Read(None),)]*2, marks=(4096,))
                  for _ in range(2)], torch.device('cuda'))
        assert b.ctx is not None and len(b.levels) > 1
        t, capacity = b.n_tokens, b.ctx['len']
        r = b.key_ranges.T.contiguous()
        q = torch.randn(t, 4, 256, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(t, 2, 256, device='cuda', dtype=q.dtype)
        v = torch.randn(t, 4, 256, device='cuda', dtype=q.dtype)[:, 2:]
        kc = torch.randn(capacity, 2, 256, device='cuda', dtype=q.dtype)
        vc = torch.randn_like(kc)
        kc.index_copy_(0, b.ctx['index'], k[:b.n_prefix_tokens])
        vc.index_copy_(0, b.ctx['index'], v[:b.n_prefix_tokens])
        tok = tuple(x.int() for x in (b.job, b.branch, b.positions))
        context = tuple(x.int() for x in b.ctx['meta'])
        cfg = attn.config(q.device, q.dtype, 256)
        def metadata():
            tiles = attn.block_ranges(*r, cfg['BLOCK_M'], cfg['BLOCK_N'], exact=True, groups=b.attn_runs)
            return tiles, prepare_attention(r, tiles, dtype=q.dtype, head_dim=256, kv_capacity=capacity)
        def run(cute, out=None):
            tiles, meta = metadata()
            return attn.attention(q, kc, vc, k, v, tok, context, tok, tiles, 1/16,
                                  cfg=cfg, qranges=r, out=out, backend_metadata=meta if cute else None)
        tiles, meta = metadata()
        assert meta.covered.sum().item() == t
        reference = run(False); target = torch.empty_like(q)
        assert run(True, target) is target
        torch.testing.assert_close(target, reference, atol=.02, rtol=.02)
        print('PASS split session levels, mapped context KV, strided V and caller output', flush=True)
        for _ in range(2): run(True)
        torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured = run(True)
        # Resumed prefixes may include earlier cached keys; the exact ranges remain authoritative.
        r[0].sub_(32).clamp_min_(0)
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(captured, run(False), atol=.02, rtol=.02)
        # Changing values in the actual context buffer is visible on graph replay.
        kc.add_(.125); vc.mul_(.75)
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(captured, run(False), atol=.02, rtol=.02)
        print('PASS resumed ranges and changing context contents on graph replay', flush=True)
        for dtype, dim in [(torch.float32, 256), (torch.bfloat16, 96)]:
            assert prepare_attention(r, tiles, dtype=dtype, head_dim=dim) is None
        assert prepare_attention(r, None, dtype=q.dtype, head_dim=256) is None
        assert optimized_attention(q, kc, vc, k, v, 1/16, 0, None) is None
        assert optimized_attention(q[:5], kc, vc, k, v, 1/16, 0, meta) is None
        assert optimized_attention(q, kc, vc, k, v, 1/16, 64, meta) is None
        assert optimized_attention(q, kc[:1], vc[:1], k, v, 1/16, 0, meta) is None
        settings.batch_invariant = True
        assert prepare_attention(r, tiles, dtype=q.dtype, head_dim=256) is None
        settings.batch_invariant = False
        configure('triton')
        with patch('flide.kernels.cute_attention.load', side_effect=AssertionError('must not load CuTe')):
            assert prepare_attention(r, tiles, dtype=q.dtype, head_dim=256) is None
            assert optimized_attention(q, kc, vc, k, v, 1/16, 0, meta) is None
        configure('cute')
        print('PASS dtype/dimension, subset, window, capacity, invariant and Triton-only fallbacks', flush=True)
        invalid = r.clone(); invalid[1, 0] = capacity + 1
        invalid_meta = prepare_attention(invalid, tiles, dtype=q.dtype, head_dim=256, kv_capacity=capacity)
        assert invalid_meta.covered[0].item() == 0
        print('PASS GPU validation rejects out-of-capacity coverage', flush=True)
    finally:
        settings.attention_backend, settings.batch_invariant = saved, invariant


if __name__ == '__main__': main()
