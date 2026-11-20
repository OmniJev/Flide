"""Segmented GDN normalization keeps bits across boundaries and CUDA graph replay."""
import sys
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flide.kernels import pointwise as pw


def main():
    torch.manual_seed(117)
    checks = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for dim in (64, 128, 192, 256):
            for lengths in ((37, 18, 109), (16384, 17, 0)):
                heads = 3
                parts = [torch.randn(n, heads * 2, dim, device='cuda', dtype=dtype)[:, ::2]
                         for n in lengths]
                t = sum(lengths)
                z = torch.randn(t, heads * dim * 3, device='cuda', dtype=dtype)[:, heads * dim:2 * heads * dim]
                w = torch.randn(dim, device='cuda', dtype=dtype)
                reference = pw.gated_rms(torch.cat(parts, 0), z, w, 1e-6)
                got = pw.gated_rms_segments(parts, z, w, 1e-6)
                assert torch.equal(got, reference), (dtype, dim, lengths)
                # Both short and long segments avoid materializing a concatenated input.
                with patch.object(torch, 'cat', side_effect=AssertionError('unexpected concatenation')):
                    assert torch.equal(pw.gated_rms_segments(parts, z, w, 1e-6), reference)
                for _ in range(3):
                    pw.gated_rms_segments(parts, z, w, 1e-6)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = pw.gated_rms_segments(parts, z, w, 1e-6)
                for p in parts:
                    p.normal_()
                z.normal_()
                w.normal_()
                graph.replay()
                assert torch.equal(out, pw.gated_rms(torch.cat(parts, 0), z, w, 1e-6))
                checks += 1
    for lengths in ((0, 0), (0, 1, 17, 0), (257,), (1,) * 16, (0, 3, 0, 17) * 8):
        ps = [torch.randn(n, 2, 128, device='cuda', dtype=torch.bfloat16) for n in lengths]
        z = torch.randn(sum(lengths), 256, device='cuda', dtype=torch.bfloat16)
        w = torch.ones(128, device='cuda', dtype=torch.bfloat16)
        assert torch.equal(pw.gated_rms_segments(ps, z, w, 1e-6), pw.gated_rms(torch.cat(ps), z, w, 1e-6))
        checks += 1
    p = torch.randn(19, 2, 128, device='cuda', dtype=torch.bfloat16)
    z = torch.randn(19, 256, device='cuda', dtype=torch.bfloat16)
    storage = torch.full((21, 256), 17, device='cuda', dtype=torch.bfloat16)
    got = pw.gated_rms(p, z, w, 1e-6, out=storage[1:-1])
    assert torch.equal(got, pw.gated_rms(p, z, w, 1e-6))
    assert bool((storage[[0, -1]] == 17).all())
    for parts, gate in (([], z), ([p], z[:-1]), ([p, p[:, :1]], torch.cat([z, z]))):
        try:
            pw.gated_rms_segments(parts, gate, w, 1e-6)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid segments accepted')
    print(f'PASS: {checks} segment cases, changed-input graphs, output boundaries and invalid shapes')


if __name__ == '__main__':
    main()
