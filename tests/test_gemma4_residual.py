"""The gemma4 backbone's `float32_residual` and `graph_step` options on a tiny random dense Gemma 4 in bfloat16.

Reference: the same bfloat16 weights computed in float32 (TF32 off) through the packed forward. Pass: with the residual stream in float32
the final-norm hidden states sit no farther from the float32 run than with it in bfloat16 (mean over branches), the
option off gives the default decoder's bits, the stream's dtype never reaches the outputs (bfloat16 either way), CUDA
graphs on the finer ladder replay what the packed forward computes, and the ladder keeps every length of the default
one from 1024 on.

    PYTHONPATH=. python tests/test_gemma4_residual.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import copy
import random
import sys

import torch

from flide.backbones.gemma4 import Gemma4
from flide.backbones.gemma4.graphs import LADDER, fine_ladder
from flide.formats.base import FINAL, Read
from flide.runtime.pack import Job, pack


def tiny(device):
    from transformers import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, global_head_dim=128, num_global_key_value_heads=1,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 2,
        sliding_window=32, attention_k_eq_v=True, final_logit_softcapping=30.0, tie_word_embeddings=True,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0,
        torch_dtype="float32")
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(1.0, 0.1)
            elif p.dim() >= 2:
                p.normal_(0.0, 0.08)
    return model


def finals(bb, jobs, device):
    b = pack([Job(j.prefix, j.suffixes, [(Read(None),)] * len(j.suffixes)) for j in jobs], device)
    return bb.forward(b)[FINAL]


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    m16 = tiny(device).to(torch.bfloat16)
    m32 = copy.deepcopy(m16).float()   # the same bfloat16 weights, computed in float32
    rng = random.Random(0)
    jobs = [Job([rng.randrange(1, 512) for _ in range(P)],
                [[rng.randrange(1, 512) for _ in range(rng.randrange(1, 40))] for _ in range(n)], [])
            for P, n in [(5, 1), (60, 3), (150, 2), (33, 4), (90, 1)]]
    ref = finals(Gemma4(m32), jobs, device).float()
    base = finals(Gemma4(m16), jobs, device)
    off = finals(Gemma4(m16, float32_residual=False), jobs, device)
    wide_bb = Gemma4(m16, float32_residual=True, cuda_graphs=True, graph_step=8)
    wide = finals(wide_bb, jobs, device)
    e16 = float((base.float() - ref).abs().mean())
    e32 = float((wide.float() - ref).abs().mean())
    same_off = bool(torch.equal(base, off))
    dtypes = base.dtype == wide.dtype == torch.bfloat16
    graphs = wide_bb.request_graphs()
    g_err = 0.0
    for j in jobs:
        h = graphs.run(j.prefix, j.suffixes)
        alone = finals(wide_bb, [j], device)
        g_err = max(g_err, float((h.float() - alone.float()).abs().max()))
    lad = fine_ladder(8)
    ladder_ok = (graphs.ladder == lad and all(b - a == 8 for a, b in zip(lad, lad[1:]) if b < 1024)
                 and [x for x in lad if x >= 1024] == [x for x in LADDER if x >= 1024])
    scale = float(ref.abs().max())
    ok = e32 <= e16 and same_off and dtypes and g_err <= 5e-2 * max(1.0, scale) and ladder_ok
    print(f"  mean gap to float32: bfloat16 stream {e16:.3e}, float32 stream {e32:.3e}")
    print(f"  option off = default bits: {same_off}; outputs bfloat16: {dtypes}")
    print(f"  graphs on the ladder of 8 vs packed {g_err:.2e} ({len(graphs.graphs) + len(graphs.bgraphs)} graphs); "
          f"ladder {lad[:3]}..{lad[-1]} ok {ladder_ok}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
