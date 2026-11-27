from itertools import accumulate

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN kernels require CUDA")
def test_packed_gdn_reads_shared_prefix_states():
    from flide.kernels.fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

    lengths = [23, 95, 134, 1]
    tokens = sum(lengths)
    cu_cpu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
    cu = cu_cpu.cuda()
    indices = torch.tensor([2, 0, 2, 1], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    opts = {"device": "cuda", "dtype": torch.bfloat16, "generator": generator}
    q = torch.randn((1, tokens, 16, 128), **opts)
    k = torch.randn(q.shape, **opts)
    v = torch.randn((1, tokens, 32, 128), **opts)
    g = torch.randn((1, tokens, 32), **opts)
    beta = torch.randn(g.shape, **opts)
    states = torch.randn((3, 32, 128, 128), device="cuda", generator=generator) * 0.025
    saved = states.clone()
    options = {
        "cu_seqlens": cu,
        "cu_seqlens_cpu": cu_cpu,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "use_gate_in_kernel": True,
        "A_log": torch.zeros(32, device="cuda"),
        "dt_bias": torch.full((32,), -4.0, device="cuda"),
    }

    expected, expected_final = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=states.index_select(0, indices), **options
    )
    output, final = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=states, initial_state_indices=indices, **options
    )

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(final, expected_final, atol=0, rtol=0)
    torch.testing.assert_close(states, saved, atol=0, rtol=0)
