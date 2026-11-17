"""Optional SM120 BF16 exact-range fused attention backend."""
from functools import lru_cache


@lru_cache(maxsize=1)
def load():
    """Private upstream compiler instance; other FlashAttention callers are unchanged."""
    import types
    import torch
    import logging
    from .cute_fused_kernel import ExactRangesAttention
    from .vendor.flash_attn.cute import interface
    from .vendor.flash_attn.cute.config import FwdHeuristicInputs, select_fwd_config

    logging.getLogger(__name__).info("using CuTe fused exact-range attention (64-token tiles, SM120 BF16 D64/128/256)")
    original = interface._flash_attn_fwd
    namespace = dict(original.__globals__)
    namespace["FlashAttentionForwardSm120"] = ExactRangesAttention
    forward = types.FunctionType(original.__code__, namespace, original.__name__,
                                 original.__defaults__, original.__closure__)
    forward.__kwdefaults__ = original.__kwdefaults__
    forward.compile_cache = {}
    namespace["_flash_attn_fwd"] = forward

    @lru_cache(maxsize=256)
    def config(device, tokens, keys, heads, kv_heads, sequences, head_dim):
        return select_fwd_config(FwdHeuristicInputs(
            device_arch=120, num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
            dtype="torch.bfloat16", head_dim=head_dim, head_dim_v=head_dim, num_heads=heads, num_heads_kv=kv_heads,
            batch_size=sequences, total_q=tokens, total_k=keys, max_seqlen_q=64, max_seqlen_k=keys+tokens,
            has_cu_seqlens_q=True, has_cu_seqlens_k=True, has_seqused_q=True, has_seqused_k=True,
            causal=False, pack_gqa=False, requested_num_splits=1, requested_tile_m=64, requested_tile_n=64))

    def run(q, k1, v1, k2, v2, metadata, scale):
        aux = [k2.detach(), v2.detach()]
        for x in aux:
            x.__leading_dim__ = 2
            x.__assumed_align__ = 16
        aux.extend([metadata.ranges, metadata.tiles])
        return forward(q, k1, v1, cu_seqlens_q=metadata.cu_q, cu_seqlens_k=metadata.cu_k,
                       seqused_q=metadata.used, seqused_k=metadata.k_lengths,
                       max_seqlen_q=64, max_seqlen_k=k1.shape[0]+k2.shape[0],
                       causal=False, pack_gqa=False, softmax_scale=scale, aux_tensors=aux,
                       config=config(q.device, q.shape[0], k1.shape[0], q.shape[1], k1.shape[1],
                                     metadata.used.numel(), q.shape[-1]))[0]
    return run
