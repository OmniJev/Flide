# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of
# https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm80.h
# and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h
# from Cutlass C++ to Cute-DSL.
# Built on Cute-DSL example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

from types import SimpleNamespace
from typing import Callable, Optional
from functools import partial


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp

from quack import layout_utils

from flide.kernels.vendor.flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flide.kernels.vendor.flash_attn.cute import utils
from flide.kernels.vendor.flash_attn.cute.softmax import Softmax
from flide.kernels.vendor.flash_attn.cute.seqlen_info import SeqlenInfoQK
from flide.kernels.vendor.flash_attn.cute.pack_gqa import PackGQA
from flide.kernels.vendor.flash_attn.cute.utils import AuxData



from flide.kernels.vendor.flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

class ExactRangesAttention(FlashAttentionForwardSm120):
    """SM120 dual-source attention with FLIDE exact per-row visibility intervals."""

    @cute.jit
    def load_K_branch(self, tiled_copy, B, S, C, C0, P, block: Int32,
                      smem_pipe_write: Int32, seqlen: Int32,
                      need_predicates: cutlass.Constexpr, shift: Int32):
        self.load_K(tiled_copy, B, S, C, C0, P, block + shift, smem_pipe_write, seqlen, need_predicates)

    @cute.jit
    def load_V_branch(self, tiled_copy, B, S, C, C0, P, block: Int32,
                      smem_pipe_write: Int32, seqlen: Int32,
                      need_predicates: cutlass.Constexpr, shift: Int32):
        self.load_V(tiled_copy, B, S, C, C0, P, block + shift, smem_pipe_write, seqlen, need_predicates)

    @cute.jit
    def load_K_packed(self, tiled_copy, A, B, S, C, C0, P, block: Int32,
                      smem_pipe_write: Int32, seqlen: Int32,
                      need_predicates: cutlass.Constexpr, prefix_len: Int32,
                      prefix_blocks: Int32, branch_first_block: Int32):
        if block < prefix_blocks:
            self.load_K(tiled_copy, A, S, C, C0, P, block, smem_pipe_write, prefix_len, True)
        else:
            self.load_K(tiled_copy, B, S, C, C0, P,
                        block - prefix_blocks + branch_first_block, smem_pipe_write, seqlen, True)

    @cute.jit
    def load_V_packed(self, tiled_copy, A, B, S, C, C0, P, block: Int32,
                      smem_pipe_write: Int32, seqlen: Int32,
                      need_predicates: cutlass.Constexpr, prefix_len: Int32,
                      prefix_blocks: Int32, branch_first_block: Int32):
        if block < prefix_blocks:
            self.load_V(tiled_copy, A, S, C, C0, P, block, smem_pipe_write, prefix_len, True)
        else:
            self.load_V(tiled_copy, B, S, C, C0, P,
                        block - prefix_blocks + branch_first_block, smem_pipe_write, seqlen, True)

    @cute.jit
    def mask_source(self, acc_S, n_block: Int32, coords, bounds, prefix_blocks: Int32,
                    source: cutlass.Constexpr):
        scores = layout_utils.reshape_acc_to_mn(acc_S)
        for r in cutlass.range_constexpr(cute.size(scores.shape[0])):
            for c in cutlass.range_constexpr(cute.size(scores.shape[1])):
                ki = (n_block - (prefix_blocks if source == 2 else 0)) * self.tile_n + coords[0,c][1]
                valid = (ki >= bounds[r, 0 if source == 1 else 2]) & (ki < bounds[r, 1 if source == 1 else 3])
                scores[r,c] = scores[r,c] if valid else -Float32.inf

    @cute.jit
    def mask_either(self, acc_S, n_block: Int32, coords, bounds, prefix_blocks: Int32):
        if n_block < prefix_blocks:
            self.mask_source(acc_S, n_block, coords, bounds, prefix_blocks, 1)
        else:
            self.mask_source(acc_S, n_block, coords, bounds, prefix_blocks, 2)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        if work_tile.is_valid_tile:
            m_block, num_head, batch_size, _ = work_tile.tile_idx

            seqlen = SeqlenInfoQK.create(
                batch_idx=batch_size,
                seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
                seqlen_k_static=mK.shape[0],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
            )
            # Each sequence is an existing FLIDE query tile (at most 64 rows).
            # Exact per-row intervals define visibility into two independent KV sources.
            table = aux_data.tensors[3]
            prefix_len = table[batch_size, 5] - table[batch_size, 2]
            second_len = table[batch_size, 9] - table[batch_size, 6]
            prefix_blocks = cute.ceil_div(prefix_len, self.tile_n)
            branch_first_block = Int32(0)
            n_block_max = prefix_blocks + cute.ceil_div(second_len, self.tile_n)
            n_block = cutlass.max(n_block_max - 1, 0)

            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            blkQ_shape = (self.tile_m, self.tile_hdim)
            blkK_shape = (self.tile_n, self.tile_hdim)
            blkV_shape = (self.tile_n, self.tile_hdimv)
            num_head_kv = num_head if const_expr(self.pack_gqa) else num_head // self.qhead_per_kvhead
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_size, dim=3)[None, None, num_head]
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur = mK[None, None, num_head_kv, batch_size]
                mV_cur = mV[None, None, num_head_kv, batch_size]
            else:
                mK_cur = cute.domain_offset((cutlass.Int64(seqlen.offset_k), 0), mK[None, None, num_head_kv])
                mV_cur = cute.domain_offset((cutlass.Int64(seqlen.offset_k), 0), mV[None, None, num_head_kv])
            if const_expr(not self.pack_gqa):
                gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
            gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
            gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))
            # Both KV sources are addressed directly; exact row ranges define visibility.
            mK2, mV2 = [
                cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))
                for t in [assume_tensor_aligned(x) for x in aux_data.tensors[:2]]
            ]
            branch_offset = cutlass.Int64(table[batch_size, 6])
            mK2_cur = cute.domain_offset((branch_offset, 0), mK2[None, None, num_head_kv])
            mV2_cur = cute.domain_offset((branch_offset, 0), mV2[None, None, num_head_kv])
            gK2 = cute.local_tile(mK2_cur, blkK_shape, (None, 0))
            gV2 = cute.local_tile(mV2_cur, blkV_shape, (None, 0))


            # ///////////////////////////////////////////////////////////////////////////////
            # Get shared memory buffer
            # ///////////////////////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            if const_expr(not self.Q_in_regs):
                sV = storage.sV.get_tensor(sV_layout)
            else:
                sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
            # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
            sVt = layout_utils.transpose_view(sV)

            gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
            gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
            # (CPY_Atom, CPY_N, CPY_K, n_block)
            tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
            # (CPY_Atom, CPY_N, CPY_K, n_block)
            tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)
            tKgK2 = gmem_thr_copy_K.partition_S(gK2)
            tVgV2 = gmem_thr_copy_V.partition_S(gV2)


            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
            tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
            acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
            acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
            acc_O.fill(0.0)

            # ///////////////////////////////////////////////////////////////////////////////
            # Smem copy atom tiling
            # ///////////////////////////////////////////////////////////////////////////////
            smem_copy_atom_QK = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.dtype,
            )
            smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)

            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////
            # Construct identity layout for KV
            cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tKcK = gmem_thr_copy_K.partition_S(cK)
            t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
            if const_expr(self.tile_hdim == self.tile_hdimv):
                tVcV = tKcK
                t0VcV = t0KcK
            else:
                cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
                tVcV = gmem_thr_copy_V.partition_S(cV)
                t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
            # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
            # use "if" on the mn dimension.
            # This is to reduce register pressure and gets 2-3% performance gain.
            tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
            if const_expr(self.same_hdim_kv):
                tVpV = tKpK
            else:
                tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

            # shape: (atom_v_m * rest_m)
            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            # group parameters for compute_one_n_block
            mma_params = SimpleNamespace(
                thr_mma_qk=thr_mma_qk,
                thr_mma_pv=thr_mma_pv,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tOrVt=tOrVt,
                acc_O=acc_O,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_Q=smem_thr_copy_Q,
                smem_thr_copy_K=smem_thr_copy_K,
                smem_thr_copy_V=smem_thr_copy_V,
                tSsQ=tSsQ,
                tSsK=tSsK,
                tOsVt=tOsVt,
            )
            load_K = partial(
                self.load_K_packed, gmem_tiled_copy_K, tKgK, tKgK2, tKsK, tKcK, t0KcK, tKpK, prefix_len=prefix_len, prefix_blocks=prefix_blocks, branch_first_block=branch_first_block, seqlen=second_len
            )
            load_V = partial(
                self.load_V_packed, gmem_tiled_copy_V, tVgV, tVgV2, tVsV, tVcV, t0VcV, tVpV, prefix_len=prefix_len, prefix_blocks=prefix_blocks, branch_first_block=branch_first_block, seqlen=second_len
            )

            compute_one_n_block = partial(
                self.compute_one_n_block,
                mma_params=mma_params,
                smem_copy_params=smem_copy_params,
                softmax=softmax,
                load_K=load_K,
                load_V=load_V,
                score_mod=self.score_mod,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

            # ///////////////////////////////////////////////////////////////////////////////
            # Prologue
            # ///////////////////////////////////////////////////////////////////////////////
            # Fixed-source callbacks for the long, fully-visible loops.
            compute_prefix = partial(compute_one_n_block,
                load_K=partial(self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK, seqlen=prefix_len),
                load_V=partial(self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV, seqlen=prefix_len))
            compute_branch = partial(compute_one_n_block,
                load_K=partial(self.load_K_branch, gmem_tiled_copy_K, tKgK2, tKsK, tKcK, t0KcK, tKpK,
                    seqlen=second_len, shift=-prefix_blocks),
                load_V=partial(self.load_V_branch, gmem_tiled_copy_V, tVgV2, tVsV, tVcV, t0VcV, tVpV,
                    seqlen=second_len, shift=-prefix_blocks))
            # Start async loads of the last mn-tile, where we take care of the mn residue
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            if const_expr(not self.pack_gqa):
                self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
            else:
                pack_gqa = PackGQA(self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead)
                pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
            cute.arch.cp_async_commit_group()

            def preprocess_Q():
                cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
                if const_expr(self.Q_in_regs):
                    cute.arch.barrier()
                    tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                    cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

            # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
            # read from smem_q to registers, then load V.
            # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
            if const_expr(self.Q_in_regs):
                load_K(n_block, smem_pipe_write=0, need_predicates=True)
                cute.arch.cp_async_commit_group()
                preprocess_Q()
                cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

            for stage in cutlass.range_constexpr(self.num_stages):
                if const_expr(not self.Q_in_regs or stage > 0):
                    if stage == 0 or n_block - stage >= 0:
                        load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                    cute.arch.cp_async_commit_group()
                if const_expr(stage < self.num_stages - 1):
                    if stage == 0 or n_block - stage >= 0:
                        load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                    cute.arch.cp_async_commit_group()
            if const_expr(not self.Q_in_regs):
                preprocess_Q()

            # ///////////////////////////////////////////////////////////////////////////////
            # Mainloop
            # ///////////////////////////////////////////////////////////////////////////////
            # Start processing of the first n-block.
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            coords = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
            bounds = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1], 4), Int32)
            for r in cutlass.range_constexpr(cute.size(bounds.shape[0])):
                qi = coords[r, 0][0]
                for col in cutlass.range_constexpr(4):
                    bounds[r, col] = Int32(0)
                    if qi < seqlen.seqlen_q:
                        base = table[batch_size, 2 if col < 2 else 6]
                        bounds[r, col] = aux_data.tensors[2][col, seqlen.offset_q + qi] - base
            mask1 = partial(self.mask_source, coords=coords, bounds=bounds,
                prefix_blocks=prefix_blocks, source=1)
            mask2 = partial(self.mask_source, coords=coords, bounds=bounds,
                prefix_blocks=prefix_blocks, source=2)
            mask_any = partial(self.mask_either, coords=coords, bounds=bounds, prefix_blocks=prefix_blocks)
            compute_one_n_block(n_block, Int32(0), Int32(0), is_first_n_block=True,
                seqlen=seqlen, mask_fn=mask_any)
            # Source 2: masked tail, fully visible blocks, masked head.
            upper2 = cutlass.max(prefix_blocks, n_block)
            hi2 = cutlass.min(upper2, prefix_blocks + (table[batch_size, 8] - table[batch_size, 6]) // self.tile_n)
            lo2 = cutlass.min(hi2, prefix_blocks + (table[batch_size, 7] - table[batch_size, 6]) // self.tile_n)
            for nt in cutlass.range(upper2 - hi2, unroll=1):
                compute_one_n_block(upper2 - 1 - nt, Int32(0), Int32(0), seqlen=seqlen, mask_fn=mask2)
            safe2 = cutlass.min(hi2, cutlass.max(lo2, prefix_blocks + 1))
            for nt in cutlass.range(hi2 - safe2, unroll=1):
                compute_branch(hi2 - 1 - nt, Int32(0), Int32(0), seqlen=seqlen, mask_fn=None)
            for nt in cutlass.range(safe2 - lo2, unroll=1):
                compute_one_n_block(safe2 - 1 - nt, Int32(0), Int32(0), seqlen=seqlen, mask_fn=None)
            for nt in cutlass.range(lo2 - prefix_blocks, unroll=1):
                compute_one_n_block(lo2 - 1 - nt, Int32(0), Int32(0), seqlen=seqlen, mask_fn=mask2)
            # Source 1. The first overall tile may already have processed its tail.
            upper1 = cutlass.min(prefix_blocks, n_block)
            hi1 = cutlass.min(upper1, (table[batch_size, 4] - table[batch_size, 2]) // self.tile_n)
            lo1 = cutlass.min(hi1, (table[batch_size, 3] - table[batch_size, 2]) // self.tile_n)
            for nt in cutlass.range(upper1 - hi1, unroll=1):
                compute_one_n_block(upper1 - 1 - nt, Int32(0), Int32(0), seqlen=seqlen, mask_fn=mask1)
            for nt in cutlass.range(hi1 - lo1, unroll=1):
                compute_prefix(hi1 - 1 - nt, Int32(0), Int32(0), seqlen=seqlen, mask_fn=None)
            for nt in cutlass.range(lo1, unroll=1):
                compute_one_n_block(lo1 - 1 - nt, Int32(0), Int32(0), seqlen=seqlen, mask_fn=mask1)

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            # reuse sQ's data iterator
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                num_head,
                batch_size,
            )
