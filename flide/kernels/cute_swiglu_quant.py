"""Fused SwiGLU + FLIDE-compatible per-row FP8 quantization."""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, BFloat16, Int64
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch
from cutlass.utils import SmemAllocator

@cute.kernel
def kernel(X: cute.Tensor, Q: cute.Tensor, S: cute.Tensor,
           N: cutlass.Constexpr, WARPS: cutlass.Constexpr, V: cutlass.Constexpr,
           TPR: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(TPR == 32):
        row = Int64(bid * WARPS + tid // 32)
        lane = tid % 32
    else:
        row = Int64(bid)
        lane = tid
        scratch = SmemAllocator().allocate_tensor(Float32, cute.make_layout(TPR // 32), byte_alignment=16)
    if row < X.shape[0]:
        layout = cute.make_layout((V, cute.ceil_div(N, TPR * V)), stride=(1, TPR * V))
        gate = cute.make_tensor(X.iterator + row * X.stride[0] + lane * V, layout)
        up = cute.make_tensor(X.iterator + row * X.stride[0] + N + lane * V, layout)
        dst = cute.make_tensor(Q.iterator + row * Q.stride[0] + lane * V, layout)
        ga = cute.make_rmem_tensor((V, cute.ceil_div(N, TPR * V)), X.element_type)
        gb = cute.make_rmem_tensor((V, cute.ceil_div(N, TPR * V)), X.element_type)
        if cutlass.const_expr(N % (TPR * V) == 0):
            cute.autovec_copy(gate, ga)
            cute.autovec_copy(up, gb)
        else:
            ga.fill(0)
            gb.fill(0)
            pred = cute.make_rmem_tensor(ga.shape, cutlass.Boolean)
            for i in cutlass.range_constexpr(cute.size(pred)):
                pred[i] = lane + i * TPR < N
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), X.element_type)
            cute.copy(atom, gate, ga, pred=pred)
            cute.copy(atom, up, gb, pred=pred)
        a = ga.load().to(Float32)
        b = gb.load().to(Float32)
        # Match Triton's div.full.f32, including values at BF16 halfway points.
        # Fast-math division can flip one BF16 rounding bit and then an FP8 code.
        shape = (V, cute.ceil_div(N, TPR * V))
        numer = cute.make_rmem_tensor(shape, Float32)
        denom = cute.make_rmem_tensor(shape, Float32)
        up_values = cute.make_rmem_tensor(shape, Float32)
        rounded = cute.make_rmem_tensor(shape, BFloat16)
        numer.store(a)
        denom.store(1.0 + cute.math.exp(-a, fastmath=True))
        up_values.store(b)
        for i in cutlass.range_constexpr(cute.size(shape)):
            value = cute.math.div(numer[i], denom[i], full=True) * up_values[i]
            rounded[i] = BFloat16(value)
        y = rounded.load().to(Float32)
        mx = cute.math.abs(y).reduce(cute.ReductionOp.MAX, Float32(0), 0)
        mx = cute.arch.warp_reduction_max(mx)
        if cutlass.const_expr(TPR > 32):
            if tid % 32 == 0:
                scratch[tid // 32] = mx
            cute.arch.sync_threads()
            mx = Float32(0)
            for i in cutlass.range_constexpr(TPR // 32):
                mx = cute.math.max(mx, scratch[i])
        scale = cute.math.max(mx, Float32(1e-12)) * Float32(1.0 / 448.0)
        if lane == 0:
            S[row, 0] = scale
        scales = cute.make_rmem_tensor(y.shape, Float32)
        scales.fill(scale)
        z = cute.math.div(y, scales.load())
        scales.fill(-448.0)
        z = cute.math.max(z, scales.load())
        scales.fill(448.0)
        z = cute.math.min(z, scales.load())
        out = cute.make_rmem_tensor((V, cute.ceil_div(N, TPR * V)), Q.element_type)
        out.store(z.to(Q.element_type))
        if cutlass.const_expr(N % (TPR * V) == 0):
            cute.autovec_copy(out, dst)
        else:
            atom_out = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Q.element_type)
            cute.copy(atom_out, out, dst, pred=pred)

@cute.jit
def launch(X: cute.Tensor, Q: cute.Tensor, S: cute.Tensor,
           stream: cuda.CUstream, N: cutlass.Constexpr,
           WARPS: cutlass.Constexpr, V: cutlass.Constexpr, TPR: cutlass.Constexpr):
    kernel(X, Q, S, N, WARPS, V, TPR).launch(
        grid=(cute.ceil_div(X.shape[0], WARPS) if TPR == 32 else X.shape[0], 1, 1),
        block=(32 * WARPS if TPR == 32 else TPR, 1, 1), stream=stream)

_CACHE = {}
def swiglu_quant(x, warps=4, vector=1, threads_per_row=32):
    n=x.shape[1]//2
    assert x.is_cuda and x.ndim == 2 and x.shape[1] == 2*n and n > 0
    assert threads_per_row in (32, 128, 256, 512)
    assert vector == 1 and n % 32 == 0 and x.stride(1)==1 and x.dtype==torch.bfloat16
    q=torch.empty((x.shape[0],n),device=x.device,dtype=torch.float8_e4m3fn)
    s=torch.empty((x.shape[0],1),device=x.device,dtype=torch.float32)
    ts=tuple(from_dlpack(t.detach(),assumed_align=16).mark_layout_dynamic(leading_dim=1) for t in (x,q,s))
    stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key=(x.device.index,n,warps,vector,threads_per_row)
    if key not in _CACHE:
        _CACHE[key]=cute.compile(launch,*ts,stream,n,warps,vector,threads_per_row)
    fn=_CACHE[key]
    def run():
        fn(*ts,cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return q,s
    return run()
