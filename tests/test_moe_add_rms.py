"""Fused MoE combine/residual/norm must retain both materialized rounding boundaries."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flide.kernels import moe,pointwise as pw
run = moe.combine_add_rms


def inputs(t,n,k,dtype,expert32=False,strided=False):
    y=torch.randn(t*k,n,device='cuda',dtype=torch.float32 if expert32 else dtype)*.2
    s=torch.randn(t,n*2 if strided else n,device='cuda',dtype=dtype)[:,:n]
    g=torch.randn(t,2 if strided else 1,device='cuda',dtype=torch.float32)[:,:1]
    g.mul_(3)
    h=torch.randn(t,n*2 if strided else n,device='cuda',dtype=dtype)[:,:n]
    w=torch.randn(n,device='cuda',dtype=dtype)*.1
    return y,s,g,k,h,w

def ref(y,s,g,k,h,w,eps=1e-6,offset=True):
    return pw.rms_norm(h,w,eps,residual=moe.combine(y,s,g,k),offset=offset)

def same(a,b):
    return all(torch.equal(x,y) for x,y in zip(a,b))
def graph(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    return g,out

torch.manual_seed(1173)
checks = 0
for dtype in [torch.bfloat16,torch.float16,torch.float32]:
 for n in [128,768,2048,4096]:
  for k in [1,2,8]:
   for expert32 in [False,True]:
    a=inputs(137,n,k,dtype,expert32,True)
    for offset in [False,True]:
     assert same(run(*a,1e-6,offset),ref(*a,1e-6,offset)),(dtype,n,k,expert32,offset)
    g,out=graph(lambda:run(*a,1e-6))
    for x in a:
     if isinstance(x,torch.Tensor):x.normal_()
    g.replay()
    expected=ref(*a)
    assert same(out, expected), (dtype, n, k, expert32, 'changed graph inputs')
    checks += 1
print('CORRECT',checks,'cases; offsets and changed graph inputs',flush=True)
# Exercise cancellation, rounding-sensitive residuals, zero and extreme finite gates.
for t in [0,1,140,1024]:
 a=inputs(t,2048,8,torch.bfloat16)
 y,s,g,k,h,w=a
 if t:
  y[1::8].copy_(-y[0::8]);g[:,0].copy_(torch.linspace(-100,100,t,device='cuda'))
 assert same(run(*a,1e-6),ref(*a))


print("PASS: 72 dtype/shape cases, both offsets, dynamic graphs and four boundary cases")

# Fixed SM120 failure: splitting the shared-product multiply/add reaches a BF16
# midpoint; the original combine uses FMA and lands one FP32 ULP below it.
y,s,g,k,h,w=inputs(137,768,1,torch.bfloat16,expert32=True,strided=True)
i,j=105,519
y[i,j] = -0.26482531428337097
s[i,j] = -0.9609375
g[i,0] = 0.5977491140365601
h[i,j] = 0.162109375
a=(y,s,g,k,h,w)
expected=ref(*a)
assert same(run(*a,1e-6),expected), 'combine FMA halfway regression'
print('PASS: fixed combine FMA halfway regression')
