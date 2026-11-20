"""CuTe exact-range routing, including masks beyond causal shared-prefix attention."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from flide import settings
from flide.kernels import attention as attn
from flide.kernels.attention_backends import configure, prepare_attention, optimized_attention, _cute


def check(r, k1_tokens=None, graph=False, head_dim=256, heads=4, kv_heads=2):
    t=r.shape[1];k1_tokens=t if k1_tokens is None else k1_tokens
    q=torch.randn(t,heads,head_dim,device='cuda',dtype=torch.bfloat16)
    k1=torch.randn(k1_tokens,kv_heads,head_dim,device='cuda',dtype=q.dtype)
    v1=torch.randn_like(k1)
    k2=torch.randn(t,kv_heads,head_dim,device='cuda',dtype=q.dtype)
    backing=torch.randn(t,2*kv_heads,head_dim,device='cuda',dtype=q.dtype);v2=backing[:,kv_heads:]
    tok=tuple(torch.zeros(t,dtype=torch.int32,device='cuda') for _ in range(3))
    ctx=tuple(torch.zeros(k1_tokens,dtype=torch.int32,device='cuda') for _ in range(3))
    cfg=attn.config(q.device,q.dtype,head_dim)
    def run(cute):
        tiles=attn.block_ranges(*r,cfg['BLOCK_M'],cfg['BLOCK_N'],exact=True)
        meta=prepare_attention(r,tiles,dtype=q.dtype,head_dim=head_dim,kv_capacity=k1_tokens) if cute else None
        return attn.attention(q,k1,v1,k2,v2,tok,ctx,tok,tiles,head_dim**-.5,cfg=cfg,qranges=r,backend_metadata=meta)
    ref=run(False);out=run(True);torch.testing.assert_close(out,ref,atol=.02,rtol=.02)
    if t <= 65:
        # Independent dense float64 oracle catches configuration errors shared by
        # the test harness and a backend (including differing Triton KV tile sizes).
        keys=torch.cat([k1,k2]).repeat_interleave(heads//kv_heads,dim=1).double()
        values=torch.cat([v1,v2]).repeat_interleave(heads//kv_heads,dim=1).double()
        i=torch.arange(k1_tokens,device='cuda')[None,:]
        j=torch.arange(t,device='cuda')[None,:]
        visible=torch.cat([(i>=r[0,:,None]) & (i<r[1,:,None]),
                           (j>=r[2,:,None]) & (j<r[3,:,None])],dim=1)
        scores=torch.einsum('qhd,khd->hqk',q.double(),keys)*head_dim**-.5
        weights=torch.softmax(scores.masked_fill(~visible[None],float('-inf')),dim=-1).nan_to_num()
        dense=torch.einsum('hqk,khd->qhd',weights,values)
        torch.testing.assert_close(out.double(),dense,atol=.02,rtol=.02)
        torch.testing.assert_close(ref.double(),dense,atol=.02,rtol=.02)
    if graph:
        for _ in range(2):run(True)
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):y=run(True)
        for mode in [0,1,2]:
            ar=torch.arange(t,device='cuda',dtype=torch.int32)
            if mode==0:r[0].zero_();r[1].fill_(k1_tokens);r[2].zero_();r[3].copy_(ar+1)
            elif mode==1:r.zero_() # every row has zero visible keys
            else:r[0].fill_(max(0,k1_tokens-7));r[1].fill_(k1_tokens);r[2].copy_((ar-13).clamp_min(0));r[3].copy_(ar+1)
            g.replay();torch.cuda.synchronize();torch.testing.assert_close(y,run(False),atol=.02,rtol=.02)
        tiles=attn.block_ranges(*r,cfg['BLOCK_M'],cfg['BLOCK_N'],exact=True)
        meta=prepare_attention(r,tiles,dtype=q.dtype,head_dim=head_dim,kv_capacity=k1_tokens)
        assert meta.covered.sum().item()==t
        assert optimized_attention(q,k1,v1,k2,v2,head_dim**-.5,7,meta) is None
        if t > 1:
            assert optimized_attention(q[:1],k1,v1,k2,v2,head_dim**-.5,0,meta) is None
        settings.batch_invariant=True
        try:assert prepare_attention(r,tiles,dtype=q.dtype,head_dim=head_dim) is None
        finally:settings.batch_invariant=False
    return run


@torch.inference_mode()
def main():
    previous=settings.attention_backend
    try:
        configure('auto')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0) or _cute() is None:
            print('SKIP requires SM120 and optional CuTe runtime');return
        configure('cute');torch.manual_seed(37)
        count=0
        for t in [1,53,64,65,127,257,1024]:
            ar=torch.arange(t,device='cuda',dtype=torch.int32)
            cases=[]
            # Bidirectional source 1, then causal and clipped-window attention.
            cases.append(torch.stack([ar*0,ar*0+t,ar*0,ar*0]))
            cases.append(torch.stack([ar*0,ar+1,ar*0,ar*0]))
            cases.append(torch.stack([(ar-31).clamp_min(0),ar+1,ar*0,ar*0]))
            # Isolated sets: full visibility in 17-token groups.
            cases.append(torch.stack([ar//17*17,((ar//17+1)*17).clamp_max(t),ar*0,ar*0]))
            # Two independent KV sources with a partial, not tile-aligned prefix.
            cases.append(torch.stack([ar*0+3,ar*0+19,ar//53*53,ar+1]))
            cases.append(torch.stack([ar*0,ar*0,ar//53*53,ar+1]))
            cases.append(torch.zeros(4,t,device='cuda',dtype=torch.int32))
            for i,r in enumerate(cases):
                check(r,k1_tokens=max(t,32));count+=1
            print('PASS length',t,flush=True)
        r=torch.stack([torch.zeros(257,device='cuda',dtype=torch.int32)]*4)
        check(r,k1_tokens=1024,graph=True)
        print('PASS dynamic graph replay, empty visibility, context capacity, subset/window/invariant fallbacks',flush=True)
        check_grouped_graph()
        print('DONE',count+2,'cases',flush=True)
    finally:settings.attention_backend=previous

@torch.inference_mode()
def check_grouped_graph(head_dim=256):
    # More than 256 sequences exercises the upstream variable-length scheduler's
    # GPU helper tables, including changed group boundaries and empty tile slots.
    t=32768
    r=torch.zeros(4,t,device='cuda',dtype=torch.int32)
    q=torch.randn(t,4,head_dim,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(t,2,head_dim,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
    tok=tuple(torch.zeros(t,dtype=torch.int32,device='cuda') for _ in range(3))
    cfg=attn.config(q.device,q.dtype,head_dim)
    def stage(p,boundary,bidirectional=False):
        r.zero_()
        for lo,hi in [(0,p),(p,512)]:
            r[0,lo:hi]=lo
            r[1,lo:hi]=hi if bidirectional else torch.arange(lo+1,hi+1,device='cuda')
        for lo,hi,pl,ph in [(512,boundary,0,p),(boundary,t,p,512)]:
            ar=torch.arange(lo,hi,device='cuda',dtype=torch.int32)
            r[0,lo:hi]=pl;r[1,lo:hi]=ph;r[2,lo:hi]=lo+(ar-lo)//53*53;r[3,lo:hi]=ar+1
    def run(cute):
        tiles=attn.block_ranges(*r,cfg['BLOCK_M'],cfg['BLOCK_N'],exact=True,groups=5)
        metadata=prepare_attention(r,tiles,dtype=q.dtype,head_dim=head_dim) if cute else None
        return attn.attention(q,k,v,k,v,tok,tok,tok,tiles,head_dim**-.5,cfg=cfg,qranges=r,backend_metadata=metadata)
    stage(256,16640)
    for _ in range(2):run(True)
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run(True)
    for p,boundary,two_way in [(257,16651,False),(255,16623,True),(256,16640,False)]:
        stage(p,boundary,two_way);graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(out,run(False),atol=.02,rtol=.02)
    r.zero_();graph.replay();torch.cuda.synchronize();assert torch.count_nonzero(out).item()==0
    print('PASS >256 dynamic tile sequences, changed groups, bidirectional prefixes and empty slots',flush=True)

if __name__=='__main__':main()
