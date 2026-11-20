"""Validated CuTe head dimensions, scale/cache separation and unsupported fallbacks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_cute_exact_attention import check, check_grouped_graph
import torch
from flide import settings
from flide.kernels import attention as attn
from flide.kernels.attention_backends import configure,prepare_attention,optimized_attention,_cute

@torch.inference_mode()
def main():
    saved=settings.attention_backend
    try:
        configure('auto')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0) or _cute() is None:
            print('SKIP requires SM120 and CuTe');return
        configure('cute');torch.manual_seed(83);count=0
        # Interleave dimensions to detect an incorrectly shared compiled configuration.
        for dim in [64,128,256,64]:
            for t in [1,53,64,65,257]:
                a=torch.arange(t,device='cuda',dtype=torch.int32)
                cases=[torch.stack([a*0,a*0+32,a//53*53,a+1]),
                       torch.stack([a*0,a*0+max(t,32),a*0,a*0]),
                       torch.stack([(a-13).clamp_min(0),a+1,a*0,a*0]),
                       torch.zeros(4,t,device='cuda',dtype=torch.int32)]
                for r in cases:check(r,k1_tokens=max(t,32),head_dim=dim);count+=1
            for h,hk in [(2,2),(16,2)]:
                t=129;a=torch.arange(t,device='cuda',dtype=torch.int32)
                r=torch.stack([a*0+3,a*0+19,a//53*53,a+1])
                check(r,k1_tokens=256,head_dim=dim,heads=h,kv_heads=hk,graph=True);count+=1
            # Pure causal attention uses the same unified backend.
            t=4096;a=torch.arange(t,device='cuda',dtype=torch.int32)
            r=torch.stack([a*0,a+1,a*0,a*0])
            q=torch.randn(t,4,dim,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(t,2,dim,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
            tok=tuple(a*0 for _ in range(3));cfg=attn.config(q.device,q.dtype,dim)
            tiles=attn.block_ranges(*r,cfg['BLOCK_M'],cfg['BLOCK_N'],exact=True)
            pre=prepare_attention(r,tiles,dtype=q.dtype,head_dim=dim)
            assert pre is not None and pre.covered.sum().item()==t
            ref=attn.attention(q,k,v,k,v,tok,tok,tok,tiles,dim**-.5,cfg=cfg,qranges=r)
            out=attn.attention(q,k,v,k,v,tok,tok,tok,tiles,dim**-.5,cfg=cfg,qranges=r,backend_metadata=pre)
            torch.testing.assert_close(out,ref,atol=.02,rtol=.02);count+=1
            print('PASS head_dim',dim,flush=True)
        for dim in [32,96,192,512]:
            assert prepare_attention(r,tiles,dtype=q.dtype,head_dim=dim) is None
        # Mixed Q/K/V head dimensions must not accidentally reach CuTe.
        pre=prepare_attention(r,tiles,dtype=q.dtype,head_dim=64)
        assert optimized_attention(q,k[:,:,:32],v[:,:,:32],k,v,64**-.5,0,pre) is None
        print('PASS unsupported and mismatched dimensions',flush=True)
        for dim in [64,128]:
            check_grouped_graph(head_dim=dim);count+=1
        print('DONE',count,'GPU cases',flush=True)
    finally:settings.attention_backend=saved
if __name__=='__main__':main()
