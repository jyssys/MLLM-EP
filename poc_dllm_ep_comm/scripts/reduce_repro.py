"""Bounded real-kernel seam for the failed small-M SGLang reduction."""
import os
import json
import torch
from poc_dllm_ep_comm.contract import MASK,write

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==MASK
    torch.cuda.set_device(0);torch.set_default_dtype(torch.bfloat16)
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_sum_reduce_torch_compile
    rows=[]
    try:
        for m in (0,1,8,16,24,32):
            x=torch.randn((m,8,2048),device='cuda',dtype=torch.bfloat16)
            out=torch.empty((m,2048),device='cuda',dtype=torch.bfloat16)
            moe_sum_reduce_torch_compile(x,out,2.5);torch.cuda.synchronize()
            expected=x.sum(dim=1)*2.5
            assert torch.allclose(out.float(),expected.float(),rtol=.025,atol=.04)
            rows.append(dict(m=m,pass_=True))
    except Exception as error:
        write(os.environ['DLLM_COMM_OUTPUT']+'/reduction_repro.json',dict(ok=False,rows=rows,error=str(error)));raise
    write(os.environ['DLLM_COMM_OUTPUT']+'/reduction_repro.json',dict(ok=True,rows=rows,compile_disabled=os.environ.get('TORCHDYNAMO_DISABLE','0')))

if __name__=='__main__':main()
