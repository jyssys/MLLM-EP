"""Test FA3 query-partition invariance without changing model weights."""
import argparse,json,time
from .common import *


def main():
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    p=argparse.ArgumentParser();p.add_argument('config');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());folder=ROOT/'raw'/cfg['name']
    write(folder/'status.json',dict(status='RUNNING',time=time.time()))
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']);device='cuda';N=8000
    q=torch.randn(N,32,128,device=device,dtype=torch.bfloat16)
    cache=torch.randn(500,8,16,256,device=device,dtype=torch.bfloat16)
    k,v=cache.transpose(1,2).split(128,dim=-1)
    blocks=torch.arange(500,device=device,dtype=torch.int32)[None]
    def attention(query,context,upper=None):
        count=query.shape[0]
        return flash_attn_varlen_func(q=query,k=k,v=v,
            cu_seqlens_q=torch.tensor([0,count],device=device,dtype=torch.int32),
            max_seqlen_q=upper or count,seqused_k=torch.tensor([context],device=device,dtype=torch.int32),
            max_seqlen_k=upper or context,block_table=blocks,causal=True,softmax_scale=128**-.5,
            fa_version=3,num_splits=1)
    for _ in range(3):reference=attention(q,N)
    torch.cuda.synchronize();assert torch.equal(reference,attention(q,N))
    results=[]
    for chunk in (256,512,1024,2048):
        for policy in ('DEFAULT','CANONICAL_UPPER_BOUND'):
            upper=N if policy=='CANONICAL_UPPER_BOUND' else None
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();pieces=[]
            for start in range(0,N,chunk):
                stop=min(N,start+chunk);pieces.append(attention(q[start:stop],stop,upper))
            end.record();end.synchronize();value=torch.cat(pieces)
            delta=(reference.float()-value.float()).abs()
            results.append(dict(chunk=chunk,policy=policy,bitwise_equal=bool(torch.equal(reference,value)),
                different_elements=int((reference!=value).sum()),elements=value.numel(),maximum_absolute_error=float(delta.max()),
                CUDA_event_enclosure_ms=begin.elapsed_time(end),
                scope='Paged GQA kernel diagnostic. The same Q/K/V and causal token positions are used. This is not model correctness or serving performance.'))
    write(folder/'results.json',dict(rows=results,reference_repeat_exact=True,
        precision='BF16',query_heads=32,KV_heads=8,head_dim=128,prompt_tokens=N,
        evidence='MEASURED_KERNEL_DIAGNOSTIC'))
    write(folder/'worker_exits.json',dict(passed=True));write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
