"""Test batch dependence of exact Attention on Qwen-shaped fixed tensors."""
import argparse
from .common import *
from .watch import GPUWatch


def main():
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    folder=ROOT/'raw'/cfg['name'];monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start()
    try:
        import torch
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        torch.set_num_threads(1);torch.manual_seed(2026111301)
        device='cuda';dtype=torch.bfloat16;block=16;context=8000;nq=32;nkv=8;head=128
        victim_blocks=context//block;neighbor_tokens=512;neighbor_blocks=neighbor_tokens//block
        key=torch.randn((victim_blocks+neighbor_blocks,block,nkv,head),device=device,dtype=dtype)
        value=torch.randn_like(key);victim=torch.randn((1,nq,head),device=device,dtype=dtype)
        neighbor=torch.randn((neighbor_tokens,nq,head),device=device,dtype=dtype)
        tables=torch.zeros((2,victim_blocks),device=device,dtype=torch.int32)
        tables[0]=torch.arange(victim_blocks,device=device,dtype=torch.int32)
        tables[1,:neighbor_blocks]=torch.arange(victim_blocks,victim_blocks+neighbor_blocks,device=device,dtype=torch.int32)
        result=[]
        for version in (3,2):
            baseline=None
            for length in (0,1,8,16,128,256,512):
                query=torch.cat((victim,neighbor[:length])) if length else victim
                cu=torch.tensor([0,1,1+length] if length else [0,1],device=device,dtype=torch.int32)
                seq=torch.tensor([context,length] if length else [context],device=device,dtype=torch.int32)
                try:
                    out=flash_attn_varlen_func(q=query,k=key,v=value,cu_seqlens_q=cu,
                        max_seqlen_q=max(1,length),max_seqlen_k=context,seqused_k=seq,
                        block_table=tables if length else tables[:1],causal=True,num_splits=1,
                        softmax_scale=head**-.5,fa_version=version)
                    torch.cuda.synchronize()
                    if baseline is None:baseline=out[:1].clone()
                    delta=(out[:1].float()-baseline.float()).abs()
                    result.append(dict(version=version,neighbor_queries=length,bitwise_equal=torch.equal(out[:1],baseline),
                        max_abs=float(delta.max()),different_elements=int((out[:1]!=baseline).sum()),passed=True))
                except Exception as error:result.append(dict(version=version,neighbor_queries=length,passed=False,error=repr(error)))
        write(folder/'attention_batch_probe.json',dict(results=result,shape=dict(Q_heads=nq,KV_heads=nkv,head_dim=head,context=context),
            evidence='CORRECTNESS_ONLY',scope='The victim Q/K/V and block mapping stay fixed. This primitive isolates Attention batch dependence. It is not serving speedup or proof of the model-level root cause.'))
        print(json.dumps(result,indent=2),flush=True)
        write(folder/'worker_exits.json',dict(passed=True,scope='Single-process primitive exited normally. No PD worker lifetime claim.'))
        write(folder/'status.json',dict(status='COMPLETE'))
    finally:monitor.stop()


if __name__=='__main__':main()
