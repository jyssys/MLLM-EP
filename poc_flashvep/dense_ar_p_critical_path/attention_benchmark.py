"""Measure isolated real-input attention and peer-copy head offloading."""
import argparse
import json
from pathlib import Path
import time
from .common import ROOT,write,digest


def groups(remote_kv_heads):
    assert remote_kv_heads in (2,4)
    return dict(local_KV=8-remote_kv_heads,remote_KV=remote_kv_heads,
        local_Q=(8-remote_kv_heads)*4,remote_Q=remote_kv_heads*4)


def updated_KV_blocks(query_start,key_tokens,block_tokens):
    assert 0<=query_start<key_tokens and block_tokens>0
    return query_start//block_tokens,(key_tokens+block_tokens-1)//block_tokens


class Case:
    def __init__(self,record,remote_kv_heads,cached_kv=None):
        import torch
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func,get_scheduler_metadata
        self.torch=torch;self.flash=flash_attn_varlen_func;self.schedule=get_scheduler_metadata
        self.record=record;self.group=groups(remote_kv_heads);self.n=record['query_tokens'];self.length=record['key_tokens']
        self.cached_kv=cached_kv
        self.new_block_start,new_end=updated_KV_blocks(record['query_start'],record['key_tokens'],record['block_tokens'])
        assert new_end==record['packed_KV'].shape[0]
        with torch.cuda.device(0):
            self.q=record['Q'].to('cuda:0');self.kv=record['packed_KV'].to('cuda:0')
            self.full=torch.empty_like(self.q);self.joined=torch.empty_like(self.q)
            self.qr=torch.empty((self.n,self.group['remote_Q'],128),device='cuda:0',dtype=torch.bfloat16)
            self.kvr=torch.empty((self.kv.shape[0],self.group['remote_KV'],128,256),device='cuda:0',dtype=torch.bfloat16)
            self.local=torch.empty((self.n,self.group['local_Q'],128),device='cuda:0',dtype=torch.bfloat16)
            self.remote0=torch.empty_like(self.qr);self.ready=torch.cuda.Event();self.done=torch.cuda.Event()
        with torch.cuda.device(1):
            self.stream=torch.cuda.Stream(device=1);self.qd=torch.empty_like(self.qr,device='cuda:1')
            self.kvd=torch.empty_like(self.kvr,device='cuda:1');self.remote1=torch.empty_like(self.qr,device='cuda:1')
        self.meta={}
        for device,qh,kh in ((0,32,8),(0,self.group['local_Q'],self.group['local_KV']),
            (0,self.group['remote_Q'],self.group['remote_KV']),(1,self.group['remote_Q'],self.group['remote_KV'])):
            with torch.cuda.device(device):
                qloc=torch.tensor([0,self.n],device=device,dtype=torch.int32)
                used=torch.tensor([self.length],device=device,dtype=torch.int32)
                table=torch.arange(self.kv.shape[0],device=device,dtype=torch.int32).view(1,-1)
                sched=self.schedule(batch_size=1,max_seqlen_q=self.n,max_seqlen_k=self.length,num_heads_q=qh,num_heads_kv=kh,
                    headdim=128,cache_seqlens=used,qkv_dtype=torch.bfloat16,cu_seqlens_q=qloc,page_size=128,
                    causal=True,window_size=(-1,-1),num_splits=record['num_splits']) if record['aot_schedule'] else None
                self.meta[device,qh,kh]=dict(cu_seqlens_q=qloc,seqused_k=used,block_table=table,scheduler_metadata=sched)

    def attention(self,q,kv,out,device):
        key,value=kv.transpose(1,2).split(128,dim=-1)
        return self.flash(q=q,k=key,v=value,out=out,max_seqlen_q=self.n,max_seqlen_k=self.length,
            softmax_scale=self.record['scale'],causal=True,softcap=self.record['softcap'],
            window_size=self.record['sliding_window'],fa_version=self.record['fa_version'],num_splits=self.record['num_splits'],
            **self.meta[device,q.shape[1],kv.shape[1]])

    def run(self,mode):
        torch=self.torch;g=self.group
        with torch.cuda.device(0):
            if mode=='LocalFull':self.attention(self.q,self.kv,self.full,0);return self.full
            cached=mode=='RemoteCachedSplit'
            begin=self.new_block_start if cached else 0
            self.qr.copy_(self.q[:,g['local_Q']:]);self.kvr[begin:].copy_(self.kv[begin:,g['local_KV']:])
            if mode=='LocalSplit':
                self.attention(self.q[:,:g['local_Q']],self.kv[:,:g['local_KV']],self.local,0)
                self.attention(self.qr,self.kvr,self.remote0,0)
            else:
                assert mode in ('RemoteSplit','RemoteCachedSplit');self.ready.record()
                target=self.cached_kv[:self.kv.shape[0]] if cached else self.kvd
                with torch.cuda.device(1),torch.cuda.stream(self.stream):
                    self.stream.wait_event(self.ready)
                    self.qd.copy_(self.qr,non_blocking=True);target[begin:].copy_(self.kvr[begin:],non_blocking=True)
                    self.attention(self.qd,target,self.remote1,1);self.done.record(self.stream)
                with torch.cuda.device(0):
                    self.attention(self.q[:,:g['local_Q']],self.kv[:,:g['local_KV']],self.local,0)
                    torch.cuda.current_stream().wait_event(self.done)
                    self.remote0.copy_(self.remote1,non_blocking=True)
            self.joined[:,:g['local_Q']].copy_(self.local);self.joined[:,g['local_Q']:].copy_(self.remote0)
            return self.joined


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--start',type=int,required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());base=ROOT/'attention_poc'
    folder=base/'raw'/f'{cfg.get("benchmark_prefix","Benchmark")}_s{args.start}';folder.mkdir(parents=True,exist_ok=True)
    import torch
    import numpy as np
    torch.set_num_threads(1);assert torch.cuda.device_count()==2
    assert torch.cuda.can_device_access_peer(0,1) and torch.cuda.can_device_access_peer(1,0)
    capture=json.loads((base/'raw'/cfg['capture_name']/'capture_manifest.json').read_text());assert capture['passed']
    rows=[];checks=[]
    def sync():
        torch.cuda.synchronize(0);torch.cuda.synchronize(1)
    def compare(a,b):
        x=a.float();y=b.float();delta=(x-y).abs()
        return dict(exact=torch.equal(a,b),max_abs=float(delta.max()),mean_abs=float(delta.mean()),
            RMS_error=float(((x-y).square().mean()).sqrt()))
    try:
        write(folder/'status.json',dict(status='RUNNING',time=time.time()))
        for prompt in cfg['prompts']:
            for kvheads in cfg['remote_KV_heads']:
                records=[row for row in capture['captures'] if row['prompt_id']==prompt['prompt_id']]
                caches={}
                if cfg.get('retain_remote_KV_across_chunks'):
                    with torch.cuda.device(1):
                        caches={layer:torch.empty(((prompt['input_tokens']+127)//128,kvheads,128,256),
                            device='cuda:1',dtype=torch.bfloat16) for layer in cfg['layers']}
                cases=[]
                for row in sorted(records,key=lambda row:(row['query_start'],row['layer'])):
                    path=base/row['path'];assert digest(path)==row['SHA256']
                    record=torch.load(path,map_location='cpu',weights_only=False)
                    case=Case(record,kvheads,caches.get(row['layer']));cases.append((row,case))
                    full=case.run('LocalFull');sync();original=record['output'].reshape_as(full).to(full.device)
                    reference=full.clone();split=case.run('LocalSplit');sync();local=split.clone()
                    remote=case.run('RemoteSplit');sync()
                    check=dict(prompt_id=prompt['prompt_id'],layer=row['layer'],query_start=row['query_start'],
                        remote_KV_heads=kvheads,full_vs_captured=compare(reference,original),
                        split_vs_full=compare(local,reference),remote_vs_local_split=compare(remote,local))
                    if caches:
                        cached=case.run('RemoteCachedSplit');sync();check['cached_vs_local_split']=compare(cached,local)
                    checks.append(check)
                for layers in cfg['layer_counts']:
                    sequence=[(row,case) for row,case in cases if row['layer']<layers]
                    assert sequence
                    for mode in cfg.get('modes',('LocalFull','LocalSplit','RemoteSplit')):
                        for _ in range(cfg['warmup_iterations']):
                            for _,case in sequence:case.run(mode)
                        sync();measurements=[]
                        for _ in range(cfg['iterations']):
                            sync();began=time.perf_counter_ns()
                            for _,case in sequence:case.run(mode)
                            sync();measurements.append((time.perf_counter_ns()-began)/1e6)
                        transfer_bytes=sum(case.qr.numel()*2+
                            case.kvr[case.new_block_start if mode=='RemoteCachedSplit' else 0:].numel()*2+
                            case.remote0.numel()*2 for _,case in sequence)
                        rows.append(dict(start=args.start,prompt_id=prompt['prompt_id'],input_tokens=prompt['input_tokens'],
                            selected_layers=layers,attention_calls=len(sequence),remote_KV_heads=kvheads,
                            remote_query_heads=kvheads*4,mode=mode,iterations=len(measurements),
                            wall_ms=measurements,p50_ms=float(np.percentile(measurements,50)),p95_ms=float(np.percentile(measurements,95)),
                            p99_ms=float(np.percentile(measurements,99)),transfer_payload_bytes=transfer_bytes if mode.startswith('Remote') else 0))
                    write(folder/'progress.json',dict(rows=rows,checks=checks))
                del cases;torch.cuda.empty_cache()
        value=dict(time=time.time(),passed_transport_exact=all(row['remote_vs_local_split']['exact'] for row in checks),
            cached_transport_exact=all(row['cached_vs_local_split']['exact'] for row in checks) if cfg.get('retain_remote_KV_across_chunks') else None,
            full_replay_exact=all(row['full_vs_captured']['exact'] for row in checks),
            split_geometry_exact=all(row['split_vs_full']['exact'] for row in checks),rows=rows,checks=checks,
            GPU_UUIDs=[str(torch.cuda.get_device_properties(i).uuid) for i in (0,1)],
            evidence='ISOLATED_ATTENTION_MEASUREMENT',scope='Real captured paged-FA3 Q/K/V and B8192 chunk boundaries. '
                'Rebuild head-specific AOT scheduling metadata. Preserve GQA groups and all heads. '
                'Wall timing includes packing,peer copies,two-device launch,waiting and output assembly. '
                'QKV projections,FFN and preceding hidden-state dependencies are excluded. '
                '1/2/4 layers mean serial attention call sequences,not complete Transformer layer or prefill timings. '
                'D is idle except this work. No active decode insertion,Safe Window,NIXL/NCCL,model-quality or online speedup claim. '
                'Three independent isolated benchmark processes are not three independent serving starts. '
                'Full model weights and42GiB serving caches are not resident during this isolated replay. '
                'Cached mode retains one remote KV allocation per actual layer and sends only newly written pages. '
                'Each sequence starts at the first chunk. Future stored pages remain masked by the real key length. '
                'CUDA events and profiler hooks are absent. Exclusive GPU compute and actual DMA duration are NOT_MEASURED.')
        write(folder/'result.json',value);write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error)));raise


if __name__=='__main__':main()
