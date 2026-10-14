"""Stage B physical kernel kill-test. Synthetic tensors, never task speedup.

The native Stage-A geometry is TP4: four Q heads and one complete KV head
per rank. The candidate transposes head ownership into sequence ownership
on the SAME four ranks. It does not pretend orthogonal TP4×CP4 fits four GPUs.
"""
import argparse
import gc
import os
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from poc_dllm_prefix_canvas_cp.contract import *


def flash(q,k,v):
    out,lse,*_=torch.ops.aten._scaled_dot_product_flash_attention(
        q,k,v,0.,False,False,scale=128**-.5)
    return out,lse


def repeat(k):return k.expand(1,4,k.shape[2],128)


class Case:
    def __init__(self,rank,prefix,layer):
        self.rank=rank;self.prefix=prefix;self.N=prefix*2;self.fixed=self.N-32
        g=torch.Generator(device=rank);g.manual_seed(19411+rank*100+layer)
        self.k=torch.randn(1,1,self.N,128,device=rank,dtype=torch.bfloat16,generator=g)
        self.v=torch.randn_like(self.k)
        self.k[:,:,prefix:-32].zero_();self.v[:,:,prefix:-32].zero_()
        self.states=[]
        for _ in range(3):
            self.states.append((torch.randn(1,32,4,128,device=rank,dtype=torch.bfloat16,generator=g).transpose(1,2),
                torch.randn(1,1,32,128,device=rank,dtype=torch.bfloat16,generator=g),
                torch.randn(1,1,32,128,device=rank,dtype=torch.bfloat16,generator=g)))

    def native(self,state):
        q,k,v=state
        a=self.k.slice_scatter(k,dim=2,start=self.fixed,end=self.N)
        b=self.v.slice_scatter(v,dim=2,start=self.fixed,end=self.N)
        return F.scaled_dot_product_attention(q,repeat(a),repeat(b),is_causal=False)

    def local(self,state):
        q,k,v=state
        self.k[:,:,-32:].copy_(k);self.v[:,:,-32:].copy_(v)
        return F.scaled_dot_product_attention(q,repeat(self.k),repeat(self.v),is_causal=False)

    def prepare_cp(self):
        self.L=self.fixed//4
        self.cp=[]
        for tensor in [self.k,self.v]:
            send=tensor[0,0,:self.fixed].contiguous().view(4,self.L,128)
            recv=torch.empty_like(send)
            dist.all_to_all_single(recv,send)
            if self.rank==0:
                work=torch.empty(1,4,self.L+32,128,device=self.rank,dtype=torch.bfloat16)
                work[:,:,:self.L].copy_(recv.unsqueeze(0))
            else:work=recv.unsqueeze(0)
            self.cp.append(work)

    def distributed(self,state):
        q,k,v=state
        source=torch.cat([q[0],k[0],v[0]],dim=0).contiguous().flatten()
        gathered=torch.empty(source.numel()*4,device=self.rank,dtype=source.dtype)
        dist.all_gather_into_tensor(gathered,source)
        all_heads=gathered.view(4,6,32,128)
        queries=all_heads[:,:4].reshape(1,16,32,128).contiguous()
        if self.rank==0:
            self.cp[0][:,:,-32:].copy_(all_heads[:,4].unsqueeze(0))
            self.cp[1][:,:,-32:].copy_(all_heads[:,5].unsqueeze(0))
        output,lse=flash(queries,self.cp[0],self.cp[1])
        maximum=lse.clone();dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
        weight=torch.exp(lse-maximum)
        payload=torch.cat([output.float()*weight.unsqueeze(-1),weight.unsqueeze(-1)],dim=-1).contiguous().flatten()
        local=torch.empty(payload.numel()//4,device=self.rank,dtype=torch.float32)
        dist.reduce_scatter_tensor(local,payload,op=dist.ReduceOp.SUM)
        local=local.view(1,4,32,129)
        return (local[...,:128]/local[...,128:]).to(torch.bfloat16)


def worker(rank,port,args):
    torch.set_grad_enabled(False);torch.cuda.set_device(rank)
    torch.manual_seed(19411+rank);torch.cuda.manual_seed_all(19411+rank)
    torch.use_deterministic_algorithms(True)
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,
        device_id=torch.device('cuda',rank))
    results=[]
    for prefix in [512,8192,16384]:
        cases=[Case(rank,prefix,layer) for layer in range(20)]
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        dist.barrier();a.record();started=time.perf_counter()
        for case in cases:case.prepare_cp()
        b.record();torch.cuda.synchronize()
        prep=dict(GPU_ms=a.elapsed_time(b),wall_ms=(time.perf_counter()-started)*1000,
            remote_prefix_payload_bytes_per_rank=20*2*3*(prefix*2-32)//4*128*2,
            label='ONCE_PER_BLOCK_HEAD_TO_SEQUENCE_RESHARD; actual physical all-to-all')
        prep['cold_GPU_ms']=prep['GPU_ms'];prep['cold_wall_ms']=prep['wall_ms']
        for _ in range(3):
            for case in cases:case.prepare_cp()
        torch.cuda.synchronize();dist.barrier();a.record();started=time.perf_counter()
        for case in cases:case.prepare_cp()
        b.record();torch.cuda.synchronize()
        prep.update(GPU_ms=a.elapsed_time(b),wall_ms=(time.perf_counter()-started)*1000,
            distinction='Warm per-block preparation is primary; first-process setup reported separately')
        errors=[]
        for layer,case in enumerate(cases):
            for step,state in enumerate(case.states):
                native=case.native(state);local=case.local(state);candidate=case.distributed(state)
                error=(candidate.float()-native.float()).abs()
                rel=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(native.float()).clamp_min(1e-10))
                okay=torch.allclose(candidate,native,atol=.03125,rtol=.01) and rel<=.01
                assert torch.equal(local,native),'Local canvas-only update must preserve exact kernel output'
                errors.append(dict(layer=layer,step=step,max_abs=float(error.max()),relative_L2=rel,pass_all=okay))
        assert all(e['pass_all'] for e in errors),'Frozen BF16 tolerance failed; do not relax it'
        # One real version change. Prefix contents change before re-preparation.
        # This is a synthetic correctness probe, not native block-commit parity.
        for case in cases:
            case.k[:,:,prefix:prefix+32].copy_(case.states[-1][1])
            case.v[:,:,prefix:prefix+32].copy_(case.states[-1][2])
            case.prepare_cp()
            assert torch.allclose(case.distributed(case.states[0]),case.native(case.states[0]),atol=.03125,rtol=.01)
        graphs={};last_outputs={}
        def execute(arm):
            for step in range(3):
                for case in cases:out=getattr(case,arm)(case.states[step])
            return out
        for arm in ['native','local','distributed']:
            for _ in range(3):execute(arm)
            dist.barrier();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):last_outputs[arm]=execute(arm)
            graphs[arm]=graph
        samples=[]
        for repeat_no in range(10):
            arms=['native','local','distributed']
            arms=arms[repeat_no%3:]+arms[:repeat_no%3]
            for arm in arms:
                dist.barrier();torch.cuda.synchronize();a.record();start=time.perf_counter()
                for _ in range(10):graphs[arm].replay()
                b.record();torch.cuda.synchronize()
                samples.append(dict(repeat=repeat_no,arm=arm,GPU_ms=a.elapsed_time(b)/10,
                    wall_ms=(time.perf_counter()-start)*1000/10,attention_calls=60))
        # Check two native ordinary states against an independent FP32 reference.
        ref=[]
        for layer in [0,19]:
            case=cases[layer];q,k,v=case.states[0]
            case.local((q,k,v))
            expected=torch.softmax((q.float()@repeat(case.k).float().transpose(-1,-2))*128**-.5,-1)@repeat(case.v).float()
            output=case.distributed((q,k,v)).float()
            ref.append(dict(layer=layer,max_abs=float((output-expected).abs().max()),
                relative_L2=float(torch.linalg.vector_norm(output-expected)/torch.linalg.vector_norm(expected))))
        results.append(dict(prefix=prefix,physical_keys=2*prefix,source_Q_heads=4,source_KV_heads=1,
            CP_partial_Q_heads=16,CP_partial_KV_heads=4,queries=32,rank=rank,prep=prep,samples=samples,
            source_Q_stride=list(cases[0].states[0][0].stride()),global_seed=19411+rank,
            correctness=errors,FP32_reference=ref,quality='NOT_RUN; synthetic attention tensors, not task outputs',
            max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved()))
        write(Path(args.output)/f'results.rank{rank}.json',dict(results=results,
            dtype='BF16 attention, FP32 global softmax merge',mask='same native unmasked regular-refinement path; physical zero slots retained',
            unsupported='Masked/cross-refresh/empty-key paths are not candidate acceleration claims',
            kernel='existing Torch FlashAttention operator; no new CUDA kernel',
            prior_scope='Established distributed softmax; no algorithm novelty claimed'))
        if rank==0:print('STAGE B',prefix,'correctness PASS; measured all3 arms,10 paired repeats',flush=True)
        for graph in graphs.values():graph.reset()
        graphs.clear();last_outputs.clear();del cases
        gc.collect();torch.cuda.empty_cache();dist.barrier()
    torch.cuda.synchronize();dist.barrier()
    write(Path(args.output)/f'worker_complete.rank{rank}.json',dict(success=True,GPUS=GPUS))
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,a),nprocs=4,join=True)
    write(Path(a.output)/'status.json',dict(success=True,scope='SYNTHETIC_SHAPE_MATCHED_PHYSICAL_ATTENTION_MICROBENCH_ONLY'))


if __name__=='__main__':main()
