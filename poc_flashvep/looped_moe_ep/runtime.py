"""Exact ragged expert ownership with NCCL dispatch/combine and source restoration.

Only the native routed-expert expression changes. Attention, routers, residuals,
and recurrence remain native. No output is reused and no expert row is skipped.
"""
import ast
from collections import Counter
import inspect
import sys
import textwrap
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

def owner_ids(experts,world,rank):
    return [e for e in range(experts) if e%world==rank]

def assign_name(node):
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
        return node.targets[0].id
    return None

def patch_class(cls,ifm):
    if hasattr(cls,'_ep_original_forward'):
        return
    original=cls.forward
    body=ast.parse(textwrap.dedent(inspect.getsource(original)))
    statements=body.body[0].body
    if ifm:
        start=next(i for i,n in enumerate(statements) if assign_name(n)=='x' and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='view')
        end=next(i for i,n in enumerate(statements) if assign_name(n)=='y2' and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Name) and n.value.func.id=='fused_permute_y')
        replacement='y2, _ep_counts = self._ep.run(self._ep_site, x, routing_indices, routing_scores)'
    else:
        start=next(i for i,n in enumerate(statements) if assign_name(n)=='x_flat')
        end=next(i for i,n in enumerate(statements) if assign_name(n)=='experts_out')
        replacement='experts_out, counts = self._ep.run(self._ep_site, norm2_out, top_experts, top_scores)'
    statements[start:end+1]=ast.parse(replacement).body
    scope=dict(vars(sys.modules[cls.__module__]))
    exec(compile(ast.fix_missing_locations(body),'<exact-routed-expert-EP-bridge>','exec'),scope)
    cls._ep_original_forward=original
    cls.forward=scope['forward']

class PhysicalEP:
    def __init__(self,adapter):
        self.a=adapter
        self.rank=dist.get_rank()
        self.world=dist.get_world_size()
        self.calls=Counter()
        self.trace=False
        self.profiler_labels=False
        self.count_trace=False
        self.count_records=[]
        self.records=[]
        self.request_ids=[]
        self.ownership=[]
        self.capture_site=sorted(adapter.recurrent)[0]
        self.captured=[]
        self.expert_fns={}
        self.start_memory=torch.cuda.memory_allocated()
        for site,block in adapter.sites.items():
            ifm=adapter.name.startswith('ifm')
            e=block.n_experts if ifm else block.num_experts
            owned=owner_ids(e,self.world,self.rank)
            tensors=[]
            if ifm:
                expert=block.experts
                for name,rows,columns in [('weight1',expert.inter_dim,expert.model_dim),
                        ('weight3',expert.inter_dim,expert.model_dim),('weight2',expert.model_dim,expert.inter_dim)]:
                    old=getattr(expert,name)
                    source=old.detach().reshape(e,rows,columns)
                    local=source[owned].clone().reshape(-1,columns)
                    assert torch.equal(local.reshape(len(owned),rows,columns),source[owned])
                    setattr(expert,name,nn.Parameter(local,requires_grad=False))
                    tensors.append(dict(name=name,shape=list(local.shape),bytes=local.numel()*2,pointer=local.data_ptr(),
                        checksum=float(local.float().sum()),exact_native_subset=True))
                    del old,source,local
                expert.n_local_experts=len(owned)
                self.expert_fns[site]=expert.forward
            else:
                for name in ['experts_w1','experts_w2','experts_w3']:
                    old=getattr(block,name)
                    local=old.detach()[owned].clone()
                    assert torch.equal(local,old[owned])
                    setattr(block,name,nn.Parameter(local,requires_grad=False))
                    tensors.append(dict(name=name,shape=list(local.shape),bytes=local.numel()*2,pointer=local.data_ptr(),
                        checksum=float(local.float().sum()),exact_native_subset=True))
                    del old,local
                def expert_fn(x,counts,block=block):
                    offs=torch.tensor(counts,device=x.device,dtype=torch.int32).cumsum(0).int()
                    return F.grouped_mm(F.silu(F.grouped_mm(x,block.experts_w1,offs=offs))*
                        F.grouped_mm(x,block.experts_w3,offs=offs),block.experts_w2,offs=offs)
                self.expert_fns[site]=expert_fn
            block._ep=self
            block._ep_site=site
            self.ownership.append(dict(site=site,rank=self.rank,logical_experts=e,owned=owned,tensors=tensors,
                recurrent=site in adapter.recurrent,per_expert_bytes=sum(t['bytes'] for t in tensors)//len(owned)))
            patch_class(type(block),ifm)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.resident_after_shard=torch.cuda.memory_allocated()

    def reset(self,request_ids=None):
        self.calls.clear()
        self.records=[]
        self.request_ids=request_ids or []
        self.count_records=[]
        if self.trace:
            self.origin=torch.cuda.Event(enable_timing=True)
            self.origin.record()

    def run(self,site,hidden,ids,weights):
        if self.profiler_labels:
            with torch.profiler.record_function(f'EP_SITE={site}_PASS={self.calls[site]}'):
                return self._run(site,hidden,ids,weights)
        return self._run(site,hidden,ids,weights)

    def _run(self,site,hidden,ids,weights):
        host_ready=time.perf_counter()
        b,l,d=hidden.shape
        k=ids.shape[-1]
        block=self.a.sites[site]
        e=block.n_experts if self.a.name.startswith('ifm') else block.num_experts
        owned=owner_ids(e,self.world,self.rank)
        loop=self.calls[site]
        self.calls[site]+=1
        events=[torch.cuda.Event(enable_timing=True) for _ in range(7)] if self.trace else None
        if events:events[0].record()
        flat=hidden.reshape(-1,d)
        local=flat[self.rank::self.world].contiguous()
        local_ids=ids.reshape(-1,k)[self.rank::self.world]
        local_weights=weights.reshape(-1,k)[self.rank::self.world]
        counts=torch.bincount(local_ids.flatten(),minlength=e).to(torch.int64)
        gathered=torch.empty(self.world*e,dtype=torch.int64,device=hidden.device)
        dist.all_gather_into_tensor(gathered,counts)
        matrix=gathered.reshape(self.world,e).tolist()
        if self.count_trace:
            self.count_records.append(dict(site=site,loop=loop,recurrent=site in self.a.recurrent,
                expert_counts=[sum(matrix[src][j] for src in range(self.world)) for j in range(e)],
                source_counts=matrix,rank_loads=[sum(matrix[src][j] for src in range(self.world)
                    for j in owner_ids(e,self.world,r)) for r in range(self.world)]))
        send_splits=[sum(matrix[self.rank][j] for j in owner_ids(e,self.world,r)) for r in range(self.world)]
        recv_splits=[sum(matrix[src][j] for j in owned) for src in range(self.world)]
        # Sort by destination, then logical expert. Stable order preserves top-k rows.
        keys=(local_ids.flatten()%self.world)*e+local_ids.flatten()
        order=keys.argsort(stable=True)
        send=local[order//k].contiguous()
        receive=hidden.new_empty((sum(recv_splits),d))
        if events:events[1].record()
        dist.all_to_all_single(receive,send,recv_splits,send_splits)
        if events:events[2].record()
        recv_experts=[j for src in range(self.world) for j in owned for _ in range(matrix[src][j])]
        local_order=torch.tensor(recv_experts,dtype=torch.int64,device=hidden.device).argsort(stable=True)
        grouped=receive[local_order].contiguous()
        expert_counts=[sum(matrix[src][j] for src in range(self.world)) for j in owned]
        if events:events[3].record()
        computed=self.expert_fns[site](grouped,expert_counts) if len(grouped) else grouped
        if self.trace and site==self.capture_site:
            self.captured.append(dict(loop=loop,x=grouped.detach().clone(),counts=expert_counts,
                send=send.detach().clone(),receive_shape=list(receive.shape),send_splits=send_splits,recv_splits=recv_splits))
            self.captured=self.captured[-2:]
        if events:events[4].record()
        receive_order=computed[local_order.argsort(stable=True)].contiguous()
        returned=hidden.new_empty(send.shape)
        dist.all_to_all_single(returned,receive_order,send_splits,recv_splits)
        if events:events[5].record()
        row_outputs=returned[order.argsort(stable=True)].reshape(len(local),k,d)
        local_result=(row_outputs*local_weights.unsqueeze(-1)).sum(1)
        restored=torch.zeros_like(flat)
        restored[self.rank::self.world]=local_result
        dist.all_reduce(restored)
        if events:
            events[6].record()
            sample=torch.linspace(0,b*l-1,min(b*l,32),device=ids.device).long()
            self.records.append(dict(site=site,loop=loop,recurrent=site in self.a.recurrent,
                rank=self.rank,batch=b,length=l,events=events,
                expert_counts=[sum(matrix[src][j] for src in range(self.world)) for j in range(e)],
                local_expert_counts=expert_counts,source_counts=matrix[self.rank],
                rank_loads=[sum(matrix[src][j] for src in range(self.world) for j in owner_ids(e,self.world,r)) for r in range(self.world)],
                send_rows=sum(send_splits),recv_rows=sum(recv_splits),send_splits=send_splits,recv_splits=recv_splits,
                remote_bytes=2*d*sum(n for r,n in enumerate(send_splits) if r!=self.rank),
                restoration_bytes=restored.numel()*restored.element_size(),
                requests=self.request_ids,sampled_positions=sample.detach().cpu().tolist(),
                host_ready_s=host_ready,clock='GPU_LOCAL_RANK_FORWARD_RELATIVE',
                sampled_ids=ids.reshape(-1,k)[sample].detach().cpu().tolist(),
                request_expert_counts=[torch.bincount(ids[i].flatten(),minlength=e).detach().cpu().tolist() for i in range(b)]))
        counts_all=None if self.a.name.startswith('ifm') else torch.bincount(ids.flatten(),minlength=e)
        return restored.reshape_as(hidden),counts_all

    def resolve(self):
        torch.cuda.synchronize()
        rows=[]
        for rec in self.records:
            event=rec['events']
            r={k:v for k,v in rec.items() if k!='events'}
            for index,name in enumerate(['metadata_pack','dispatch','receive_grouping','expert','combine','restore']):
                r[name+'_ms']=event[index].elapsed_time(event[index+1])
            r['total_span_ms']=event[0].elapsed_time(event[-1])
            r['ready_forward_ms']=self.origin.elapsed_time(event[0])
            r['complete_forward_ms']=self.origin.elapsed_time(event[-1])
            r['evidence']='LIVE_CUDA_EVENT_API_SPANS_NOT_KERNEL_ONLY'
            rows.append(r)
        self.records=[]
        return rows
