"""Deferred CUDA events and compact native target-only routing capture.

No per-layer synchronization, CPU tensor downloads, hidden dumps or routing
changes. All drains run by an explicit RPC after a completed serving wave.
Clean/trace/bypass are separate runs. Draft dense layers are not instrumented.
"""
import functools
import json
import os
import re
import threading
from pathlib import Path

import torch

TLS = threading.local()
FRAMES = []
POOL = {}
INSTALLED = False
OWNER_MAP = None


def control():
    return json.loads(Path(os.environ['SD_EP_CONTROL']).read_text())


def events(key, n):
    if key not in POOL:
        POOL[key] = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    return POOL[key]


def flush(worker):
    """Called outside the measured wave; worker argument is vLLM RPC contract."""
    import numpy as np
    from vllm.distributed import get_ep_group,get_tp_group,get_dp_group,get_pp_group
    rank = get_ep_group().rank_in_group
    root = Path(os.environ['SD_EP_OUTPUT'])
    torch.cuda.synchronize()
    if OWNER_MAP is not None:
        owner=OWNER_MAP.cpu().tolist()
        expected=[-1]*128
        expected[rank*32:(rank+1)*32]=list(range(32))
        assert owner==expected, 'runtime ownership is not linear contiguous32'
        proof=root/f'ownership.rank{rank}.json'
        if not proof.exists():
            proof.write_text(json.dumps(dict(ep_rank=rank,expert_map=owner,
                experts_per_rank=32,verified_loaded_dtype='torch.bfloat16',
                worker_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                logical_cuda_device=torch.cuda.current_device(),
                worker_pid=os.getpid(),
                actual_cuda_uuid=str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),'uuid','UNAVAILABLE')),
                tp_global_ranks=get_tp_group().ranks,dp_global_ranks=get_dp_group().ranks,
                ep_global_ranks=get_ep_group().ranks,pp_global_ranks=get_pp_group().ranks))+'\n')
    summaries = []
    routes = {}
    for frame in FRAMES:
        out = {k:v for k,v in frame.items() if k not in ('layers','events','sampled','draft_events')}
        if frame.get('sampled') is not None:
            out['sampled_tokens'] = frame['sampled'].cpu().tolist()
        if frame.get('events'):
            a,b = frame['events'];out['target_forward_ms']=a.elapsed_time(b)
        if frame.get('draft_events'):
            a,b = frame['draft_events'];out['draft_ms']=a.elapsed_time(b)
        layers=[]
        for row in frame['layers']:
            record = {k:v for k,v in row.items() if k not in ('ev','route')}
            ev=row['ev']
            record.update(dispatch_ms=ev[0].elapsed_time(ev[1]),expert_ms=ev[1].elapsed_time(ev[2]),
                          combine_ms=ev[2].elapsed_time(ev[3]),moe_ms=ev[0].elapsed_time(ev[3]),
                          router_selection_ms=ev[4].elapsed_time(ev[5]))
            if row.get('route') is not None:
                ids,weights=row['route']
                key=f"w{frame['wave']}_f{frame['forward']}_l{row['layer']}"
                routes[key+'_ids']=ids.cpu().numpy()
                routes[key+'_weights']=weights.cpu().numpy()
                record['route_key']=key
            layers.append(record)
        out['layers']=layers
        summaries.append(out)
    with (root/f'frames.rank{rank}.jsonl').open('a') as f:
        for row in summaries:f.write(json.dumps(row, separators=(',',':'))+'\n')
    if routes:
        wave=summaries[0]['wave']
        np.savez_compressed(root/f'routes.wave{wave}.rank{rank}.npz',**routes)
    FRAMES.clear()
    return dict(rank=rank,frames=len(summaries))


def install():
    global INSTALLED
    if INSTALLED:return
    INSTALLED=True
    from vllm.distributed import get_ep_group,get_tp_group,get_dp_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl as Kernel
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
    Worker.sd_ep_flush=flush
    oinit,olayer=Qwen3MoeDecoderLayer.__init__,Qwen3MoeDecoderLayer.forward
    oapply,opre,oexp,ofin=Kernel.apply,Kernel._prepare,Kernel._fused_experts,Kernel._finalize
    oselect=BaseRouter.select_experts
    oexecute,osample,odraft=GPUModelRunner.execute_model,GPUModelRunner._sample,GPUModelRunner.propose_draft_token_ids

    def init(self,*args,**kw):
        oinit(self,*args,**kw)
        match=re.search(r'layers\.(\d+)',str(kw.get('prefix',args[1] if len(args)>1 else '')))
        self._sdep_layer=int(match[1]) if match else -1

    def layer(self,*args,**kw):
        TLS.layer=self._sdep_layer
        if TLS.layer==0:
            c=control()
            if getattr(TLS,'wave',None)!=c.get('wave'):
                TLS.wave=c.get('wave');TLS.forward=-1
            TLS.forward=getattr(TLS,'forward',-1)+1
            meta=get_forward_context().dp_metadata
            dp=meta.num_tokens_across_dp_cpu.tolist() if meta is not None else []
            enabled=c.get('lane') in ('clean','trace') and dp and dp[0]>0
            TLS.frame=None
            if enabled:
                frame={**c,'forward':TLS.forward,'ep_rank':get_ep_group().rank_in_group,
                       'tp_rank':get_tp_group().rank_in_group,'dp_rank':get_dp_group().rank_in_group,
                       'coordinated_dp_tokens':dp,'worker_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'logical_cuda_device':torch.cuda.current_device(),'layers':[],
                       'scheduler':getattr(TLS,'scheduler',None)}
                frame['events']=events(('model',TLS.forward),2)
                frame['events'][0].record()
                TLS.frame=frame;FRAMES.append(frame)
        out=olayer(self,*args,**kw)
        if TLS.layer==47 and getattr(TLS,'frame',None):
            TLS.frame['events'][1].record()
        return out

    def select(self,*args,**kw):
        frame=getattr(TLS,'frame',None)
        pair=events(('router',TLS.forward,TLS.layer),2) if frame else None
        if pair:pair[0].record()
        out=oselect(self,*args,**kw)
        if pair:pair[1].record();TLS.router=pair
        return out

    def apply(self,*args,**kw):
        global OWNER_MAP
        frame=getattr(TLS,'frame',None)
        if not frame:return oapply(self,*args,**kw)
        backend=type(self.prepare_finalize).__name__
        if backend!='DeepEPHTPrepareAndFinalize':raise RuntimeError('silent fallback '+backend)
        values=dict(zip(['hidden_states','w1','w2','topk_ids','topk_weights'],args));values.update(kw)
        if OWNER_MAP is None:
            OWNER_MAP=kw.get('expert_map')
            if OWNER_MAP is None:raise RuntimeError('runtime expert owner map unavailable')
        ids=values['topk_ids'];weights=values['topk_weights'];w1,w2=values['w1'],values['w2']
        assert tuple(w1.shape)==(32,1536,2048) and tuple(w2.shape)==(32,2048,768)
        assert ids.shape[1]==8 and w1.dtype==torch.bfloat16
        n=frame['coordinated_dp_tokens'][frame['dp_rank']]
        local=ids.shape[0]
        # This campaign has exactly one active request on DP0. DP1 executes
        # runtime dummy rows; never count those as replicated user requests.
        valid=max(0,min(local,n-frame['tp_rank']*local)) if frame['dp_rank']==0 else 0
        ev=events(('stage',TLS.forward,TLS.layer),4)+getattr(TLS,'router',events(('router-fallback',TLS.forward,TLS.layer),2))
        row=dict(layer=TLS.layer,source_rows=local,valid_source_rows=valid,
                 prepare_backend=backend,expert_backend=type(self.fused_experts).__name__,
                 expert_shapes=[list(w1.shape),list(w2.shape)],
                 expert_weight_bytes_each=(w1.numel()*w1.element_size()+w2.numel()*w2.element_size())//32,
                 ev=ev,route=None)
        # Routes copied only in diagnostic lane, and downloaded after the wave.
        if frame['lane']=='trace':row['route']=(ids.detach().clone(),weights.detach().clone())
        TLS.stage=row
        try:
            out=oapply(self,*args,**kw);frame['layers'].append(row);return out
        finally:TLS.stage=None

    def prepare(self,*args,**kw):
        row=getattr(TLS,'stage',None)
        if row:row['ev'][0].record()
        out=opre(self,*args,**kw)
        if row:row['ev'][1].record()
        return out

    def expert(self,*args,**kw):
        out=oexp(self,*args,**kw);row=getattr(TLS,'stage',None)
        if row:row['ev'][2].record()
        return out

    def combine(self,*args,**kw):
        out=ofin(self,*args,**kw);row=getattr(TLS,'stage',None)
        if row:row['ev'][3].record()
        return out

    def execute(self,scheduler_output,*args,**kw):
        TLS.scheduler=dict(scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
                           draft_ids=dict(scheduler_output.scheduled_spec_decode_tokens))
        return oexecute(self,scheduler_output,*args,**kw)

    def sample(self,*args,**kw):
        frame=getattr(TLS,'frame',None)
        # Include final norm/LM projection, stop before rejection sampling.
        if frame is not None:frame['events'][1].record()
        out=osample(self,*args,**kw)
        if frame is not None:frame['sampled']=out.sampled_token_ids.detach().clone()
        return out

    def draft(self,*args,**kw):
        frame=getattr(TLS,'frame',None)
        pair=events(('draft',TLS.forward),2) if frame else None
        if pair:pair[0].record()
        out=odraft(self,*args,**kw)
        if pair:pair[1].record();frame['draft_events']=pair
        return out

    Qwen3MoeDecoderLayer.__init__,Qwen3MoeDecoderLayer.forward=init,layer
    BaseRouter.select_experts=select
    Kernel.apply,Kernel._prepare,Kernel._fused_experts,Kernel._finalize=apply,prepare,expert,combine
    GPUModelRunner.execute_model,GPUModelRunner._sample,GPUModelRunner.propose_draft_token_ids=execute,sample,draft
