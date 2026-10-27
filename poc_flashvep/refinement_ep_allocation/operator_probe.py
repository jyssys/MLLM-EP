"""Physical MoE microbenchmarks on captured Mini activations and native weights.

These are operator timings, not request speedups. Precomputed masks are an oracle.
"""
import random
import time
import torch
import torch.distributed as dist
from .common import append,write
from .semantic_emulator import select_edges

class OperatorProbe:
    def __init__(self,model,rank,destination):
        self.rank=rank
        self.destination=destination
        self.active=False
        self.mode='FULL'
        self.captured={}
        self.masks={}
        self.events=[]
        self.geometry={}
        self.blocks={}
        for layer in model.model.layers:
            block=layer.mlp
            if not hasattr(block,'deepep_dispatcher') or block.layer_id not in (1,19):
                continue
            self.blocks[block.layer_id]=block
            original=block.forward
            def forward(hidden,*a,_original=original,_layer=block.layer_id,**kw):
                shape=hidden.numel()//2048
                if not self.active and shape in (32,64,128) and (_layer,shape) not in self.captured:
                    self.captured[_layer,shape]=hidden.detach().clone()
                return _original(hidden,*a,**kw)
            block.forward=forward
            original_dispatch=block.deepep_dispatcher.dispatch
            def dispatch(*a,_original=original_dispatch,_layer=block.layer_id,**kw):
                if not self.active:
                    return _original(*a,**kw)
                ids=kw['topk_idx']
                weights=kw['topk_weights']
                key=(_layer,ids.shape[0],self.mode)
                if self.record_stages:
                    start=torch.cuda.Event(enable_timing=True)
                    end=torch.cuda.Event(enable_timing=True)
                    start.record()
                if self.mode!='FULL':
                    fraction=.05 if self.mode.endswith('05') else .20
                    if key not in self.masks:
                        mask=select_edges(weights,torch.zeros(len(ids),device=ids.device,dtype=torch.bool),
                            torch.zeros(len(ids),device=ids.device),'P2',fraction,self.rank,_layer,1)
                        self.masks[key]=mask
                    mask=self.masks[key]
                    ids=ids.masked_fill(mask,-1)
                    weights=weights.masked_fill(mask,0.)
                    self.keepalive.extend((ids,weights))
                    kw=dict(kw,topk_idx=ids,topk_weights=weights)
                if self.record_stages:
                    end.record()
                    self.events.append(('mask_materialization',start,end))
                result=_original(*a,**kw)
                if self.capture_geometry:
                    self.geometry[_layer,ids.shape[0]*4,self.mode]=dict(
                        source_rows=ids.shape[0],received_rows=result[0].shape[0],
                        active_source_edges=int((ids>=0).sum()),
                        current_weight_stride=list(weights.stride()),
                        active_received_edges=int((result[1]>=0).sum()),
                        source_ids=ids.detach().cpu().tolist(),received_ids=result[1].detach().cpu().tolist())
                return result
            block.deepep_dispatcher.dispatch=dispatch
            for label,owner,attribute in [('dispatch',block.deepep_dispatcher,'dispatch'),
                    ('expert',block.experts.quant_method,'apply'),
                    ('combine',block.deepep_dispatcher._normal_dispatcher,'_combine_core')]:
                original_stage=getattr(owner,attribute)
                def measured(*a,_original=original_stage,_label=label,**kw):
                    if not self.active or not self.record_stages:
                        return _original(*a,**kw)
                    start=torch.cuda.Event(enable_timing=True)
                    end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    result=_original(*a,**kw)
                    if _label=='combine':
                        result[1].current_stream_wait()
                    end.record()
                    self.events.append((_label,start,end))
                    return result
                setattr(owner,attribute,measured)

    def run(self,repetitions=30):
        assert self.captured,'No real activation geometry was captured'
        self.active=True
        order=[]
        for lane in ('CLEAN_OPERATOR','REGION_OBSERVER'):
            for key in sorted(self.captured):
                for repeat in range(repetitions+3):
                    methods=['FULL','PRECOMPUTED_ROUTER_05','PRECOMPUTED_ROUTER_20']
                    random.Random(20261028+repeat).shuffle(methods)
                    order.extend((lane,key,repeat,method) for method in methods)
        for lane,(layer,shape),repeat,method in order:
            self.record_stages=lane=='REGION_OBSERVER'
            self.mode=method
            self.capture_geometry=repeat==0
            self.events=[]
            self.keepalive=[]
            dist.barrier()
            torch.cuda.synchronize(self.rank)
            start=time.perf_counter()
            with torch.inference_mode():
                output=self.blocks[layer](self.captured[layer,shape])
            torch.cuda.synchronize(self.rank)
            elapsed=1000*(time.perf_counter()-start)
            assert torch.isfinite(output).all()
            if repeat>=3:
                append(self.destination/f'operator_samples.rank{self.rank}.jsonl',dict(
                    evidence='MEASURED_GPU_OPERATOR_NOT_REQUEST_E2E',rank=self.rank,layer=layer,
                    full_rows=shape,repeat=repeat,method=method,wall_ms=elapsed,
                    timing_lane=lane,
                    stages_ms={name:float(a.elapsed_time(b)) for name,a,b in self.events},
                    geometry=self.geometry[layer,shape,method],
                    boundary='Native routed MoE, shared expert and source-row all-gather. Excludes attention and model head.',
                    selection='OFFLINE_PRECOMPUTED_CURRENT_ROUTER_MASK. Optimistic preparation cost, not an online policy.',
                    padding_rule='Native BF16 Triton expert tiles. IDs -1 remove actual dispatch and expert assignments.'))
        self.active=False
        write(self.destination/f'operator_complete.rank{self.rank}.json',dict(
            geometries=[list(k) for k in sorted(self.captured)],repetitions=repetitions,
            interpretation='Standalone operator measurements on real model activations and original BF16 expert weights.'))
