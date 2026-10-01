"""CUDA-only component lane: no router-mask construction or route download.

Uses the same installed DeepEP prepare/expert/finalize boundaries as the prior
instrumentation. CPU count metadata already produced by DeepEP is serialized
only after all 48 layers. No profiler streams or cross-device clock subtraction.
"""
import functools
import json
import os
import re
from pathlib import Path

import torch

from poc_mllm_ep_communication_motivation import instrumentation as base

PENDING=[]
INVOCATION=-1
WAVE=None


def drain():
    if not PENDING:
        return
    torch.cuda.synchronize()
    root=Path(os.environ['MLLM_COMM_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
    with (root/f'stage.pid{os.getpid()}.jsonl').open('a') as f:
        for record,events,meta in PENDING:
            hist=[] if meta is None or meta.expert_num_tokens_cpu is None else meta.expert_num_tokens_cpu.tolist()
            record.update(local_expert_histogram=hist,local_token_expert_pairs=sum(hist),
                          local_distinct_active_experts=sum(x>0 for x in hist))
            record.update(dispatch_ms=events[0].elapsed_time(events[1]),
                          expert_execution_ms=events[1].elapsed_time(events[2]),
                          combine_ms=events[2].elapsed_time(events[3]),
                          moe_ms=events[0].elapsed_time(events[3]))
            f.write(json.dumps(record,separators=(',',':'))+'\n')
    PENDING.clear()


def install():
    from vllm.distributed import get_dp_group,get_ep_group,get_tp_group
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl as Kernel
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original_init=Qwen3MoeDecoderLayer.__init__
    original_layer=Qwen3MoeDecoderLayer.forward
    original_execute=GPUModelRunner.execute_model
    original_apply=Kernel.apply
    original_prepare=Kernel._prepare
    original_experts=Kernel._fused_experts
    original_finalize=Kernel._finalize

    def layer_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        prefix=str(kwargs.get('prefix',args[1] if len(args)>1 else ''))
        match=re.search(r'layers\.(\d+)',prefix)
        self._native_trace_layer=int(match.group(1)) if match else -1

    def layer(self,*args,**kwargs):
        prior=getattr(base._TLS,'layer',-1)
        base._TLS.layer=self._native_trace_layer
        try:return original_layer(self,*args,**kwargs)
        finally:base._TLS.layer=prior

    def execute(self,*args,**kwargs):
        global INVOCATION,WAVE
        base._TLS.light_control=base._control()
        wave=base._TLS.light_control.get('wave')
        if wave!=WAVE:WAVE=wave;INVOCATION=-1
        INVOCATION+=1
        try:return original_execute(self,*args,**kwargs)
        finally:
            drain()
            base._TLS.light_control=None

    def apply(self,*args,**kwargs):
        from .native import TLS
        control=getattr(base._TLS,'light_control',None)
        if not control or not control.get('capture_stage',False):
            return original_apply(self,*args,**kwargs)
        states=getattr(TLS,'states',None) or []
        record={'evidence':'MEASURED_TRUE_EP4_DEEPEP_CUDA_EVENTS',
                'lane':'LOW_OVERHEAD_COMPONENT_TRACE',
                'wave':control['wave'],'replicate':control['replicate'],
                'condition':control['condition'],'outer_iteration':control['outer_iteration'],
                'iteration':INVOCATION,
                'layer':getattr(base._TLS,'layer',-1),
                'ep_rank':get_ep_group().rank_in_group,'tp_rank':get_tp_group().rank_in_group,
                'dp_rank':get_dp_group().rank_in_group,
                'scheduled_tokens':sum(s['count'] for s in states),
                'injected_rows':sum(s['count'] for s in states if s['injection'] and s['computed']<s['prompt']),
                'global_prefill_rows':getattr(TLS,'global_prefill_rows',0),
                'prefill_offset':getattr(TLS,'prefill_offset',-1),
                'global_decode_step':getattr(TLS,'global_decode_step',-1),
                'native_lp_active_prefill':getattr(TLS,'active_prefill',None),
                'native_lp_stage':getattr(TLS,'lp_stage',None),
                'prepare_finalize_backend':type(self.prepare_finalize).__name__,
                'expert_backend':type(self.fused_experts).__name__,
                'topology':{'tp':2,'dp':2,'ep':4}}
        events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        base._TLS.stage=(record,events)
        base._TLS.light_meta=None
        try:
            result=original_apply(self,*args,**kwargs)
            PENDING.append((record,events,base._TLS.light_meta))
            return result
        finally:base._TLS.stage=None

    def prepare(self,*args,**kwargs):
        stage=getattr(base._TLS,'stage',None)
        if stage:stage[1][0].record()
        result=original_prepare(self,*args,**kwargs)
        if stage:stage[1][1].record()
        return result

    def experts(self,*args,**kwargs):
        result=original_experts(self,*args,**kwargs)
        stage=getattr(base._TLS,'stage',None)
        if stage:
            stage[1][2].record()
            base._TLS.light_meta=kwargs.get('expert_tokens_meta',args[12] if len(args)>12 else None)
        return result

    def finalize(self,*args,**kwargs):
        result=original_finalize(self,*args,**kwargs)
        stage=getattr(base._TLS,'stage',None)
        if stage:stage[1][3].record()
        return result

    Qwen3MoeDecoderLayer.__init__=layer_init
    Qwen3MoeDecoderLayer.forward=layer
    GPUModelRunner.execute_model=execute
    Kernel.apply=apply
    Kernel._prepare=prepare
    Kernel._fused_experts=experts
    Kernel._finalize=finalize
