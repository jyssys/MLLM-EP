"""Pooled same-device CUDA-event boundaries, with one drain per 48-layer forward.

Clean lane never clones routes/downloads tensors. Internal lane separately
captures route tensors after the measured apply and downloads after forward.
Buffer API timers wait for their own completion event; they are composite
protocol intervals, NOT pure network-transfer timers.
"""
import functools
import json
import os
import re
import threading
from pathlib import Path

import torch
from .selection import is_native_prefill

TLS = threading.local()
PENDING = []
POOL = {}
INSTALLED = False


def control():
    path = Path(os.environ['ASYMMETRY_CONTROL'])
    return json.loads(path.read_text()) if path.exists() else {}


def events(key, n):
    if key not in POOL:
        POOL[key] = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for event in POOL[key]:
            event.record()
    return POOL[key]


def drain():
    if not PENDING:
        return
    torch.cuda.synchronize()
    root = Path(os.environ['ASYMMETRY_OUTPUT'])
    with (root/f'stage.pid{os.getpid()}.jsonl').open('a') as file:
        for record, ev, meta, routes, sub in PENDING:
            record.update(dispatch_ms=ev[0].elapsed_time(ev[1]),
                expert_ms=ev[1].elapsed_time(ev[2]), combine_ms=ev[2].elapsed_time(ev[3]),
                moe_ms=ev[0].elapsed_time(ev[3]),
                router_to_dispatch_ms=ev[4].elapsed_time(ev[0]))
            hist = [] if meta is None or meta.expert_num_tokens_cpu is None else meta.expert_num_tokens_cpu.tolist()
            record['local_expert_histogram'] = hist
            record['subcomponents'] = {name: {'ms': pair[0].elapsed_time(pair[1]),
                'label': 'DIRECTLY_TIMED', 'scope': 'COMPOSITE_API_NOT_PURE_DATA_MOVEMENT'} for name, pair in sub.items()}
            if routes is not None:
                h, ids, weights, expert_map = routes
                root_capture = root/'captures'; root_capture.mkdir(exist_ok=True)
                name = f"wave{record['wave']}.rank{record['ep_rank']}.layer{record['layer']}.pt"
                # Hidden only layer0/24/47, one internal repeat per arm/size.
                payload = {'ids': ids.cpu(), 'weights': weights.cpu(), 'record': record}
                if expert_map is not None:
                    payload['expert_map'] = expert_map.cpu()
                if h is not None:
                    payload['hidden'] = h.cpu()
                torch.save(payload, root_capture/name)
                record['capture'] = str(root_capture/name)
            file.write(json.dumps(record, separators=(',', ':'))+'\n')
    PENDING.clear()


def install():
    global INSTALLED
    if INSTALLED:
        return
    INSTALLED = True
    from vllm.distributed import get_ep_group, get_tp_group, get_dp_group
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl as Kernel
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    import deep_ep
    original_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_select = BaseRouter.select_experts
    original_apply = Kernel.apply
    original_prepare, original_experts, original_finalize = Kernel._prepare, Kernel._fused_experts, Kernel._finalize

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get('prefix', args[1] if len(args)>1 else ''))
        match = re.search(r'layers\.(\d+)', prefix)
        self._as_layer = int(match.group(1)) if match else -1

    def layer(self, *args, **kwargs):
        prev = getattr(TLS, 'layer', -1); TLS.layer = self._as_layer
        # Read control once per forward, including external-DP dummy ranks.
        if self._as_layer == 0:
            TLS.control = control()
            from vllm.forward_context import get_forward_context
            metadata = get_forward_context().dp_metadata
            if metadata is None or metadata.num_tokens_across_dp_cpu.device.type != 'cpu':
                raise RuntimeError('CPU DP coordination metadata unavailable')
            TLS.dp_tokens = metadata.num_tokens_across_dp_cpu.tolist()
            TLS.native_prefill = is_native_prefill(TLS.dp_tokens,TLS.control.get('prompt_tokens',-1))
            TLS.profile_active = (os.environ.get('ASYMMETRY_NSYS')=='1' and TLS.native_prefill
                                  and TLS.control.get('repeat')==0)
            if TLS.profile_active:
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push('ASYMMETRY_NATIVE_'+TLS.control['arm'])
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            TLS.layer = prev
            if self._as_layer == 47:
                drain()
                if getattr(TLS,'profile_active',False):
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.profiler.stop()

    def select(self, *args, **kwargs):
        result = original_select(self, *args, **kwargs)
        c = getattr(TLS, 'control', {})
        if c.get('lane') in ('clean', 'internal') and getattr(TLS,'native_prefill',False):
            e = events(('router', getattr(TLS, 'layer', -1)), 1)[0]
            e.record(); TLS.router = e
        return result

    def apply(self, *args, **kwargs):
        c = getattr(TLS, 'control', {})
        layer_id = getattr(TLS, 'layer', -1)
        # Warm all timing event objects on unmeasured startup/warmup calls.
        ev = events(('stage', id(self), layer_id), 5)
        if c.get('lane') not in ('clean','internal') or not getattr(TLS,'native_prefill',False):
            return original_apply(self, *args, **kwargs)
        if type(self.prepare_finalize).__name__ != 'DeepEPHTPrepareAndFinalize':
            raise RuntimeError('silent backend fallback: '+type(self.prepare_finalize).__name__)
        rank = get_ep_group().rank_in_group
        record = {**c, 'layer': layer_id, 'ep_rank': rank,
            'coordinated_dp_tokens': TLS.dp_tokens,
            'logical_cuda_device': torch.cuda.current_device(),
            'worker_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'tp_rank': get_tp_group().rank_in_group, 'dp_rank': get_dp_group().rank_in_group,
            'prepare_backend': type(self.prepare_finalize).__name__, 'expert_backend': type(self.fused_experts).__name__,
            'timing_label': 'DIRECTLY_TIMED', 'timing_scope': 'LOCAL_PREPARE_EXPERT_FINALIZE_COMPOSITE'}
        vals = dict(zip(['hidden_states','w1','w2','topk_ids','topk_weights'],args)); vals.update(kwargs)
        record['loaded_expert_shapes'] = [list(vals[k].shape) for k in ('w1','w2')]
        record['loaded_expert_dtype'] = str(vals['w1'].dtype)
        record['expert_weight_bytes_each'] = sum(vals[k].numel()*vals[k].element_size() for k in ('w1','w2'))//32
        if vals['w1'].shape[0] != 32 or vals['topk_ids'].shape[1] != 8:
            raise RuntimeError('wrong expert bank or top-k')
        TLS.stage = [record, ev, None, None, {}]
        captured = None
        if c['lane'] == 'internal' and c.get('capture_routes'):
            names = ['hidden_states','w1','w2','topk_ids','topk_weights']
            values = dict(zip(names,args)); values.update(kwargs)
            h = values['hidden_states']
            ids, w = values['topk_ids'], values['topk_weights']
            record['source_rows'] = ids.shape[0]
            captured = (h.detach().clone() if layer_id in (0,24,47) else None,
                        ids.detach().clone(), w.detach().clone(),
                        kwargs['expert_map'].detach().clone() if kwargs.get('expert_map') is not None else None)
        router = getattr(TLS,'router',None)
        # ev4 is a local router completion marker (no cross-rank timestamp subtraction).
        if router is not None:
            # The completion event is copied as a reference, rather than rerecorded.
            ev = [*ev[:4], router]; TLS.stage[1] = ev
        else:
            ev[4].record()
        try:
            result = original_apply(self, *args, **kwargs)
            TLS.stage[3] = captured
            PENDING.append(tuple(TLS.stage))
            return result
        finally:
            TLS.stage = None

    def prepare(self,*args,**kwargs):
        stage = getattr(TLS,'stage',None)
        if stage: stage[1][0].record()
        result = original_prepare(self,*args,**kwargs)
        if stage:
            stage[1][1].record()
            stage[2] = result[2]
        return result

    def experts(self,*args,**kwargs):
        result = original_experts(self,*args,**kwargs)
        stage = getattr(TLS,'stage',None)
        if stage: stage[1][2].record()
        return result

    def finalize(self,*args,**kwargs):
        result = original_finalize(self,*args,**kwargs)
        stage = getattr(TLS,'stage',None)
        if stage: stage[1][3].record()
        return result

    def api_wrapper(name, original):
        @functools.wraps(original)
        def wrapped(self,*args,**kwargs):
            stage = getattr(TLS,'stage',None)
            if stage is None or stage[0]['lane'] != 'internal':
                return original(self,*args,**kwargs)
            ev = events(('api', name, stage[0]['layer']),2)
            ev[0].record()
            torch.cuda.nvtx.range_push('ASYMMETRY_'+name)
            try:
                out = original(self,*args,**kwargs)
                if name in ('dispatch','combine') and out[-1].event is not None:
                    out[-1].current_stream_wait()
                ev[1].record()
                stage[4][name] = ev
                return out
            finally:
                torch.cuda.nvtx.range_pop()
        return wrapped

    Qwen3MoeDecoderLayer.__init__, Qwen3MoeDecoderLayer.forward = init, layer
    BaseRouter.select_experts = select
    Kernel.apply, Kernel._prepare, Kernel._fused_experts, Kernel._finalize = apply, prepare, experts, finalize
    for name in ('get_dispatch_layout','dispatch','combine'):
        setattr(deep_ep.Buffer, name, api_wrapper(name,getattr(deep_ep.Buffer,name)))
    if os.environ.get('ASYMMETRY_ENVELOPE')=='1':
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel
        def envelope(original):
            @functools.wraps(original)
            def model_forward(self,*args,**kwargs):
                from vllm.forward_context import get_forward_context
                c=control();meta=get_forward_context().dp_metadata
                selected=meta is not None and is_native_prefill(meta.num_tokens_across_dp_cpu.tolist(),c.get('prompt_tokens',-1))
                if not selected:return original(self,*args,**kwargs)
                ev=events(('common_model_envelope',id(self)),2)
                ev[0].record();out=original(self,*args,**kwargs);ev[1].record()
                torch.cuda.synchronize()
                record={**c,'ep_rank':get_ep_group().rank_in_group,'model_forward_ms':ev[0].elapsed_time(ev[1]),
                    'scope':'language_model_forward inclusive instrumentation drain; same two events and completion sync in bypass/clean'}
                root=Path(os.environ['ASYMMETRY_OUTPUT'])
                with (root/f'envelope.rank{record["ep_rank"]}.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
                return out
            return model_forward
        for model_class in (Qwen3MoeModel,Qwen3MoeLLMModel):
            model_class.forward=envelope(model_class.forward)
