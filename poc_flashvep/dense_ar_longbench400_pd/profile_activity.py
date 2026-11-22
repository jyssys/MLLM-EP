"""Collect GPU kernels after warmup and correlate eager operators explicitly."""
import ctypes
import json
import time
from .common import ROOT,write,digest

CURRENT=None


class Activity:
    def __init__(self,folder,include_copies=False,graph_enclosures=False,runtime_api=False):
        global CURRENT
        self.folder=folder;self.include_copies=include_copies;self.enabled=False;self.started=False
        cfg=json.loads((folder.parent/'configuration.json').read_text())
        self.delay=float(cfg.get('activity_delay_seconds',0));self.duration=cfg.get('activity_ROI_seconds')
        self.armed=False;self.enable_at=None;self.disable_at=None;self.roi={}
        self.steps=[];self.ops=[];self.step=None;self.counter=0;self.op_counter=1000000
        namespace=cfg.get('collector_namespace','correlated_collector_v1')
        assert namespace in ('correlated_collector_v1','correlated_collector_api_v2','kernel_only_collector_v3')
        self.kernel_only=namespace=='kernel_only_collector_v3'
        assert not self.kernel_only or not (runtime_api or graph_enclosures or cfg.get('operator_correlation'))
        build=json.loads((ROOT/'references'/namespace/'build.json').read_text())
        for key in ('source','library','CUPTI_library'):assert digest(__import__('pathlib').Path(build[key]))==build[key+'_SHA256']
        self.lib=ctypes.CDLL(build['library'])
        self.runtime_api=runtime_api;self.sdk=ctypes.CDLL(build['CUPTI_library'])
        assert not graph_enclosures or not runtime_api
        self.lib.activity_prepare.argtypes=[ctypes.c_size_t]
        self.lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        self.lib.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
        self.lib.activity_push_id.argtypes=[ctypes.c_uint64]
        self.lib.activity_pop_id.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        for key in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes'):
            getattr(self.lib,'activity_'+key).restype=ctypes.c_uint64
        assert self.lib.activity_graph_mode(int(graph_enclosures))==0
        assert self.lib.activity_prepare(1024*1024**2)==0
        write(folder/'activity_provenance.json',dict(build=build,include_copies=include_copies,
            enabled_after_warmup=True,graph_trace_activity=graph_enclosures,runtime_API_activity=runtime_api,
            deferred_ROI_delay_seconds=self.delay,deferred_ROI_duration_seconds=self.duration,
            kernel_only=self.kernel_only,
            scope='Actual kernel activity and optional copies. Kernel-only mode does not collect APIs or external correlation. '
                'Other modes collect driver/external correlation. Observer overhead and attribution coverage require separate gates.'))
        CURRENT=self

    def timestamp(self):
        value=ctypes.c_uint64();before=time.monotonic_ns()
        assert self.lib.activity_timestamp(ctypes.byref(value))==0
        return dict(cupti_ns=value.value,host_lower_ns=before,host_upper_ns=time.monotonic_ns())

    def begin(self):
        if self.delay:
            import torch
            sync=time.monotonic();torch.cuda.synchronize();self.roi['start_sync_host_ms']=1000*(time.monotonic()-sync)
        assert self.lib.activity_enable(int(self.include_copies))==0
        if self.runtime_api:assert self.sdk.cuptiActivityEnable(5)==0
        self.started=self.enabled=True;self.roi['left']=self.timestamp()
        if self.duration is not None:self.disable_at=time.monotonic()+float(self.duration)

    def stamp(self):
        if not self.armed:
            self.armed=True;self.enable_at=time.monotonic()+self.delay
            if not self.delay:self.begin()
        return self.timestamp()

    def refresh(self):
        if self.armed and not self.started and time.monotonic()>=self.enable_at:self.begin()
        if self.enabled and self.disable_at is not None and time.monotonic()>=self.disable_at:self.pause()

    def push(self,identifier):assert self.lib.activity_push_id(identifier)==0

    def pop(self,identifier):
        value=ctypes.c_uint64();assert self.lib.activity_pop_id(ctypes.byref(value))==0 and value.value==identifier

    def pause(self):
        if self.enabled:
            if self.runtime_api:assert self.sdk.cuptiActivityDisable(5)==0
            assert self.lib.activity_disable()==0;self.enabled=False
            self.roi['right']=self.timestamp()

    def close(self):
        self.pause();code=int(self.lib.activity_end(str(self.folder/'kernels.csv').encode(),0,2**64-1))
        counters={key:int(getattr(self.lib,'activity_'+key)()) for key in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes')}
        write(self.folder/'activity_counters.json',dict(counters=counters,returncode=code,passed=code==0 and not counters['dropped'] and not counters['invalid']))
        write(self.folder/'correlation_scopes.json',dict(steps=self.steps,operators=self.ops,
            scope='External correlation IDs associate eager CUDA launches with host call sites. '
                'Graph replay kernels need a separate measured attribution check. Host scopes are not GPU-exclusive durations.'))
        write(self.folder/'actual_ROI.json',dict(**self.roi,requested_delay_seconds=self.delay,
            requested_duration_seconds=self.duration,
            scope='Actual collector enable/disable anchors. A deferred ROI synchronizes its left boundary. '
                'Kernel union and gaps must be clipped to these GPU-clock anchors. '
                'Observer overhead is checked on the full clean/profile/clean serving trace.'))
        assert code==0,('Collector export failed. Inspect buffer diagnostics.',code,counters)

    def attach_hooks(self,cfg,role):
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        original_step=LLMEngine.step;original_prepare=GPUModelRunner.prepare_inputs;me=self
        if self.kernel_only:
            def kernel_step(engine,*a,**kw):
                me.refresh()
                return original_step(engine,*a,**kw)
            LLMEngine.step=kernel_step
            return
        def step(engine,*a,**kw):
            me.refresh()
            if not me.enabled:return original_step(engine,*a,**kw)
            me.counter+=1;identifier=me.counter
            row=dict(external_id=identifier,host_start=time.monotonic(),queries=[]);me.step=row;me.push(identifier)
            try:
                result=original_step(engine,*a,**kw)
                row['outputs']=[dict(request_id=value.request_id,count=len(value.outputs[0].token_ids),finished=value.finished)
                    for value in result]
                return result
            finally:
                me.pop(identifier);row['host_end']=time.monotonic();me.steps.append(row);me.step=None
        def prepare(runner,*a,**kw):
            batch=original_prepare(runner,*a,**kw)
            if me.step is not None:
                me.step['queries']=[dict(request_id=rid.split('-')[0],start=int(batch.num_computed_tokens_np[i]),
                    count=int(batch.num_scheduled_tokens[i]),prefilling=bool(batch.is_prefilling_np[i]),
                    actual_request_batch=int(batch.num_reqs),padded_queries=int(batch.num_tokens_after_padding)) for i,rid in enumerate(batch.req_ids)]
            return batch
        LLMEngine.step=step;GPUModelRunner.prepare_inputs=prepare
        if not cfg.get('operator_correlation'):return
        assert cfg['execution_family']=='MATCHED_NO_COMPILE'
        from vllm.model_executor.layers.linear import QKVParallelLinear,RowParallelLinear,MergedColumnParallelLinear
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        def wrap(cls,category):
            original=cls.forward
            def forward(module,*a,**kw):
                if not me.enabled or me.step is None or not any(q['prefilling'] for q in me.step['queries']):return original(module,*a,**kw)
                me.op_counter+=1;identifier=me.op_counter
                tensor=a[0] if a else None
                shape=list(tensor.shape) if hasattr(tensor,'shape') else None
                weight=list(module.weight.shape) if hasattr(module,'weight') else None
                row=dict(external_id=identifier,step_id=me.step['external_id'],category=category,
                    module_prefix=getattr(module,'prefix',None),input_shape=shape,weight_shape=weight,host_start=time.monotonic())
                me.push(identifier)
                try:return original(module,*a,**kw)
                finally:
                    me.pop(identifier);row['host_end']=time.monotonic();me.ops.append(row)
            cls.forward=forward
        for cls,category in ((QKVParallelLinear,'GEMM_QKV'),(MergedColumnParallelLinear,'GEMM_GATE_UP'),
                (RowParallelLinear,'GEMM_O_OR_DOWN'),(RMSNorm,'RMSNORM'),(Attention,'ATTENTION'),(RotaryEmbedding,'ROPE')):
            wrap(cls,category)
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        apply_head=LogitsProcessor._apply_head
        def lm_head(processor,head,hidden,bias):
            if not me.enabled or me.step is None:return apply_head(processor,head,hidden,bias)
            me.op_counter+=1;identifier=me.op_counter
            row=dict(external_id=identifier,step_id=me.step['external_id'],category='GEMM_LM_HEAD',
                module_prefix=getattr(head,'prefix','lm_head'),input_shape=list(hidden.shape),
                weight_shape=list(head.weight.shape),host_start=time.monotonic())
            me.push(identifier)
            try:return apply_head(processor,head,hidden,bias)
            finally:
                me.pop(identifier);row['host_end']=time.monotonic();me.ops.append(row)
        LogitsProcessor._apply_head=lm_head
