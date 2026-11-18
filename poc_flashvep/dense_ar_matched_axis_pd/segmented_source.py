"""Assemble exact prefix frontier rows while standard chunking proceeds."""
import hashlib
import inspect
import json
from .common import ROOT,write

OFFSETS={}


def transform_extension(code):
    edits=[
        ("assert all(int(v)==0 for v in batch.num_computed_tokens_np[:batch.num_reqs])",
         "assert role=='prefill' or all(int(v)==0 for v in batch.num_computed_tokens_np[:batch.num_reqs])",1),
        ("free=transport['value'].slots.free_count()",
         "free=transport['value'].slots.free_count()+len(getattr(transport['value'],'segments',{}))",1),
        ("event('HIDDEN_SOURCE_CREDIT_BACKPRESSURE',free_slots=free,normal_limit=old_limit,",
         "event('HIDDEN_SOURCE_CREDIT_BACKPRESSURE',free_slots=transport['value'].slots.free_count(),continuation_credits=len(getattr(transport['value'],'segments',{})),scheduling_credits=free,normal_limit=old_limit,",1),
    ]
    for old,new,count in edits:
        assert code.count(old)==count,old
        code=code.replace(old,new)
    return code


def install(cfg,role):
    if not cfg.get('source_chunk_packing') or role!='prefill':
        return
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        batch=prepare(self,*a,**kw)
        for i,rid in enumerate(batch.req_ids):
            OFFSETS[rid]=(int(batch.num_computed_tokens_np[i]),int(batch.num_scheduled_tokens[i]))
        return batch
    GPUModelRunner.prepare_inputs=prepared
    from ..dense_ar_layerwise_pd.hidden_transport import GPUHiddenTransport
    class SegmentedTransport(GPUHiddenTransport):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.segments={}
        def put(self,rid,hidden,residual,cut):
            start,count=OFFSETS[rid]
            assert hidden.shape==residual.shape==(count,4096)
            self.reap()
            if rid not in self.segments:
                assert start==0,'First frontier fragment must start at token0'
                self.segments[rid]=dict(slot=self.slots.take(rid),cursor=0,cut=cut,chunks=0)
            state=self.segments[rid]
            assert state['cursor']==start and state['cut']==cut,'Duplicate, missing or mixed-cut frontier rows'
            assert 0<=start<start+count<=self.cfg['length']
            slot=state['slot']
            self.pool[slot,0,start:start+count].copy_(hidden)
            self.pool[slot,1,start:start+count].copy_(residual)
            state['cursor']+=count;state['chunks']+=1
            self.event('HIDDEN_FRONTIER_FRAGMENT',request_id=rid,slot=slot,start=start,count=count,
                cursor=state['cursor'],cut=cut,complete=state['cursor']==self.cfg['length'],
                bytes=count*4096*2*2,
                scope='Fill the owned GPU frontier buffer at original token positions. No source descriptor is published until every row is present.')
            if state['cursor']!=self.cfg['length']:
                return
            take=self.slots.take
            def reuse_slot(request):
                assert request==rid and self.slots.owners[slot]==rid
                return slot
            self.slots.take=reuse_slot
            try:
                # The full-buffer self-copies preserve the original publication,
                # hashing and source-sample readiness protocol without new slots.
                super().put(rid,self.pool[slot,0],self.pool[slot,1],cut)
            finally:
                self.slots.take=take
            self.event('HIDDEN_FRONTIER_ASSEMBLED',request_id=rid,slot=slot,cut=cut,
                tokens=state['cursor'],chunks=state['chunks'],bytes=self.slot_bytes)
            del self.segments[rid]
        def close(self):
            assert not self.segments,'Do not retire an incomplete source frontier'
            return super().close()
    from ..dense_ar_slo_layer_pd import hidden_transport,dynamic_extension
    original_class=hidden_transport.transport_class
    hidden_transport.transport_class=lambda config: SegmentedTransport if config.get('source_chunk_packing') else original_class(config)
    original=inspect.getsource(dynamic_extension.install)
    transformed=transform_extension(original)
    scope=dict(dynamic_extension.__dict__)
    exec(compile(transformed,__file__+':chunked_prefix_install','exec'),scope)
    dynamic_extension.install=scope['install']
    write(ROOT/'raw'/cfg['name']/'chunked_prefix_execution_contract.json',dict(
        original_SHA256=hashlib.sha256(original.encode()).hexdigest(),
        executed_SHA256=hashlib.sha256(transformed.encode()).hexdigest(),source=transformed,
        scope='Allow nonzero P query offsets. Keep D suffix at query0. Continuation-owned slots remain runnable when new-slot credit is0. Preserve original NIXL descriptor, ACK and final publication protocol. No model arithmetic or KV manager change.'))
