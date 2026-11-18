"""Optionally time the actual four projections in every executed D layer."""
import atexit
import collections


def install(cfg,role):
    if role!='decode' or not cfg.get('priority_projection_events'):
        return
    import torch
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from .common import ROOT,write
    active={};pending={};events=[];pool=collections.deque()
    prepare=GPUModelRunner.prepare_inputs
    execute=GPUModelRunner.execute_model
    construct=Qwen3ForCausalLM.__init__
    def prepared(self,*args,**kwargs):
        batch=prepare(self,*args,**kwargs)
        ids=[rid.split('-')[0] for rid in batch.req_ids]
        measured=(bool(batch.is_prefilling_np[:batch.num_reqs].any()) and bool(ids)
            and all(rid.startswith('q') and rid[1:].isdigit() and int(rid[1:])>=9 for rid in ids))
        active.update(measured=measured,requests=ids,query_tokens=batch.num_tokens)
        return batch
    def executed(self,*args,**kwargs):
        try:return execute(self,*args,**kwargs)
        finally:active.clear()
    def before(module,args,title):
        if not active.get('measured'):return
        assert not torch.cuda.is_current_stream_capturing()
        assert id(module) not in pending and pool
        x=args[0];weight=module.weight
        assert x.ndim==weight.ndim==2 and x.dtype==weight.dtype==torch.bfloat16
        a,b=pool.popleft()
        meta=dict(module=title,requests=list(active['requests']),query_tokens=active['query_tokens'],
            M=int(x.shape[0]),N=int(weight.shape[0]),K=int(weight.shape[1]),dtype=str(x.dtype))
        a.record();pending[id(module)]=(a,b,meta)
    def after(module,args,output):
        value=pending.pop(id(module),None)
        if value is not None:
            a,b,meta=value;b.record();events.append((a,b,meta))
    def initialized(self,*args,**kwargs):
        construct(self,*args,**kwargs)
        # Initialize event storage before the measurement barrier. Do not
        # allocate a new CUDA event inside each measured projection.
        for _ in range(cfg['count']*36*4):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();b.record();pool.append((a,b))
        torch.cuda.synchronize()
        endings=('self_attn.qkv_proj','self_attn.o_proj','mlp.gate_up_proj','mlp.down_proj')
        found=[]
        for name,module in self.named_modules():
            if '.layers.' not in name or not name.endswith(endings):continue
            found.append(name)
            module.register_forward_pre_hook(lambda module,args,title=name:before(module,args,title))
            module.register_forward_hook(after)
        assert len(found)==36*4,found
    def export():
        assert not pending
        torch.cuda.synchronize()
        write(ROOT/'raw'/cfg['name']/f'actual_projection_events_{role}.json',dict(
            rows=[dict(meta,elapsed_ms=a.elapsed_time(b)) for a,b,meta in events],
            evidence='DIAGNOSTIC',measurement_class='ACTUAL_EXECUTED_PROJECTION_CUDA_EVENT_ENCLOSURES',
            scope='Every recorded projection ran in the real transferred-state continuation. No layer0 multiplicity estimate. Events enclose each actual module invocation and can include launch gaps. This is not exclusive GEMM kernel time. Compare whole-forward timing against the unprobed control before interpreting the sums.'))
    Qwen3ForCausalLM.__init__=initialized
    GPUModelRunner.prepare_inputs=prepared
    GPUModelRunner.execute_model=executed
    atexit.register(export)


def main():
    from .common import bind_legacy
    bind_legacy()
    from ..dense_ar_slo_layer_pd import native
    previous=native.stage
    def stage(cfg,role,address):
        install(cfg,role)
        return previous(cfg,role,address)
    native.stage=stage
    native.main()


if __name__=='__main__':
    main()
