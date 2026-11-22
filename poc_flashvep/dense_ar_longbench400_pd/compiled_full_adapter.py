"""Preserve compiled full-model calls and isolate partial-prefill dispatch."""
import hashlib
import inspect
import time
from .adapter import Adapter


class CompiledFullAdapter(Adapter):
    def completed_full_graph(self,runner,execution='UNCHANGED_FULL36_STANDARD_RUNNER'):
        self.full_observed=True
        return super().completed_full_graph(runner,execution=execution)

    def install(self):
        from vllm.model_executor.models.qwen3 import Qwen3Model,Qwen3ForCausalLM,Qwen3DecoderLayer
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        assert self.cfg.get('preserve_compiled_full_model')
        original_inner=Qwen3Model.forward
        original_layer=Qwen3DecoderLayer.forward
        original_outer=Qwen3ForCausalLM.forward
        # Reuse ownership, NIXL, chunk-boundary and graph-safety hooks. Restore
        # the original compile-decorated model and Transformer before LLM init.
        super().install()
        Qwen3Model.forward=original_inner
        Qwen3DecoderLayer.forward=original_layer
        me=self;self.partial_executed=False;self.full_observed=False

        def outer(module,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
            batch=me.active.get('batch')
            partial=False
            if batch is not None and all(me.external(rid) in me.specs for rid in batch.req_ids):
                if me.role=='prefill':
                    partial=any(me.partition(rid).upper and bool(batch.is_prefilling_np[i])
                        for i,rid in enumerate(batch.req_ids))
                else:
                    partial=any(me.partition(rid).upper and bool(batch.is_prefilling_np[i]) and
                        int(batch.num_computed_tokens_np[i])<me.partition(rid).prefix
                        for i,rid in enumerate(batch.req_ids))
            if partial:
                assert intermediate_tensors is None and inputs_embeds is None
                me.partial_executed=True;me.active['python_observed']=True
                return me.partial_forward(module.model,input_ids,positions)
            return original_outer(module,input_ids,positions,intermediate_tensors,inputs_embeds)
        Qwen3ForCausalLM.forward=outer

        execute=GPUModelRunner.execute_model
        def executed(runner,*a,**kw):
            me.partial_executed=False;me.full_observed=False
            result=execute(runner,*a,**kw)
            schedule=a[0] if a else kw.get('scheduler_output')
            dummy=kw.get('dummy_run',a[2] if len(a)>2 else False)
            if dummy or schedule is None or schedule.total_num_scheduled_tokens==0:return result
            if not me.partial_executed and not me.full_observed:
                frame=runner.execute_model_state
                if frame is not None and frame.hidden_states is not None and all(
                        me.external(rid) in me.specs for rid in frame.input_batch.req_ids):
                    assert len(runner.model.model.layers)==36
                    me.completed_full_graph(runner)
            return result
        GPUModelRunner.execute_model=executed
        assert Qwen3Model.forward is original_inner and Qwen3DecoderLayer.forward is original_layer
        self.rows.append(dict(event='PRESERVED_COMPILED_FULL_MODEL',
            original_inner_forward_identity=True,original_decoder_forward_identity=True,
            inner_forward_source_SHA256=hashlib.sha256(inspect.getsource(original_inner).encode()).hexdigest(),
            decoder_forward_source_SHA256=hashlib.sha256(inspect.getsource(original_layer).encode()).hexdigest(),
            time=time.monotonic(),
            scope='Only the outer partial-prefill dispatch changes. Full-model calls retain the original compile-decorated model and layers. '
                'Partial prefills use the qualified row-aware causal implementation. '
                'This source invariant requires separate strict state, mixed-row and natural-output validation.'))


def cpu_identity_test():
    from vllm.model_executor.models.qwen3 import Qwen3Model,Qwen3DecoderLayer
    from .common import ROOT
    expected=(Qwen3Model.forward,Qwen3DecoderLayer.forward)
    obj=CompiledFullAdapter.__new__(CompiledFullAdapter)
    obj.cfg=dict(preserve_compiled_full_model=True,axis_kind='Layer',axis_cut=4)
    obj.role='prefill';obj.rows=[];obj.specs={};obj.imports={};obj.active={};obj.folder=ROOT/'state'
    obj.install()
    assert expected==(Qwen3Model.forward,Qwen3DecoderLayer.forward)
    print('ORIGINAL_COMPILE_MODEL_AND_LAYER_IDENTITIES_PASS',flush=True)


if __name__=='__main__':cpu_identity_test()
