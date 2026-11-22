"""Keep every prompt call eager/CUDA while preserving native compiled decode."""
import copy
import time
from .compiled_full_adapter import CompiledFullAdapter


def needs_eager_prompt(request_ids,parts,completed):
    return any(rid.split('-')[0] in parts and rid.split('-')[0] not in completed for rid in request_ids)


def pending_prompt_schedule(adapter):
    schedule=getattr(adapter,'dispatch_schedule',None)
    return schedule is not None and needs_eager_prompt(schedule.num_scheduled_tokens,
        adapter.specs,adapter.prompt_completed)


class EagerPrefillCompiledAdapter(CompiledFullAdapter):
    def install(self):
        import vllm.v1.worker.gpu.model_runner as runner_module
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        from vllm.forward_context import get_forward_context,override_forward_context
        from vllm.config.kernel import IrOpPriorityConfig
        from vllm.config import CUDAGraphMode
        assert self.role=='decode' and self.cfg.get('D_eager_prefill_IR_norm')
        self.prompt_completed=set()
        super().install()
        original_dispatch=runner_module.dispatch_cg_and_sync_dp
        me=self
        def dispatch(*a,**kw):
            if pending_prompt_schedule(me):
                kw['need_eager']=True
            return original_dispatch(*a,**kw)
        runner_module.dispatch_cg_and_sync_dp=dispatch
        original_outer=Qwen3ForCausalLM.forward
        def outer(module,*a,**kw):
            batch=me.active.get('batch')
            if batch is None or not all(me.external(rid) in me.specs for rid in batch.req_ids):
                return original_outer(module,*a,**kw)
            prompt=any(int(batch.num_computed_tokens_np[i])<me.partition(rid).length
                for i,rid in enumerate(batch.req_ids))
            if not prompt:return original_outer(module,*a,**kw)
            context=copy.copy(get_forward_context())
            context.skip_compiled=True;context.cudagraph_runtime_mode=CUDAGraphMode.NONE
            context.batch_descriptor=None
            with override_forward_context(context):
                with IrOpPriorityConfig.with_default(['vllm_c','native']).set_priority():
                    me.rows.append(dict(event='EAGER_PREFILL_IR_PRIORITY',time=time.monotonic(),
                        providers=['vllm_c','native'],requests=[me.external(rid) for rid in batch.req_ids],
                        scope='All real prompt queries stay eager with original CUDA norm. Decode-only calls keep native compiled execution.'))
                    result=original_outer(module,*a,**kw)
            for i,rid in enumerate(batch.req_ids):
                start=int(batch.num_computed_tokens_np[i]);count=int(batch.num_scheduled_tokens[i])
                if start<me.partition(rid).length<=start+count:
                    me.prompt_completed.add(me.external(rid))
            return result
        Qwen3ForCausalLM.forward=outer


def cpu_test():
    from types import SimpleNamespace
    from .geometry import Partition
    from .graph_guard import requires_eager
    part={'q0':Partition(3987,4,496)}
    assert not requires_eager('decode',['q0-a'],part,{})
    assert needs_eager_prompt(['q0-a'],part,set())
    assert not needs_eager_prompt(['q0-a'],part,{'q0'})
    assert not needs_eager_prompt(['dummy-a'],part,set())
    assert not pending_prompt_schedule(SimpleNamespace())
    assert pending_prompt_schedule(SimpleNamespace(dispatch_schedule=SimpleNamespace(num_scheduled_tokens={'q0-a':496}),
        specs=part,prompt_completed=set()))
    print('WHOLE_PROMPT_QUERY_GRAPH_GUARD_AND_DECODE_RELEASE_PASS',flush=True)


if __name__=='__main__':cpu_test()
