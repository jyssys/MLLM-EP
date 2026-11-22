"""Match execution paths in an independent, unchanged full-model teacher."""
import atexit
import copy
import time
from .common import ROOT,write


def install(cfg,role,info):
    if not cfg.get('teacher_eager_prefix_compiled_suffix'):return
    assert role=='single' and cfg['serial'] and not cfg.get('axis_kind')
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.forward_context import get_forward_context,override_forward_context
    from vllm.config import CUDAGraphMode
    import vllm.v1.worker.gpu.model_runner as runner_module
    state={'eager':False};rows=[]
    original_schedule=Scheduler.schedule
    def schedule(scheduler,*a,**kw):
        requests=list(scheduler.running)+list(scheduler.waiting)+list(scheduler.skipped_waiting)
        requests=[r for r in requests if r.request_id.split('-')[0] in info]
        assert len(requests)<=1,'The independent teacher is serial'
        state['eager']=False
        for request in requests:
            length=request.num_prompt_tokens
            fraction=cfg.get('teacher_suffix_fraction',0)
            suffix=max(16,int(round(length*fraction/16))*16) if fraction else 0
            state['eager']=request.num_computed_tokens<length-suffix
        return original_schedule(scheduler,*a,**kw)
    Scheduler.schedule=schedule
    original_dispatch=runner_module.dispatch_cg_and_sync_dp
    def dispatch(*a,**kw):
        if state['eager']:kw['need_eager']=True
        return original_dispatch(*a,**kw)
    runner_module.dispatch_cg_and_sync_dp=dispatch
    original_outer=Qwen3ForCausalLM.forward
    def outer(module,*a,**kw):
        if not state['eager']:return original_outer(module,*a,**kw)
        context=copy.copy(get_forward_context());context.skip_compiled=True
        context.cudagraph_runtime_mode=CUDAGraphMode.NONE;context.batch_descriptor=None
        rows.append(dict(time=time.monotonic(),execution='ORIGINAL_FULL36_EAGER_PREFIX'))
        with override_forward_context(context):return original_outer(module,*a,**kw)
    Qwen3ForCausalLM.forward=outer
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'independent_teacher_execution_path.json',dict(
        rows=rows,original_inner_model_unchanged=True,original_layers_unchanged=True,
        no_partition_adapter=True,no_NIXL=True,no_tensor_correction=True,evidence='CORRECTNESS_ONLY',
        scope='Original full36 model. The prefix uses the eager numerical path of real partial-layer calls. '
            'The full suffix and decode retain the original compiled path. Only execution geometry changes. '
            'The earlier all-compiled teacher mismatch remains preserved. This teacher is not a performance baseline.')))
