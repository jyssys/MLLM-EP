"""Keep a full36 decode graph out of partial-layer prefill,even at query length1."""
from .common import ROOT,write


def requires_eager(role,request_ids,parts,imports):
    for rid in request_ids:
        part=parts.get(rid.split('-')[0])
        if part is not None and part.upper and (role=='prefill' or rid in imports):return True
    return False


def install(adapter):
    import vllm.v1.worker.gpu.model_runner as module
    original=module.dispatch_cg_and_sync_dp
    def dispatch(*a,**kw):
        scheduled=getattr(adapter,'dispatch_schedule',None)
        if scheduled is not None and requires_eager(adapter.role,scheduled.num_scheduled_tokens,adapter.specs,adapter.imports):
            kw['need_eager']=True
        return original(*a,**kw)
    module.dispatch_cg_and_sync_dp=dispatch


def witness():
    from types import SimpleNamespace
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager,BatchExecutionDescriptor,get_uniform_token_count
    from .geometry import Partition
    manager=SimpleNamespace(_resolve_effective_loras=lambda _:0,_graphs_captured=True,
        _candidates={(1,0):[BatchExecutionDescriptor(CUDAGraphMode.FULL,1,1,1,0)]})
    desc=CudaGraphManager.dispatch(manager,1,1,get_uniform_token_count(1,1,1),0)
    assert desc.cg_mode==CUDAGraphMode.FULL
    parts={'q0':Partition(8193,4,0)}
    assert requires_eager('prefill',['q0-a'],parts,{})
    assert requires_eager('decode',['q0-a'],parts,{'q0-a':'source'})
    assert not requires_eager('decode',['q0-a'],parts,{})
    result=dict(stock_dispatch='FULL36_GRAPH_FOR_SHAPE1',partial_prefill_requires_eager=True,
        passed=True,evidence='CPU_ACTUAL_DISPATCH_REPRODUCTION',
        scope='The actual stock shape dispatcher cannot see partial-layer ownership. '
            'The campaign guard keeps real partial-prefill calls eager and leaves full36 decode eligible. GPU edge shapes require separate tests.')
    write(ROOT/'analysis/partial_prefill_query1_graph_guard_CPU.json',result);print(result)


if __name__=='__main__':witness()
