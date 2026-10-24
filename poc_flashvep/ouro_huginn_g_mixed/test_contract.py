"""Small CPU fixtures verify scheduler semantics, not pretrained quality."""
from dataclasses import replace
import torch
from vllm_rlt import CacheConfig, ExecutionConfig, SchedulerConfig, SamplingParams
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import HuginnForCausalLM
from tests.test_huginn import tiny_huginn_config
from .policy import attach

def execute(arm,group):
    torch.manual_seed(123)
    model=HuginnForCausalLM(tiny_huginn_config())
    e=StageEngine(model,cache_config=CacheConfig(num_blocks=256,block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=3,max_num_batched_tokens=8,
            prefill_chunk_size=3,prefill_loop_group_size=group,
            mode='mixed' if arm in ['mixed','hybrid'] else 'refill'),
        execution_config=ExecutionConfig(atomic_prefill=arm=='atomic'))
    attach(e,arm,group)
    params=SamplingParams(max_tokens=4,min_loops=3,max_loops=3,ignore_eos=True,seed=17)
    result={}
    for rid,ids in [('a',[4,7,9,2,5]),('b',[8,3,7])]:e.add_request(rid,ids,params)
    for _ in range(1000):
        if not e.has_unfinished_requests():break
        for out in e.step():result[out.request_id]=(out.token_ids,out.exit_depths)
    else:raise AssertionError('No scheduler progress')
    assert all(depths==[3]*len(tokens) for tokens,depths in result.values())
    assert e.cache_manager.num_used_blocks==0
    e.close()
    return result

def test_huginn_common_boundary_path():
    torch.set_num_threads(1)
    reference=execute('atomic',1)
    for arm,g in [('split',1),('group',2),('group',3),('mixed',1),('hybrid',2)]:
        assert execute(arm,g)==reference

def test_heterogeneous_prompt_remainders_use_one_legal_group():
    from vllm_rlt.core.scheduler import SchedulerOutput,ScheduledRequest
    from vllm_rlt.request import Request,RequestPhase
    from vllm_rlt.worker.executor import build_execution_batch
    # Real descriptors can meet after one-loop Mixed work advances only some rows.
    from vllm_rlt.core.scheduler import Scheduler
    from tests.helpers import tiny_ouro_model
    model=tiny_ouro_model()
    e=StageEngine(model,cache_config=CacheConfig(num_blocks=128,block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=2,max_num_batched_tokens=8,
            prefill_chunk_size=4,prefill_loop_group_size=2),
        execution_config=ExecutionConfig(atomic_prefill=False))
    for rid in ['late','fresh']:
        e.add_request(rid,[4,7,3,2],SamplingParams(max_tokens=2,min_loops=4,max_loops=4))
    s=e.scheduler
    # Admission and prelude are real. Set only the legal per-depth causal frontier.
    s._admit()
    for rid,depth in [('late',3),('fresh',0)]:
        request=s.requests[rid]
        request.num_preluded_tokens=4
        request.prefill_chunk_start=0
        request.prefill_depth_ranges=[[4,4] if d<depth else [0,4] if d==depth else [0,0] for d in range(4)]
        for q in s.queues.values():
            while rid in q:q.remove(rid)
        s.enqueue(request,RequestPhase.PREFILLING_RECURRENT)
    batch=s.take_phases((RequestPhase.PREFILLING_RECURRENT,))
    bound=build_execution_batch(batch,4)
    assert bound.loop_count==1
    assert len(bound.spans)==2
    assert [r.completes_prompt for r in batch.requests]==[True,False]
    e.close()
