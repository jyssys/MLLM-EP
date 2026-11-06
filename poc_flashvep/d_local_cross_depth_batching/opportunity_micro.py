"""Closed-loop ready-work control. No PD throughput or online-placement claim."""
import time
from dataclasses import replace
from .common import *


def run(model, cfg, folder):
    import torch
    from vllm_rlt import CacheConfig, ExecutionConfig, SchedulerConfig, ExitConfig, SamplingParams
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    from vllm_rlt.worker.model_runner import ModelRunner
    chunk=cfg.get('opportunity_chunk',32)
    token_budget=cfg.get('opportunity_token_budget',136)
    graph_rows=cfg.get('opportunity_graph_rows',136)
    engine = StageEngine(model, attention_backend='flash_attn_4',
        cache_config=CacheConfig(num_blocks=cfg['D_kv_blocks'],block_size=16,layout='last_exited',enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(mode='mixed',max_num_seqs=8,max_num_batched_tokens=token_budget,
            prefill_chunk_size=chunk,wavefront_chunked_prefill=False),
        execution_config=ExecutionConfig(async_scheduling=False,prefill_uva=False,cuda_graphs=True,
            cuda_graph_max_batch_size=graph_rows,cuda_graph_max_graphs=8),exit_config=ExitConfig('ouro'))
    native_policy = engine.scheduler.policy
    native_take = engine.scheduler.take_phases
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    from .qualification import measured
    reference=measured(ROOT/'raw'/f'local_{cfg["model"]}_native_n8_s2026110701_off_state')
    results=[]
    old=ModelRunner.submit
    counters={}
    def submit(self,scheduled_batch=None,*,batch):
        if self is engine.model_runner:
            phases={str(s.request.phase) for s in batch.spans}
            key=(batch.operation.value,batch.num_rows,batch.loop_count,len(phases)>1,
                len({s.depth for s in batch.spans})>1)
            counters[key]=counters.get(key,0)+1
        return old(self,scheduled_batch,batch=batch)
    ModelRunner.submit=submit
    policies=('separate','same','same_align','cross') if cfg.get('opportunity_same_align') else ('separate','same','cross')
    try:
        for repetition in range(4):
            order=policies if repetition%2==0 else tuple(reversed(policies))
            for method in order:
                # Preserve the native phase/prelude/Coda priorities in every
                # arm. Change only which ready recurrent rows may join.
                engine.scheduler.policy=native_policy
                native_policy.prefill_batches_since_recurrent=0
                alignment_steps=[0]
                def controlled_take(phases, *, eligible=None):
                    if P.RECURRENT not in phases or P.PREFILLING_RECURRENT not in phases or method=='cross':
                        return native_take(phases,eligible=eligible)
                    if method=='separate':
                        return native_take((P.RECURRENT,),eligible=eligible)
                    depths={engine.scheduler.requests[rid].loops_done
                        for rid in engine.scheduler.queues[P.RECURRENT]}
                    if method=='same_align' and alignment_steps[0]==0 and depths:
                        prompt=engine.scheduler.queues[P.PREFILLING_RECURRENT]
                        pdepths=[engine.scheduler.requests[rid].prefill_depth(engine.scheduler.requests[rid].num_prefilled_tokens) for rid in prompt]
                        if pdepths and not depths.intersection(pdepths) and min(depths)-min(pdepths)==1:
                            # One legal prefill core step can align the queues.
                            # This deliberately delays ready decode once. It is
                            # a measured statistical control, not strict priority.
                            selected=native_take((P.PREFILLING_RECURRENT,),eligible=eligible)
                            if selected is not None:
                                assert selected.loop_count==1
                                alignment_steps[0]+=1
                                return selected
                    def compatible(request):
                        accepted=eligible(request) if eligible is not None else True
                        if request.phase==P.PREFILLING_RECURRENT:
                            accepted=accepted and request.prefill_depth(request.num_prefilled_tokens) in depths
                        return accepted
                    return native_take(phases,eligible=compatible)
                engine.scheduler.take_phases=controlled_take
                counters.clear()
                def graph_counts():
                    objects=[engine.model_runner.graphs,getattr(engine,'_autonomous_prefill_graphs',None)]
                    return {f'{index}_{type(graph).__name__}':dict(captures=graph.captures,replays=graph.replays,
                        fallbacks=graph.fallbacks,entries=len(graph.entries))
                        for index,graph in enumerate(objects) if graph is not None}
                graph_before=graph_counts()
                token_events={};completed={};local_added=False;background_ready=set();at_local={}
                clock=time.monotonic();local_start=None
                for i in range(2):
                    engine.add_request(f'decode_{i}',data[i]['input_ids'][cfg['model']],SamplingParams(
                        max_tokens=cfg['cap'],min_loops=cfg['depth'],max_loops=cfg['depth'],
                        exit_threshold=1.,temperature=0.,ignore_eos=True,seed=cfg['seed']+i))
                while engine.has_unfinished_requests():
                    assert time.monotonic()-clock<240,'Bounded ready-work micro'
                    outputs=engine.step()
                    now=time.monotonic()
                    for output in outputs:
                        if output.request_id.startswith('decode_') and output.token_ids:
                            background_ready.add(output.request_id)
                        token_events.setdefault(output.request_id,[]).append((now,len(output.token_ids)))
                        if output.finished:
                            completed[output.request_id]=dict(tokens=list(output.token_ids),depths=list(output.exit_depths),end=now)
                    join_ready=len(background_ready)==2 if cfg.get('opportunity_strict_join') else any(len(out.token_ids) for out in outputs)
                    if not local_added and join_ready:
                        local_start=time.monotonic()
                        at_local=dict(counters)
                        engine.add_request('local',data[2]['input_ids'][cfg['model']],SamplingParams(
                            max_tokens=2,min_loops=cfg['depth'],max_loops=cfg['depth'],exit_threshold=1.,
                            temperature=0.,ignore_eos=True,seed=cfg['seed']+2))
                        local_added=True
                row=dict(method=method,repetition=repetition,warmup=repetition==0,
                    chunk=chunk,token_budget=token_budget,graph_rows=graph_rows,
                    one_step_alignment_submissions=alignment_steps[0],
                    cohort_host_ms=1000*(time.monotonic()-clock),local_first_ms=1000*(token_events['local'][0][0]-local_start),
                    token_events=token_events,completed=completed,
                    graph_before=graph_before,graph_after=graph_counts(),
                    submissions=[dict(operation=k[0],rows=k[1],loops=k[2],cross_phase=k[3],cross_depth=k[4],count=v)
                        for k,v in counters.items()],
                    submissions_after_local_enqueue=[dict(operation=k[0],rows=k[1],loops=k[2],cross_phase=k[3],
                        cross_depth=k[4],count=v-at_local.get(k,0)) for k,v in counters.items() if v>at_local.get(k,0)],
                    evidence='MEASURED_CLOSED_LOOP_READY_WORK_MICRO_NOT_PD_SERVING',
                    scope='Two existing decode streams plus one new whole local prompt. Same inputs, placement and output caps. No request-arrival QPS or sustainable capacity claim.')
                results.append(row)
                write(folder/'opportunity_micro.json',results)
                for rid,value in completed.items():
                    index=2 if rid=='local' else int(rid.rsplit('_',1)[1])
                    expected=reference[index]
                    assert value['tokens']==expected['output_ids'][:len(value['tokens'])], 'Micro output differs from native reference'
                    assert value['depths']==expected['exit_depths'][:len(value['depths'])], 'Micro recurrent depth differs'
                expected_rows=(3*cfg['length']+2*cfg['cap']-1)*cfg['depth']
                actual_rows=sum(k[1]*k[2]*v for k,v in counters.items())
                assert actual_rows==expected_rows,'Micro full recurrent work differs'
    finally:
        ModelRunner.submit=old
        engine.close()


def main():
    configure()
    cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text());folder=ROOT/'raw'/cfg['name']
    import torch
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.cuda.set_device(0)
    from ..looped_pd_b_numerical_diagnosis import common as numerical
    numerical.ROOT=ROOT
    os.environ.update(B_NUMERICAL_VARIANT='down_norm' if cfg['model']=='ouro' else 'down_norm_adapter',
        B_NUMERICAL_RUN=cfg['name'],ABC_ALLOW_STAGED_BOUNDARY='1')
    from ..looped_pd_b_numerical_diagnosis.patches import install
    install()
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    from vllm_rlt.models import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATHS[cfg['model']],device='cuda:0',dtype=torch.bfloat16)
    from ..looped_pd_autonomous_headroom.prefill_graph import install as prefill_graph
    prefill_graph(dict(cfg,colocated_prefill_graph_rows=cfg.get('opportunity_graph_rows',136),P_prefill_graph_shapes=8))
    run(model,cfg,folder)
    write(folder/'status.json',dict(status='COMPLETE'))


if __name__=='__main__':
    main()
