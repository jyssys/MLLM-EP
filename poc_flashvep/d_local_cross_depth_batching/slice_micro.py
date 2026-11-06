"""Measure full-semantic prompt slices. Do not interpret serial costs as slack."""
import time
from dataclasses import replace
from .common import *


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
    from ..looped_pd_autonomous_headroom.prefill_graph import install as prefill_graph
    prefill_graph(dict(cfg,colocated_prefill_graph_rows=256,P_prefill_graph_shapes=8))
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    from vllm_rlt.models import AutoModelForCausalLM
    from vllm_rlt import CacheConfig,ExecutionConfig,SchedulerConfig,ExitConfig,SamplingParams
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.request import RequestPhase as P
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATHS[cfg['model']],device='cuda:0',dtype=torch.bfloat16)
    engine=StageEngine(model,attention_backend='flash_attn_4',
        cache_config=CacheConfig(num_blocks=cfg['D_kv_blocks'],block_size=16,layout='last_exited',enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(mode='mixed',max_num_seqs=8,max_num_batched_tokens=256,
            prefill_chunk_size=128,wavefront_chunked_prefill=False),
        execution_config=ExecutionConfig(async_scheduling=False,prefill_uva=False,cuda_graphs=True,
            cuda_graph_max_batch_size=256,cuda_graph_max_graphs=8),exit_config=ExitConfig('ouro'))
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    planned=engine.scheduler._planned_loop_count
    quantum=[cfg['depth']]
    def loops(items):
        count=planned(items)
        if quantum[0]==1 and any(i.phase==P.PREFILLING_PRELUDE for i in items):return 0
        if any(i.phase in (P.PREFILLING_PRELUDE,P.PREFILLING_RECURRENT) for i in items):
            return min(count,quantum[0])
        return count
    engine.scheduler._planned_loop_count=loops
    original=ModelRunner._execute
    events=[]
    def execute(runner,scheduled,prepared):
        if runner is not engine.model_runner:return original(runner,scheduled,prepared)
        batch=prepared.batch
        spans=[dict(phase=str(s.request.phase),depth=s.depth,start=s.token_start,count=s.token_count) for s in batch.spans]
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();start=time.perf_counter();result=original(runner,scheduled,prepared);host=time.perf_counter()-start;b.record()
        events.append(dict(begin=a,end=b,operation=batch.operation.value,loops=batch.loop_count,spans=spans,host_submit_ms=1000*host))
        return result
    ModelRunner._execute=execute
    rows=[]
    try:
        for chunk in (16,32,64,128):
            for q in (cfg['depth'],1):
                quantum[0]=q
                engine.scheduler.config=replace(engine.scheduler.config,prefill_chunk_size=chunk)
                engine.scheduler.policy.config=engine.scheduler.config
                for repetition in range(2):
                    events.clear();start=time.monotonic();finished=None
                    engine.add_request('serial_prompt',data[0]['input_ids'][cfg['model']],SamplingParams(
                        max_tokens=2,min_loops=cfg['depth'],max_loops=cfg['depth'],exit_threshold=1.,
                        temperature=0.,ignore_eos=True,seed=cfg['seed']))
                    while engine.has_unfinished_requests():
                        assert time.monotonic()-start<180,'Bounded slice calibration'
                        for output in engine.step():
                            if output.finished:finished=dict(tokens=list(output.token_ids),depths=list(output.exit_depths))
                    torch.cuda.synchronize()
                    operations=[]
                    for event in events:
                        operations.append(dict(**{k:v for k,v in event.items() if k not in ('begin','end')},
                            enclosure_ms=event['begin'].elapsed_time(event['end'])))
                    prompt_rows=sum(s['count']*v['loops'] for v in operations for s in v['spans']
                        if s['phase'] in (str(P.PREFILLING_PRELUDE),str(P.PREFILLING_RECURRENT)))
                    assert prompt_rows==cfg['length']*cfg['depth'],'Full recurrent prompt work changed'
                    rows.append(dict(chunk=chunk,quantum=q,repetition=repetition,warmup=repetition==0,
                        host_cohort_ms=1000*(time.monotonic()-start),prompt_recurrent_rows=prompt_rows,
                        completed=finished,operations=operations,
                        evidence='MEASURED_SERIAL_MODEL_STREAM_ENCLOSURES_NOT_ONLINE_SLACK'))
                    write(folder/'slice_costs.json',rows)
    finally:
        ModelRunner._execute=original;engine.close()
    write(folder/'status.json',dict(status='COMPLETE'))


if __name__=='__main__':main()
