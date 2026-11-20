"""Add step correlation only in profiled workers. Preserve the BI0 math path."""
import json
import sys
import time
from .common import ROOT,activate,write
activate()


def main():
    cfg=json.load(open(sys.argv[1]));role=sys.argv[sys.argv.index('--role')+1]
    profile=bool(cfg.get('correlated_profile')) and role in cfg.get('kernel_activity_roles',['prefill'])
    if profile:
        from . import correlated_activity
        from ..dense_ar_bi0_shortanswer_pd import activity as prior_activity
        if cfg.get('runtime_only_correlation'):
            from .runtime_only_activity import Activity
            prior_activity.Activity=Activity
        elif cfg.get('runtime_api_correlation'):
            from .runtime_activity import Activity
            prior_activity.Activity=Activity
        else:
            prior_activity.Activity=correlated_activity.Activity
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.core.sched.scheduler import Scheduler
        original_step=LLMEngine.step;original_prepare=GPUModelRunner.prepare_inputs
        original_schedule=Scheduler.schedule;original_ack=Scheduler._update_from_kv_xfer_finished
        state=dict(counter=0,current=None);rows=[]
        if cfg.get('runtime_only_correlation'):
            from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
            original_graph=CudaGraphManager.run_fullgraph
            def graph(manager,*a,**kw):
                if state['current'] is not None:state['current']['graph_replays']+=1
                return original_graph(manager,*a,**kw)
            CudaGraphManager.run_fullgraph=graph
        def section(name,function,*a,**kw):
            left=time.monotonic()
            try:return function(*a,**kw)
            finally:
                if state['current'] is not None:
                    state['current']['host_sections'].append(dict(name=name,start=left,end=time.monotonic()))
        def schedule(scheduler,*a,**kw):
            return section('scheduler_schedule',original_schedule,scheduler,*a,**kw)
        def ack(scheduler,*a,**kw):
            return section('scheduler_KV_completion_update',original_ack,scheduler,*a,**kw)
        def prepare(runner,*a,**kw):
            batch=section('prepare_inputs',original_prepare,runner,*a,**kw)
            if state['current'] is not None:
                for i,rid in enumerate(batch.req_ids):
                    state['current']['queries'].append(dict(request_id=rid.split('-')[0],
                        start=int(batch.num_computed_tokens_np[i]),count=int(batch.num_scheduled_tokens[i]),
                        actual_batch=int(batch.num_reqs),padded_queries=int(batch.num_tokens_after_padding)))
            return batch
        def step(engine,*a,**kw):
            collector=correlated_activity.CURRENT
            if collector is None:
                return original_step(engine,*a,**kw)
            scheduler=engine.engine_core.engine_core.scheduler
            selected=cfg.get('profile_sample_request_ids')
            if selected is not None:
                assert role=='prefill' and cfg.get('controlled_pipeline')
                active_ids={r.request_id.split('-')[0] for r in list(scheduler.running)+list(scheduler.waiting)+list(scheduler.skipped_waiting)}
                include=bool(active_ids&set(selected))
                if include and not collector.enabled:collector.resume()
                elif not include and collector.enabled:collector.pause()
            if not collector.enabled:
                return original_step(engine,*a,**kw)
            if not (scheduler.running or scheduler.waiting or scheduler.skipped_waiting):
                return original_step(engine,*a,**kw)
            state['counter']+=1;identifier=state['counter']
            row=dict(external_id=identifier,host_start=time.monotonic(),queries=[],host_sections=[],graph_replays=0)
            state['current']=row;collector.push(identifier)
            try:return original_step(engine,*a,**kw)
            finally:
                collector.pop(identifier);row['host_end']=time.monotonic();rows.append(row);state['current']=None
        GPUModelRunner.prepare_inputs=prepare;LLMEngine.step=step
        Scheduler.schedule=schedule;Scheduler._update_from_kv_xfer_finished=ack
    from ..dense_ar_bi0_shortanswer_pd import standard_worker as worker
    if profile:
        original_write=worker.write
        def observed_write(path,value):
            if path.name==f'engine_steps_{role}.json':
                write(ROOT/'raw'/cfg['name']/f'engine_step_external_ids_{role}.json',dict(rows=rows,
                    scope='External IDs enclose an engine-step call. Query spans identify the model work. Kernel records are joined by CUPTI correlation, not by host interval containment.'))
            return original_write(path,value)
        worker.write=observed_write
    worker.main()


if __name__=='__main__':main()
