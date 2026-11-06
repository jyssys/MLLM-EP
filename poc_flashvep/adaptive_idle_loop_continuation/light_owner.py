"""Sample CPU ownership without a CUDA memory query during serving."""
import os
import time
from .common import ROOT,write


def install(cfg):
    if not cfg.get('idle_light_owner'):return
    assert cfg.get('memory_sample_seconds',0)>=1e9,'Disable the inherited live HBM-query sampler'
    from vllm_rlt.engine.stage_engine import StageEngine
    initialized,step,closed=StageEngine.__init__,StageEngine._step,StageEngine.close
    def initialize(engine,*args,**kwargs):
        result=initialized(engine,*args,**kwargs)
        engine._idle_light_owner=dict(next=0.,samples=[])
        return result
    def stepped(engine,*args,**kwargs):
        state=engine._idle_light_owner;now=time.monotonic()
        if now>=state['next']:
            state['next']=now+.25
            pending={s.request.request_id for ticket in engine._inflight for s in ticket.batch.requests}
            blocked=set(engine._connector_blocked)
            queued={str(p):list(q) for p,q in engine.scheduler.queues.items()}
            eligible={rid for phase,ids in queued.items() if not any(v in phase for v in ('RECEIVING','TRANSFERRING','FINISHED')) for rid in ids}
            cache=engine.cache_manager
            state['samples'].append(dict(time=now,role=engine.stage_config.role,event='LIGHT_OWNER_PROGRESS',
                KV_total_blocks=cache.num_blocks,KV_used_blocks=cache.num_used_blocks,KV_free_blocks=cache.num_free_blocks,
                requests=[dict(request_id=r.request_id,phase=str(r.phase),prompt_tokens=len(r.prompt_token_ids),
                    output_tokens=len(r.generated_token_ids)) for r in engine.scheduler.requests.values()],
                all_submitted_unretired_ids=sorted(pending),in_flight_ids=sorted(pending),
                queued_not_inflight_not_connector_blocked_ids=sorted(eligible-pending-blocked),
                connector_blocked=sorted(blocked),scheduler_queue_ids=queued,
                scope='CPU ownership only. No live HBM query, event query, tensor readback, or GPU-utilization claim.'))
        return step(engine,*args,**kwargs)
    def close(engine):
        from ..looped_pd_autonomous_headroom.common import append
        folder=ROOT/'raw'/cfg['name']
        for row in engine._idle_light_owner['samples']:
            append(folder/f'memory_progress_light_{os.getpid()}.jsonl',row)
        write(folder/f'light_owner_contract_{os.getpid()}.json',dict(role=engine.stage_config.role,
            samples=len(engine._idle_light_owner['samples']),live_CUDA_memory_queries=0,
            scope='Only initial and final legacy snapshots query HBM. In-serving ownership records remain in memory until close.'))
        return closed(engine)
    StageEngine.__init__,StageEngine._step,StageEngine.close=initialize,stepped,close
