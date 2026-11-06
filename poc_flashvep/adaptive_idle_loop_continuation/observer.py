"""Separate host readiness, sparse stream enclosures, and graph activity."""
import os
import time
from .common import *


def install(cfg):
    mode=cfg.get('idle_observer','off')
    if mode=='off':return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from .graph_activity import library,clock
    original_init,original_schedule,original_execute,original_close=StageEngine.__init__,StageEngine._schedule,ModelRunner._execute,StageEngine.close
    original_submit=ModelRunner.submit
    folder=ROOT/'raw'/cfg['name']
    roles=cfg.get('idle_observer_roles',('prefill','decode'))

    def initialize(engine,*args,**kwargs):
        role=getattr(kwargs.get('stage_config'),'role',None)
        handle=None;start=None;activation=None
        if role in roles and mode=='graph':
            # Graph trace mode must exist before target graph instantiation.
            handle=library()
            if cfg.get('idle_activity_device_context'):
                model=args[0]
                with torch.cuda.device(next(model.parameters()).device):
                    activation=dict(model_device=str(next(model.parameters()).device),current_device=torch.cuda.current_device())
                    assert handle.activity_begin(3)==0;start=clock(handle)
            else:
                activation=dict(model_device=str(next(args[0].parameters()).device),current_device=torch.cuda.current_device())
                assert handle.activity_begin(3)==0;start=clock(handle)
        result=original_init(engine,*args,**kwargs)
        if engine.stage_config.role not in roles:return result
        state=dict(role=engine.stage_config.role,handle=handle,start=start,activation=activation,origin=None,next_poll=0.,
            queues=[],operations=[],pool=[],anchor=None,calls=0,queue_calls=0,last_queue=-1.,closed=False)
        runner=engine.model_runner
        if mode in ('sparse','submit_events'):
            stream=runner.core_stream or torch.cuda.current_stream(runner.device)
            state['pool']=[torch.cuda.Event(enable_timing=True) for _ in range(16384 if mode=='submit_events' else 2048)]
            for event in state['pool']:event.record(stream)
            stream.synchronize()
            anchor=state['pool'].pop();left=time.monotonic();anchor.record(stream);anchor.synchronize();right=time.monotonic()
            state['anchor']=anchor;state['anchor_host']=dict(lower=left,upper=right,midpoint=(left+right)/2)
        runner._idle_observation=engine._idle_observation=state
        return result

    def within(state,now):
        if state['origin'] is None:
            if now<state['next_poll']:return False
            state['next_poll']=now+.2
            path=folder/'measurement_begin.json'
            if path.exists():state['origin']=json.loads(path.read_text())['time']
            if state['origin'] is None:return False
        begin=state['origin']+cfg.get('idle_trace_fraction',.35)*cfg['arrival_offsets'][-1]
        return begin<=now<=begin+cfg.get('idle_trace_seconds',8.)

    def schedule(engine,*args,**kwargs):
        state=getattr(engine,'_idle_observation',None)
        if state is None:return original_schedule(engine,*args,**kwargs)
        begin=time.monotonic();state['queue_calls']+=1
        result=original_schedule(engine,*args,**kwargs)
        if (within(state,begin) and state['queue_calls']%cfg.get('idle_queue_stride',1)==0
                and begin-state['last_queue']>=.002):
            state['last_queue']=begin
            state['queues'].append(dict(time=begin,return_time=time.monotonic(),
                phase_queue_counts={str(p):len(q) for p,q in engine.scheduler.queues.items()},
                selected=[] if result is None else [dict(id=s.request.request_id,seed=s.request.sampling_params.seed,
                    phase=str(result.phase_for(s)),depth=s.depth,tokens=s.token_count) for s in result.requests],
                connector_blocked=list(engine._connector_blocked),KV_free_blocks=engine.cache_manager.num_free_blocks,
                scope='Sampled actual schedule selections and queue membership. Unselected membership does not prove device readiness.'))
        return result

    def execute(runner,scheduled,prepared):
        state=getattr(runner,'_idle_observation',None)
        if state is None or mode!='sparse' or not within(state,time.monotonic()):
            return original_execute(runner,scheduled,prepared)
        state['calls']+=1
        if state['calls']%cfg.get('idle_event_stride',16):return original_execute(runner,scheduled,prepared)
        assert len(state['pool'])>=2,'Bounded sparse event pool exhausted'
        left,right=state['pool'].pop(),state['pool'].pop();left.record()
        begin=time.monotonic();result=original_execute(runner,scheduled,prepared);right.record()
        state['operations'].append(dict(begin=left,end=right,host_begin=begin,host_end=time.monotonic(),
            operation=prepared.batch.operation.value,loops=prepared.batch.loop_count,rows=prepared.batch.num_rows))
        return result

    def submit(runner,scheduled_batch=None,*,batch):
        state=getattr(runner,'_idle_observation',None)
        begin=time.monotonic()
        if state is None or mode!='submit_events' or not within(state,begin):
            return original_submit(runner,scheduled_batch,batch=batch)
        stream=(runner.core_stream if batch.loop_count else runner.boundary_stream) or torch.cuda.current_stream(runner.device)
        assert len(state['pool'])>=2,'Bounded submission event pool exhausted'
        left,right=state['pool'].pop(),state['pool'].pop()
        left.record(stream)
        result=original_submit(runner,scheduled_batch,batch=batch)
        right.record(stream)
        state['calls']+=1
        state['operations'].append(dict(begin=left,end=right,host_begin=begin,host_end=time.monotonic(),
            operation=batch.operation.value,loops=batch.loop_count,rows=batch.num_rows,
            request_ids=[s.request.request_id for s in batch.spans],
            phases=[str(s.request.phase) for s in batch.spans],depths=[s.depth for s in batch.spans],
            stream=int(stream.cuda_stream)))
        return result

    def close(engine):
        state=getattr(engine,'_idle_observation',None)
        if state is not None and not state['closed']:
            state['closed']=True;runner=engine.model_runner;runner.synchronize()
            records=[];capture=None;invalid_capture=False
            if state['handle'] is not None:
                stop=clock(state['handle'])
                path=folder/f'activity_{state["role"]}_{os.getpid()}.csv'
                begin=time.monotonic();close_context=torch.cuda.current_device()
                export=state['handle'].activity_end_forced if cfg.get('idle_forced_final_flush') else state['handle'].activity_end
                if cfg.get('idle_drain_all_device_before_export'):
                    torch.cuda.synchronize(runner.device)
                    export=state['handle'].activity_end_drain_first
                if cfg.get('idle_close_device_context'):
                    with torch.cuda.device(runner.device):
                        close_context=torch.cuda.current_device()
                        status=export(str(path).encode())
                else:status=export(str(path).encode())
                capture=dict(start=state['start'],stop=stop,status=status,dropped=state['handle'].activity_dropped(),
                    invalid=state['handle'].activity_invalid(),deferred_flush_s=time.monotonic()-begin,file=path.name,
                    activation=state['activation'],close_current_device=close_context,
                    forced_final_flush=cfg.get('idle_forced_final_flush',False),native_graph_owners=[],
                    all_device_drained_before_export=cfg.get('idle_drain_all_device_before_export',False),
                    collector_build=getattr(state['handle'],'_build_record',None),
                    scope='Graph envelopes plus non-graph kernels. Envelope union is not exclusive kernel union. Complement is a conservative no-recorded-compute lower bound.')
                capture['collector_counters']={key:getattr(state['handle'],'activity_'+key)() for key in
                    ('requests','completions','bytes','parsed','version') if hasattr(state['handle'],'activity_'+key)}
                seen=set()
                for label,graphs in (('native',runner.graphs),('coda',getattr(runner,'coda_graphs',None)),
                        ('packed_prefill',getattr(engine,'_autonomous_prefill_graphs',None))):
                    if graphs is None or id(graphs) in seen:continue
                    seen.add(id(graphs));capture['native_graph_owners'].append(dict(owner=label,
                        replays=getattr(graphs,'replays',0),captures=getattr(graphs,'captures',0),fallbacks=getattr(graphs,'fallbacks',0)))
                invalid_capture=not status==capture['dropped']==capture['invalid']==0
            if state['anchor'] is not None:
                ending=torch.cuda.Event(enable_timing=True);stream=runner.core_stream or torch.cuda.current_stream(runner.device)
                left=time.monotonic();ending.record(stream);ending.synchronize();right=time.monotonic()
                for row in state['operations']:
                    records.append(dict(**{k:v for k,v in row.items() if k not in ('begin','end')},
                        begin_device_ms=state['anchor'].elapsed_time(row['begin']),
                        end_device_ms=state['anchor'].elapsed_time(row['end']),enclosure_ms=row['begin'].elapsed_time(row['end'])))
                capture=dict(anchor_host=state['anchor_host'],end_anchor_host=dict(lower=left,upper=right,midpoint=(left+right)/2),
                    CUDA_anchor_interval_ms=state['anchor'].elapsed_time(ending),
                    submission_samples=state['calls'],all_captured_operations=len(records),
                    scope=('All model-submission CUDA-event enclosures in the bounded window. This includes waits and CPU submission gaps. '
                        'The complement is a lower bound on no-model-compute time, not exclusive physical kernel idle. NIXL/DMA can remain active.'
                        if mode=='submit_events' else 'Sparse CUDA-event enclosures. Not physical kernel activity or a complete idle-gap series.'))
            write(folder/f'idle_observer_{state["role"]}_{os.getpid()}.json',dict(mode=mode,role=state['role'],
                origin=state['origin'],queues=state['queues'],operations=records,capture=capture,
                scope='No in-serving trace export. Readiness and activity classes remain separate.'))
            assert not invalid_capture,'Incomplete activity capture. Diagnostic counters were retained before failure.'
        return original_close(engine)
    StageEngine.__init__,StageEngine._schedule,StageEngine.close=initialize,schedule,close
    ModelRunner._execute=execute
    ModelRunner.submit=submit
