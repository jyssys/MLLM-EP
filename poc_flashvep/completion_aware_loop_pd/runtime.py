"""Process-local boundary negotiation. Native transport guards stay enabled."""
import os
import time
from collections import deque,defaultdict
from dataclasses import replace
from .protocol import Boundary,legal_loops,prefix_budget,matches_quantum,initial_token_prefix,token_reservation_path,distinct_P_pressure,take_ready_batch,forecast_quantum,offer_completed_boundary
from .common import ROOT,write
from .completion import plan_completion,can_reconsider,execution_margin,mean_deadline_remaining,empty_epoch_recoverable,suffix_execution_cost


def install(cfg):
    import torch
    from vllm_rlt.pd.engine import PDEngine
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    from ..native_loop_2d_oracle import native_path
    Base=native_path.ACTIVE_CONNECTOR
    assert Base is not None,'The qualified native connector must load first'
    folder=ROOT/'raw'/cfg['name'];records=[]
    R=cfg['depth'];token=cfg['intervention_method']=='dynamic_token'
    token_family=bool(cfg.get('native_frontier') and cfg['native_frontier'][0]==0)
    dynamic=cfg['intervention_method'] in ('dynamic_loop','dynamic_tile','dynamic_token')
    total=cfg['length'] if token else R
    checkpoints=cfg.get('boundary_checkpoints',([total-512,total-256,total-128,total-64,total-32]
        if token else [1,2,3] if R==4 else [24,28,30,31]))
    NativeBase=Base.__mro__[1] if token_family else Base
    decode_phases=(P.CODA,P.PRELUDE,P.RECURRENT)
    prefill_phases=(P.PREFILLING_RECURRENT,P.PREFILLING_PRELUDE,P.PREFILL)
    def log(event,**fields):
        records.append(dict(time=time.monotonic(),event=event,**fields))
        if event in ('GPU_COMPLETED_BOUNDARY_OFFER','SOURCE_COMMIT','DESTINATION_COMMIT','SOURCE_CONTINUE'):
            import json
            with (folder/f'protocol_live_{os.getpid()}.jsonl').open('a') as handle:
                handle.write(json.dumps(records[-1])+'\n')
    def flush():
        write(folder/f'intervention_{os.getpid()}.json',dict(records=records,
            scope='Host protocol and scheduling observations. Not physical GPU kernel intervals.'))
    def measured(request):
        index=request.sampling_params.seed-cfg['seed']
        return 0<=index<cfg['count'] and not request.request_id.startswith('native_graph_')

    def advertised_plan(ad,remaining):
        future=ad.get('future_quantum_budget_ms',ad.get('budget_ms',0.))
        work,floor=suffix_execution_cost(cfg.get('execution_candidates',[]),R if token else remaining,
            remaining if token else cfg['length'],future,cfg.get('guard_ms',1.),token)
        return plan_completion(work,floor,ad.get('budget_ms',0.),future,
            ad.get('decode_reserve_ms',cfg.get('decode_reserve_ms',0.)),
            cfg.get('handoff_upper_ms',50.),0.,cfg.get('completion_deadline_ms',350.))

    # The coordinator relays generation-scoped messages. It does not predict
    # future arrivals or change the existing transport ACK/release path.
    init_pd,message_pd=PDEngine.__init__,PDEngine._message
    def credit_clear_waiters(engine,force=False):
        cached=getattr(engine,'_idle_pressure_cached',None);now=time.monotonic()
        if not force and cached is not None and now-cached[0]<.05:return cached[1]
        count=0
        for pending in engine.transfers.values():
            if pending.cancelled or pending.phase!='waiting':continue
            fits=True
            for role in ('prefill','decode'):
                length=len(pending.request.prompt_token_ids)+(pending.request.sampling_params.max_tokens-1 if role=='decode' else 0)
                fits= fits and any(peer.role==role and
                    (role=='prefill' or peer.slots<peer.client.info['transfer_slots']) and
                    peer.reserved_blocks+engine._blocks(peer,length)<=peer.client.info['num_blocks']-
                        (peer.client.info['watermark_blocks'] if peer.slots else 0)
                    for peer in engine.peers.values())
            count+=bool(fits)
        engine._idle_pressure_cached=(now,count)
        return count
    def initialized_pd(engine,*args,**kwargs):
        result=init_pd(engine,*args,**kwargs)
        write(folder/f'effective_PD_config_{os.getpid()}.json',dict(
            shutdown_timeout=engine.config.shutdown_timeout,startup_timeout=engine.config.startup_timeout,
            request_timeout=engine.config.request_timeout,max_inflight_bytes=engine.config.max_inflight_bytes,
            max_draining_requests=engine.config.max_draining_requests,
            scope='Resolved native PDConfig. A similarly named top-level experiment field does not configure this object.'))
        engine._online_frontiers={}
        for peer in engine.peers.values():
            original=peer.client.send
            def sending(kind,_send=original,**fields):
                frontier=engine._online_frontiers.get(fields.get('tid'))
                if frontier is not None and kind in ('reserve','prefill','commit'):
                    fields['arc_cut']=R if token else frontier
                    if token:fields['idle_prefix']=frontier
                pending=engine.transfers.get(fields.get('tid'))
                if pending is not None and pending.request.request_id.startswith('idle_shape_warm_') and kind in ('reserve','prefill','commit'):
                    width=int(pending.request.request_id.rsplit('_',1)[1])
                    token_family=bool(cfg.get('native_frontier') and cfg['native_frontier'][0]==0)
                    fields['arc_cut']=R if token_family else R-1
                    fields['idle_warm_width']=width
                    if token_family:fields['prefix_k']=len(pending.request.prompt_token_ids)-width
                return _send(kind,**fields)
            peer.client.send=sending
        return result
    def messaged(engine,peer,m):
        kind=m['kind']
        if kind=='idle_capacity':
            assert peer.role=='decode'
            fields={k:v for k,v in m.items() if k!='kind'}
            fields['P_credit_clear_waiters']=credit_clear_waiters(engine)
            for destination in engine.peers.values():
                if destination.role=='prefill':engine._send(destination.name,kind,**fields)
            return
        if not kind.startswith('idle_'):
            if token and kind=='commit' and 'idle_prefix' in m:
                engine._online_frontiers[m['tid']]=m['idle_prefix']
            return message_pd(engine,peer,m)
        w=engine.transfers.get(m.get('tid'))
        if w is None or w.cancelled:return
        assert peer.name in (w.p,w.d),'Negotiation message from an unrelated owner'
        fields={k:v for k,v in m.items() if k!='kind'}
        if kind=='idle_offer':
            assert peer.name==w.p
            fields['stage_P_ready_leases']=fields.get('P_ready_leases',fields['P_pressure'])
            fields['stage_P_pending_device_requests']=fields.get('P_pending_device_requests',0)
            fields['credit_clear_P_capacity_waiters']=credit_clear_waiters(engine,True)
            fields['P_pressure']+=fields['credit_clear_P_capacity_waiters']
            fields['coordinator_waiting_or_prefill']=sum(not x.cancelled and x.phase in ('waiting','prefill')
                for x in engine.transfers.values())
            engine._send(w.d,'idle_offer',**fields)
        elif kind in ('idle_prepared','idle_rejected','idle_committed'):
            assert peer.name==w.d
            if kind=='idle_committed':engine._online_frontiers[w.tid]=m['depth']
            engine._send(w.p,kind,**fields)
        elif kind in ('idle_commit','idle_cancel_lease'):
            assert peer.name==w.p
            engine._send(w.d,kind,**fields)
        else:raise RuntimeError('Unknown idle protocol message: '+kind)
        log('COORDINATOR_RELAY',kind=kind,**fields)
    PDEngine.__init__,PDEngine._message=initialized_pd,messaged
    close_pd=PDEngine.close
    def closed_pd(engine,*args,**kwargs):
        try:return close_pd(engine,*args,**kwargs)
        finally:flush()
    PDEngine.close=closed_pd

    class OnlineConnector(Base):
        @classmethod
        def create(cls,engine,*,config,notify):
            self=super().create(engine,config=config,notify=notify)
            self.boundaries={};self.pending_callbacks={};self.leases={};self.committed_loans=set()
            self.last_outputs={};self.intervals=deque(maxlen=32);self.credit=True
            self.first_outputs={};self.output_counts={}
            self.suspended=False;self.last_quantum_event=None
            self.suspension_reason=None
            self.last_quantum_time=0.;self.current_choice=None
            self.advertisement=None;self.last_advertised=0.
            self.commit_epochs={}
            self.last_supply_sample=0.
            self.pending_quantum_record=None
            self.last_quantum_submit_time=0.
            self.decode_intervals_by_batch=defaultdict(lambda:deque(maxlen=128))
            self.measurement_started=False
            self.completion_reservations={}
            self.completion_missed=set()
            self.handoff_copy_stream=(torch.cuda.Stream(device=self.cache.device)
                if cfg.get('dedicated_handoff_copy_stream') and not engine.stage_config.produces_tokens else None)
            self.prefill_execution_grant=False
            if token:
                previous=self.notify
                def notifying(kind,**fields):
                    if kind=='commit':fields['idle_prefix']=self.transfers[fields['tid']].prefix
                    return previous(kind,**fields)
                self.notify=notifying
            return self

        def reserve(self,tid,tokens,*args,**kwargs):
            native_zero=(token_family and
                token_reservation_path(self._next_frontier_prefix,len(tokens))=='native')
            if native_zero:
                result=NativeBase.reserve(self,tid,tokens,*args,**kwargs)
                t=self.transfers.get(tid)
                if t is not None:
                    t.prefix=len(tokens)
                    if t.outgoing:
                        t.request.native_staircase_prefix=t.prefix
                        t.request.native_staircase_c=0
                log('NATIVE_ZERO_SUFFIX_RESERVATION',tid=tid,prompt_tokens=len(tokens),
                    scope='Full Native reservation, not a relaxed partial-frontier validity check.')
            else:result=super().reserve(tid,tokens,*args,**kwargs)
            t=self.transfers.get(tid)
            if t is not None and getattr(self,'_next_warm_width',None) is not None:
                t.request.idle_warm_width=self._next_warm_width
            if token and t is not None and t.outgoing:
                request=t.request
                if hasattr(request,'idle_warm_width'):
                    request.idle_armed=False
                else:
                    request.idle_armed=(None if cfg.get('defer_token_arming') else measured(request) and self.can_arm())
                    t.prefix=checkpoints[0] if request.idle_armed else len(tokens)
                request.native_staircase_prefix=t.prefix
            return result

        def can_arm(self):
            if cfg.get('qualification_allow_unpressured',False):return True
            ad=self.advertisement
            ready_ids=self.ready_P_ids();pending_ids=self.pending_P_device_ids()
            ready=len(ready_ids);pending=len(pending_ids)
            clear=ad.get('P_credit_clear_waiters',0) if ad else 0
            self.last_arm_evaluation=dict(P_ready=ready,P_credit_clear_waiters=clear,
                P_pending_device_requests=pending,
                P_distinct_stage_work=distinct_P_pressure(ready_ids,pending_ids),
                D_available=ad.get('available') if ad else None,
                D_execution_budget_ms=ad.get('budget_ms') if ad else None,
                D_future_quantum_budget_ms=ad.get('future_quantum_budget_ms') if ad else None,
                advertisement_age_ms=(time.monotonic()-ad['observed'])*1000 if ad else None)
            if cfg.get('source_monitor_boundaries'):
                # Candidate monitoring is not destination admission. A busy D
                # must not permanently erase later legal source boundaries.
                self.last_arm_evaluation['scope']='Checkpoint-capable source. D admission remains independent.'
                return True
            return bool(ad and time.monotonic()-ad['observed']<=cfg.get('advertisement_max_age_ms',50.)/1000
                and ad['available'] and (ad.get('future_quantum_budget_ms',ad['budget_ms'])
                    if cfg.get('predictive_admission') else ad['budget_ms'])>=cfg.get('minimum_quantum_ms',1.)
                and distinct_P_pressure(ready_ids,pending_ids,clear)>=cfg.get('minimum_P_pressure',2))

        def pending_P_device_ids(self):
            return {item.request.request_id for ticket in self.engine._inflight
                if ticket.batch.phase in prefill_phases and not ticket.ready()
                for item in ticket.batch.requests}

        def pending_P_device_requests(self):
            return len(self.pending_P_device_ids())

        def ready_pressure(self,executable=False):
            return len(self.ready_P_ids(executable))

        def ready_P_ids(self,executable=False):
            pending=self.pending_P_device_ids() if executable else set()
            return {t.request.request_id for t in self.transfers.values() if t.outgoing and not t.cancelled and not t.compute_done and
                t.request.phase in (P.WAITING,*prefill_phases) and Base.can_schedule(self,t.request)
                and (not executable or t.request.request_id not in pending and t.tid not in self.pending_callbacks and
                    (t.tid not in self.boundaries or self.boundaries[t.tid].status=='running'))}

        def accept_cut(self,request,hidden,*,cached_tokens=0):
            if token_family and self.transfers[request.request_id].prefix==len(request.prompt_token_ids):
                return NativeBase.accept_cut(self,request,hidden,cached_tokens=cached_tokens)
            return super().accept_cut(request,hidden,cached_tokens=cached_tokens)

        def active_decode(self):
            return [r for r in self.engine.scheduler.requests.values() if r.phase in decode_phases]

        def deadline_margin(self):
            now=time.monotonic();active=self.active_decode()
            if self.suspended and active and not cfg.get('completion_reservations'):return 0.
            if self.last_quantum_event is not None and not self.last_quantum_event.query():return 0.
            if not active:return cfg.get('empty_decode_budget_ms',0.)
            if not self.credit and not cfg.get('multiple_quanta_per_observed_deadline'):return 0.
            last=[self.last_outputs.get(r.request_id) for r in active]
            if any(t is None for t in last):return 0.
            target=cfg.get('native_ITL_target_ms',40. if R==4 else 120.)
            age=max((now-t)*1000 for t in last)
            deadline_remaining=target-age
            if cfg.get('additional_work_TPOT_limit_ms'):
                limit=cfg['additional_work_TPOT_limit_ms']
                for request in active:
                    rid=request.request_id
                    if rid not in self.first_outputs:return 0.
                    # The next cumulative mean uses only delivered tokens.
                    # It does not assume a future output length or arrival.
                    next_interval=max(1,self.output_counts.get(rid,1))
                    deadline_remaining=min(deadline_remaining,
                        mean_deadline_remaining(self.first_outputs[rid],next_interval,now,limit))
            reserve=self.reserve_for_batch(len(active))
            if cfg.get('device_decode_costs'):
                # Logical progress can lead device completion. Retain one
                # additional Loop plus a host/boundary guard for pending work.
                pending=any(not ticket.ready() and ticket.batch.phase in decode_phases
                    for ticket in self.engine._inflight)
                remaining=max((R if r.phase==P.PRELUDE else 0 if r.phase==P.CODA
                    else max(0,R-r.loops_done) for r in active),default=R)
                costs=cfg['device_decode_costs']
                reserve=(min(R+1,remaining+int(pending))*costs['loop_p99_ms']
                    +costs['boundary_guard_ms'])
            guard=cfg.get('decode_guard_ms',cfg.get('guard_ms',2.))
            return execution_margin(deadline_remaining,0.,reserve,guard)

        def reserve_for_batch(self,batch):
            if cfg.get('device_decode_costs'):
                costs=cfg['device_decode_costs']
                return R*costs['loop_p99_ms']+costs['boundary_guard_ms']
            reserve=cfg.get('initial_decode_reserve_by_batch',{}).get(str(batch),
                cfg.get('decode_reserve_ms',35. if R==4 else 105.))
            if cfg.get('adaptive_decode_reserve'):
                history=self.decode_intervals_by_batch[batch]
                if len(history)>=cfg.get('minimum_decode_history',16):
                    ordered=sorted(history)
                    reserve=max(cfg.get('minimum_decode_reserve_ms',0.),
                        ordered[min(len(ordered)-1,int(.95*len(ordered)))]+cfg.get('decode_prediction_guard_ms',1.))
            return reserve

        def future_quantum_budget(self):
            if self.suspended:return 0.
            # A forecast after a future output is not an execution grant now.
            # It uses current batch state and past service samples only.
            active_count=len(self.active_decode())
            if cfg.get('reserve_known_decode_transitions'):
                # Accepted continuations become Decode work after completion.
                # Their existence is already known, unlike future arrivals.
                active_count+=sum(tid in self.transfers and self.transfers[tid].request.phase
                    in (P.RECEIVING,*prefill_phases) for tid in self.committed_loans)
            batch=max(1,active_count)
            target=cfg.get('native_ITL_target_ms',40. if R==4 else 120.)
            if cfg.get('additional_work_TPOT_limit_ms'):target=min(target,cfg['additional_work_TPOT_limit_ms'])
            return forecast_quantum(active_count,target,self.reserve_for_batch(batch),
                cfg.get('decode_guard_ms',cfg.get('guard_ms',2.)),cfg.get('maximum_quantum_ms',8.),
                self.budget() if cfg.get('empty_admission_uses_current_cap') and not active_count else None)

        def budget(self):
            margin=self.deadline_margin()
            cap=(cfg.get('empty_quantum_cap_ms',cfg.get('empty_decode_budget_ms',0.))
                if not self.active_decode() else cfg.get('maximum_quantum_ms',8.))
            return min(margin,cap)

        def admission_budget(self):
            if not cfg.get('separate_admission_budget'):return self.budget()
            if cfg.get('predictive_admission') and cfg.get('admission_horizon_ms'):
                quantum=self.future_quantum_budget()
                cycle=self.reserve_for_batch(max(1,len(self.active_decode())))+quantum
                return quantum*cfg['admission_horizon_ms']/max(cycle,1.)
            multiplier=(cfg.get('empty_admission_window_multiplier',1.) if not self.active_decode()
                else cfg.get('admission_window_multiplier',1.))
            return (self.future_quantum_budget() if cfg.get('predictive_admission') else self.deadline_margin())*multiplier

        def completion_quantum_floor(self,remaining):
            if not cfg.get('phase_aware_costs'):return cfg.get('minimum_quantum_ms',1.)
            loops=R if token else remaining;tokens=remaining if token else cfg['length']
            candidates=[c for c in cfg['execution_candidates'] if c['tokens']<=tokens and c['loops']<=loops]
            terminal=[c['cost_ms'] for c in candidates if c['operation']=='recurrent' and c['terminal']]
            if not terminal:return float('inf')
            costs=[min(terminal)]
            if loops>1:
                core=[c['cost_ms'] for c in candidates if c['operation']=='recurrent' and not c['terminal']]
                if not core:return float('inf')
                costs.append(min(core))
            if token:
                prelude=[c['cost_ms'] for c in candidates if c['operation']=='prelude']
                if not prelude:return float('inf')
                costs.append(min(prelude))
            return max(costs)+cfg.get('guard_ms',1.)

        def completion_work(self,remaining,quantum):
            # Price the actual permitted execution geometry, not the serial
            # full-width Loop when only small quanta fit the forecast budget.
            return suffix_execution_cost(cfg.get('execution_candidates',[]),R if token else remaining,
                remaining if token else cfg['length'],quantum,cfg.get('guard_ms',1.),token)

        def completion_plan(self,remaining):
            future=self.future_quantum_budget()
            work,floor=self.completion_work(remaining,future)
            now=time.monotonic()
            queued=sum((v['estimated_remaining_ms']*(1+self.reserve_for_batch(max(1,len(self.active_decode())))/max(future,1.))
                +(cfg.get('handoff_upper_ms',50.) if self.transfers[tid].request.phase==P.RECEIVING else 0.)
                for tid,v in self.completion_reservations.items()
                if tid in self.committed_loans and tid in self.transfers
                and self.transfers[tid].request.phase in (P.RECEIVING,*prefill_phases)))
            deadline=cfg.get('completion_deadline_ms',350.)
            return plan_completion(work,floor,self.budget(),future,self.reserve_for_batch(max(1,len(self.active_decode()))),
                cfg.get('handoff_upper_ms',50.),queued,deadline)

        def can_schedule(self,request):
            b=self.boundaries.get(request.request_id)
            if b is not None and b.status!='running':return False
            if request.request_id in self.pending_callbacks:return False
            if not self.engine.stage_config.produces_tokens:return super().can_schedule(request)
            if request.phase in prefill_phases:
                if not self.prefill_execution_grant:return False
                if hasattr(request,'idle_warm_width'):
                    assert not (folder/'measurement_begin.json').exists(),'Startup request entered measured scheduling'
                    return super().can_schedule(request)
                if cfg.get('execution_candidates') and self.current_choice is None:return False
                if (cfg.get('phase_aware_costs') and self.current_choice is not None
                        and self.current_choice.get('tid')!=request.request_id):return False
                live_budget=self.budget()
                cost=(self.current_choice.get('cost_ms') or 0.) if self.current_choice is not None else 0.
                if live_budget<max(cfg.get('minimum_quantum_ms',1.),cost+cfg.get('guard_ms',1.)):return False
            return super().can_schedule(request)

        def prefill_submitted(self,request,*,phase,start,progress,event,stream):
            t=self.transfers[request.request_id]
            depth=(request.num_prefilled_tokens if token else
                sum(left==len(request.prompt_token_ids) for left,right in request.prefill_depth_ranges))
            if (dynamic and measured(request) and not t.compute_done and depth<total
                    and getattr(request,'idle_armed',False)):
                assert event is not None,'Online boundary negotiation requires a native CUDA completion event'
                if token and depth<t.prefix:
                    t.last_event=event;return
                if not token and not offer_completed_boundary(depth,getattr(request,'idle_planned_boundary',depth),checkpoints):
                    log('SOURCE_INTERMEDIATE_GROUPED_BOUNDARY',tid=t.tid,depth=depth,
                        target=request.idle_planned_boundary)
                    return super().prefill_submitted(request,phase=phase,start=start,
                        progress=progress,event=event,stream=stream)
                assert depth in checkpoints,'Unexpected source checkpoint'
                t.last_event=event
                self.pending_callbacks[t.tid]=dict(request=request,phase=phase,start=start,
                    progress=progress,event=event,stream=stream,depth=depth)
                return
            if token_family and t.prefix==len(request.prompt_token_ids):
                # A zero-suffix fallback is a full native export. It must not
                # ask Huginn's suffix exporter for an unused initial-state map.
                return NativeBase.prefill_submitted(self,request,phase=phase,start=start,
                    progress=progress,event=event,stream=stream)
            partial=t.cut<R or getattr(t,'prefix',len(request.prompt_token_ids))<len(request.prompt_token_ids)
            if self.handoff_copy_stream is not None and not dynamic and partial:
                stream=self.handoff_copy_stream
            return super().prefill_submitted(request,phase=phase,start=start,
                progress=progress,event=event,stream=stream)

        def command(self,m):
            kind=m['kind']
            if kind in ('reserve','prefill'):self._next_warm_width=m.get('idle_warm_width')
            if kind=='idle_capacity':
                assert not self.engine.stage_config.produces_tokens
                self.advertisement=m;return True
            if token and kind=='commit' and 'idle_prefix' in m:
                t=self.transfers[m['tid']];prefix=m['idle_prefix']
                if t.prefix!=prefix:
                    assert prefix==len(t.request.prompt_token_ids) and t.tid not in self.committed_loans
                    assert not t.outgoing and not t.ready and t.request.phase==P.RECEIVING
                    t.prefix=prefix
            if not kind.startswith('idle_'):return super().command(m)
            tid=m['tid'];t=self.transfers.get(tid);epoch=m['epoch'];now=time.monotonic()
            if t is None or t.cancelled:return True
            if kind=='idle_offer':
                assert not t.outgoing
                # Receiving ownership already includes a complete KV lease.
                # No second full-request reservation is required.
                occupied=len(self.leases)+sum(x in self.transfers and self.transfers[x].request.phase
                    in (P.RECEIVING,*prefill_phases)
                    for x in self.committed_loans)
                budget=self.budget();admission_budget=self.admission_budget();remaining=total-m['depth']
                admission_quantum=self.future_quantum_budget() if cfg.get('predictive_admission') else budget
                cost=cfg.get('remaining_cost_ms',{}).get(str(remaining),float('inf'))
                mincost=(self.completion_quantum_floor(remaining) if cfg.get('predictive_admission')
                    else cfg.get('minimum_quantum_ms',1.))
                allowed=(t.request.phase==P.RECEIVING and not t.received and not t.ready
                    and not self.suspended
                    and occupied<cfg.get('max_continuations',1) and admission_quantum>=mincost
                    and now<=m['expires'] and (m['P_pressure']>=cfg.get('minimum_P_pressure',2)
                    or cfg.get('qualification_allow_unpressured',False)))
                # Admission must include remaining completion and communication.
                admission=cost+cfg.get('handoff_upper_ms',50.)
                allowed=allowed and admission<=cfg.get('admission_limit_ms',250.)
                if cfg.get('require_whole_suffix_budget',False):allowed=allowed and cost<=admission_budget
                completion=None
                if cfg.get('completion_reservations'):
                    completion=self.completion_plan(remaining)
                    allowed=allowed and completion.feasible
                if cfg.get('force_destination_reject'):
                    allowed=False
                reason=('EXPIRED_OFFER' if now>m['expires'] else
                    'DECODE_SAFETY_SUSPENDED' if self.suspended else
                    'CONTINUATION_BACKLOG_LIMIT' if occupied>=cfg.get('max_continuations',1) else
                    'DECODE_NO_EXECUTION_BUDGET' if admission_quantum<mincost else
                    'NO_P_READY_BACKLOG' if m['P_pressure']<cfg.get('minimum_P_pressure',2)
                        and not cfg.get('qualification_allow_unpressured',False) else
                    'SUFFIX_COMPLETION_TOO_COSTLY' if admission>cfg.get('admission_limit_ms',250.) else
                    'REMAINING_SUFFIX_TOO_LARGE' if cfg.get('require_whole_suffix_budget',False) and cost>admission_budget else
                    'DESTINATION_STATE_NOT_RECEIVING')
                if cfg.get('force_destination_reject'):reason='CONTROL_FORCED_REJECT'
                if completion is not None and not completion.feasible:reason=completion.reason
                if allowed:
                    self.leases[tid]=dict(epoch=epoch,depth=m['depth'],expires=m['expires'])
                    if completion is not None:
                        self.leases[tid]['completion_plan']=completion
                    self.notify('idle_prepared',tid=tid,epoch=epoch,depth=m['depth'],budget_ms=budget,admission_budget_ms=admission_budget)
                else:self.notify('idle_rejected',tid=tid,epoch=epoch,depth=m['depth'],reason=reason,budget_ms=budget)
                log('ADMISSION',tid=tid,epoch=epoch,depth=m['depth'],remaining=remaining,
                    accepted=allowed,budget_ms=budget,admission_budget_ms=admission_budget,
                    future_quantum_budget_ms=admission_quantum,
                    remaining_cost_ms=cost,occupied=occupied,completion_quantum_floor_ms=mincost if mincost<float('inf') else None,
                    stage_P_ready_leases=m.get('stage_P_ready_leases'),
                    stage_P_pending_device_requests=m.get('stage_P_pending_device_requests'),
                    credit_clear_P_capacity_waiters=m.get('credit_clear_P_capacity_waiters'),
                    rejection_reason=None if allowed else reason)
                if completion is not None:
                    log('COMPLETION_PLAN',tid=tid,epoch=epoch,remaining=remaining,
                        **{k:(v if not isinstance(v,float) or v<float('inf') else None)
                            for k,v in vars(completion).items()})
            elif kind=='idle_cancel_lease':
                lease=self.leases.get(tid)
                if lease and lease['epoch']==epoch:self.leases.pop(tid)
            elif kind=='idle_prepared':
                b=self.boundaries.get(tid)
                if b is not None and b.prepared(epoch,now):
                    self.notify('idle_commit',tid=tid,epoch=epoch,depth=b.depth)
                else:
                    self.notify('idle_cancel_lease',tid=tid,epoch=epoch)
            elif kind=='idle_commit':
                if self.commit_epochs.get(tid)==epoch:
                    assert (t.prefix if token else t.cut)==m['depth'],'Duplicate commit changed its frontier'
                    self.notify('idle_committed',tid=tid,epoch=epoch,depth=m['depth']);return True
                lease=self.leases.get(tid)
                valid=(lease is not None and lease['epoch']==epoch and lease['depth']==m['depth']
                    and now<=lease['expires'] and t.request.phase==P.RECEIVING and not t.received
                    and not self.suspended and (self.future_quantum_budget() if cfg.get('predictive_admission')
                        else self.budget())>=cfg.get('minimum_quantum_ms',1.))
                if cfg.get('predictive_admission'):
                    valid=valid and self.future_quantum_budget()>=self.completion_quantum_floor(total-m['depth'])
                if cfg.get('require_whole_suffix_budget'):
                    valid=valid and cfg['remaining_cost_ms'][str(total-m['depth'])]<=self.admission_budget()
                completion=self.completion_plan(total-m['depth']) if cfg.get('completion_reservations') else None
                if completion is not None:valid=valid and completion.feasible
                if not valid:
                    self.leases.pop(tid,None)
                    self.notify('idle_rejected',tid=tid,epoch=epoch,reason='STALE_OR_EXPIRED_COMMIT')
                else:
                    assert t.cut==R and not t.ready,'Destination cut can change only before import'
                    if token:t.prefix=m['depth']
                    else:t.cut=m['depth'];t.request.arc_received_cut=t.cut
                    self.leases.pop(tid);self.committed_loans.add(tid)
                    self.commit_epochs[tid]=epoch
                    if completion is not None:
                        self.completion_reservations[tid]=dict(committed=now,last_progress=now,
                            predicted_finish=now+completion.predicted_ms/1000,
                            estimated_remaining_ms=completion.execution_ms,
                            deadline=now+completion.deadline_ms/1000,remaining_rows=(total-m['depth'])*cfg['length']
                                if not token else (total-m['depth'])*R)
                    self.notify('idle_committed',tid=tid,epoch=epoch,depth=m['depth'])
                    log('DESTINATION_COMMIT',tid=tid,epoch=epoch,depth=m['depth'])
            elif kind=='idle_committed':
                b=self.boundaries[tid]
                if not b.committed(epoch):return True
                assert t.outgoing and t.cut==R and not t.compute_done
                callback=self.pending_callbacks.pop(tid)
                if self.handoff_copy_stream is not None:
                    callback['stream']=self.handoff_copy_stream
                if token:t.prefix=b.depth;t.request.native_staircase_prefix=b.depth
                else:t.cut=b.depth;t.request.arc_export_cut=b.depth
                log('SOURCE_COMMIT',tid=tid,epoch=epoch,depth=b.depth,remaining=total-b.depth,token_split=token,
                    seed=t.request.sampling_params.seed)
                Base.prefill_submitted(self,**{k:v for k,v in callback.items() if k!='depth'})
            elif kind=='idle_rejected':
                b=self.boundaries.get(tid)
                if b is not None and b.reject(epoch):
                    self.pending_callbacks.pop(tid,None)
                    if m.get('reason') in ('DECODE_SAFETY_SUSPENDED','NO_P_READY_BACKLOG'):
                        t.request.idle_armed=False
                    elif not token and m.get('reason') in ('CONTINUATION_BACKLOG_LIMIT','DECODE_NO_EXECUTION_BUDGET'):
                        # Keep one late re-evaluation. Run the intervening loops
                        # as a native grouped submission instead of polling each.
                        t.request.idle_next_checkpoint=checkpoints[-1] if b.depth<checkpoints[-1] else total
                    if cfg.get('source_monitor_boundaries'):
                        t.request.idle_armed=True
                        t.request.idle_next_checkpoint=next((c for c in checkpoints if c>b.depth),total)
                    if token:
                        if cfg.get('token_single_offer_fallback'):t.request.idle_armed=False
                        if not t.request.idle_armed:t.prefix=total
                        elif m.get('reason') in ('CONTINUATION_BACKLOG_LIMIT','DECODE_NO_EXECUTION_BUDGET'):
                            t.prefix=checkpoints[-1] if b.depth<checkpoints[-1] else total
                        else:t.prefix=next((c for c in checkpoints if c>b.depth),total)
                        t.request.native_staircase_prefix=t.prefix
                    log('SOURCE_CONTINUE',tid=tid,epoch=epoch,depth=b.depth,reason=m.get('reason'))
            else:raise RuntimeError('Unknown connector idle message')
            return True

        def progress(self):
            result=super().progress();now=time.monotonic()
            if self.pending_quantum_record is not None and self.last_quantum_event is not None and self.last_quantum_event.query():
                pending=self.pending_quantum_record;self.pending_quantum_record=None
                tid=(pending.get('choice') or {}).get('tid')
                reservation=self.completion_reservations.get(tid)
                if reservation is not None:
                    reservation['last_progress']=now
                    reservation['remaining_rows']=max(0,reservation['remaining_rows']-pending['rows']*pending['loops'])
                    reservation['estimated_remaining_ms']=max(0.,reservation['estimated_remaining_ms']-
                        ((pending.get('choice') or {}).get('cost_ms') or 0.))
                log('D_QUANTUM_HOST_COMPLETION',**pending,
                    elapsed_host_ms=(now-pending['submitted_host_time'])*1000,
                    scope='Host enclosure from ModelRunner call entry through observed event completion. Includes submission and polling. Not exclusive GPU execution time.')
            for tid,reservation in self.completion_reservations.items():
                t=self.transfers.get(tid)
                if t is None or t.request.phase not in (P.RECEIVING,*prefill_phases):continue
                if now>reservation['deadline'] and tid not in self.completion_missed:
                    self.completion_missed.add(tid)
                    # Preserve committed ownership and exact completion. A
                    # missed empirical reservation is evidence of failure.
                    self.suspended=True
                    self.suspension_reason='COMPLETION_RESERVATION_MISSED'
                    log('COMPLETION_RESERVATION_MISSED',tid=tid,
                        age_ms=(now-reservation['committed'])*1000,
                        progress_age_ms=(now-reservation['last_progress'])*1000,
                        remaining_rows=reservation['remaining_rows'])
            marker=folder/'measurement_begin.json'
            if not self.measurement_started and marker.exists():
                self.measurement_started=True
                # Startup byte hashing is excluded from clean serving. Its
                # feedback must not silently disable the later measured arm.
                assert not self.active_decode(),'Measurement began before startup decode drained'
                self.suspended=False;self.intervals.clear();self.last_outputs.clear();self.credit=True
                self.suspension_reason=None
                self.first_outputs.clear();self.output_counts.clear()
                self.decode_intervals_by_batch.clear()
                log('MEASUREMENT_FEEDBACK_RESET')
            if (cfg.get('recover_empty_decode_epoch') and self.engine.stage_config.produces_tokens
                    and self.suspended and empty_epoch_recoverable(self.suspension_reason,
                        len(self.active_decode()),sum(tid in self.transfers and
                            self.transfers[tid].request.phase in (P.RECEIVING,*prefill_phases)
                            for tid in self.committed_loans))):
                # No Decode deadline or committed continuation remains.
                # A past tail event must not disable every later empty epoch.
                self.suspended=False;self.suspension_reason=None;self.intervals.clear();self.credit=True
                log('EMPTY_DECODE_EPOCH_RECOVERY',scope='Past requests finished. New admission still requires cost, KV, and ownership checks.')
            if (self.engine.stage_config.produces_tokens and self.transfers
                    and now-self.last_advertised>=cfg.get('advertisement_interval_ms',20.)/1000):
                occupied=len(self.leases)+sum(tid in self.transfers and self.transfers[tid].request.phase
                    in (P.RECEIVING,*prefill_phases) for tid in self.committed_loans)
                budget=self.budget()
                self.notify('idle_capacity',observed=now,budget_ms=budget,
                    future_quantum_budget_ms=self.future_quantum_budget() if cfg.get('predictive_admission') else budget,
                    admission_budget_ms=self.admission_budget(),
                    decode_reserve_ms=self.reserve_for_batch(max(1,len(self.active_decode()))),
                    available=not self.suspended and occupied<cfg.get('max_continuations',1),decode_batch=len(self.active_decode()))
                self.last_advertised=now
            if now-self.last_supply_sample>=.05:
                self.last_supply_sample=now
                log('SUPPLY_SAMPLE',role=self.engine.stage_config.role,
                    ready_P_leases=self.ready_pressure() if not self.engine.stage_config.produces_tokens else None,
                    executable_P_ready=self.ready_pressure(True) if not self.engine.stage_config.produces_tokens else None,
                    pending_P_device_requests=self.pending_P_device_requests() if not self.engine.stage_config.produces_tokens else None,
                    P_boundary_waiting=len(self.pending_callbacks) if not self.engine.stage_config.produces_tokens else None,
                    Decode_ready=sum(len(self.engine.scheduler.queues[p]) for p in decode_phases),
                    active_Decode=len(self.active_decode()),KV_free_blocks=self.cache.num_free_blocks,
                    KV_total_blocks=self.cache.num_blocks,
                    predicted_budget_ms=self.budget() if self.engine.stage_config.produces_tokens else None)
            for tid,callback in list(self.pending_callbacks.items()):
                t=self.transfers.get(tid)
                if t is None or t.cancelled:continue
                b=self.boundaries.setdefault(tid,Boundary())
                event=callback['event']
                if b.status=='running' and event.query():
                    if cfg.get('source_local_admission_filter'):
                        ad=self.advertisement
                        age=(now-ad['observed'])*1000 if ad else None
                        reason=('NO_FRESH_CAPACITY_ADVERTISEMENT' if ad is None or
                            age>cfg.get('advertisement_max_age_ms',50.) else
                            'ADVERTISED_DESTINATION_UNAVAILABLE' if not ad['available'] else
                            'ADVERTISED_QUANTUM_TOO_SMALL' if ad.get('future_quantum_budget_ms',ad['budget_ms'])
                                <cfg.get('minimum_quantum_ms',1.) else None)
                        if reason is None and cfg.get('source_shared_completion_filter'):
                            forecast=advertised_plan(ad,total-callback['depth'])
                            if not forecast.feasible:reason='ADVERTISED_'+forecast.reason
                        if reason is not None:
                            # No offer exists, so no destination lease can own
                            # this boundary. Reevaluate at the next legal cut.
                            self.pending_callbacks.pop(tid)
                            next_cut=next((c for c in checkpoints if c>callback['depth']),total)
                            t.request.idle_next_checkpoint=next_cut
                            if token:
                                t.prefix=total if cfg.get('token_single_offer_fallback') else next_cut
                                t.request.native_staircase_prefix=t.prefix
                                if cfg.get('token_single_offer_fallback'):t.request.idle_armed=False
                            log('SOURCE_LOCAL_BOUNDARY_SKIP',tid=tid,depth=callback['depth'],
                                next_checkpoint=t.prefix if token else next_cut,reason=reason,
                                advertisement_age_ms=age,
                                scope='No destination offer or ownership change. Later legal Loop boundaries remain eligible.')
                            continue
                    epoch=b.offer(callback['depth'],True,now,cfg.get('negotiation_ttl_ms',50.)/1000)
                    ready_ids=self.ready_P_ids();pending_ids=self.pending_P_device_ids()
                    ready=len(ready_ids);pending=len(pending_ids)
                    self.notify('idle_offer',tid=tid,epoch=epoch,depth=b.depth,expires=b.expires,
                        P_pressure=distinct_P_pressure(ready_ids,pending_ids),
                        P_ready_leases=ready,P_pending_device_requests=pending)
                    log('GPU_COMPLETED_BOUNDARY_OFFER',tid=tid,epoch=epoch,depth=b.depth,
                        native_event_present=True,native_event_complete=True)
                elif b.status=='offered' and now>b.expires:
                    self.notify('idle_cancel_lease',tid=tid,epoch=b.epoch)
                    b.reject(b.epoch);self.pending_callbacks.pop(tid)
                    if token:
                        t.prefix=(total if cfg.get('token_single_offer_fallback') else
                            next((c for c in checkpoints if c>b.depth),total))
                        if cfg.get('token_single_offer_fallback'):t.request.idle_armed=False
                        t.request.native_staircase_prefix=t.prefix
                    log('SOURCE_LEASE_TIMEOUT',tid=tid,depth=b.depth)
                # A commit in flight must not fall back. Ownership is unresolved.
            for tid,lease in list(self.leases.items()):
                if now>lease['expires']:self.leases.pop(tid)
            return result

        def close(self):
            flush();return super().close()

        def _forget(self,t):
            result=super()._forget(t)
            self.pending_callbacks.pop(t.tid,None);self.boundaries.pop(t.tid,None)
            self.leases.pop(t.tid,None);self.committed_loans.discard(t.tid)
            self.commit_epochs.pop(t.tid,None)
            self.completion_reservations.pop(t.tid,None)
            return result
    native_path.ACTIVE_CONNECTOR=OnlineConnector

    init_stage=StageEngine.__init__
    def initialized_stage(engine,*args,**kwargs):
        result=init_stage(engine,*args,**kwargs)
        role=engine.stage_config.role;scheduler=engine.scheduler
        planned=scheduler._planned_loop_count;make=scheduler._make_scheduled_request
        if role=='prefill' and dynamic and not token:
            def source_planned(items):
                if all(hasattr(item.request,'idle_warm_width') for item in items):return planned(items)
                if all(item.phase in (P.PREFILL,P.PREFILLING_RECURRENT) for item in items):
                    choices=[]
                    for item in items:
                        r=item.request
                        depth=0 if item.phase==P.PREFILL else item.depth
                        reconsider=(cfg.get('source_monitor_boundaries') and measured(r)
                            and not getattr(r,'idle_armed',False) and depth<R)
                        if not hasattr(r,'idle_armed') or reconsider:
                            r.idle_armed=measured(r) and engine.connector.can_arm()
                            if measured(r):log('SOURCE_ARM_DECISION',seed=r.sampling_params.seed,
                                tid=r.request_id,armed=r.idle_armed,
                                **getattr(engine.connector,'last_arm_evaluation',{}))
                        target=(getattr(r,'idle_next_checkpoint',None) or next((c for c in checkpoints if c>depth),R)) if r.idle_armed else R
                        if target<=depth:target=next((c for c in checkpoints if c>depth),R)
                        r.idle_planned_boundary=target
                        choices.append(target-depth)
                    return min(choices)
                return planned(items)
            scheduler._planned_loop_count=source_planned
        if role=='prefill' and token:
            def source_token_span(request,phase,token_budget):
                if request.idle_armed is None:
                    connector=engine.connector;t=connector.transfers[request.request_id]
                    request.idle_armed=measured(request) and connector.can_arm()
                    if measured(request):log('SOURCE_ARM_DECISION',seed=request.sampling_params.seed,
                        tid=request.request_id,armed=request.idle_armed,
                        **getattr(connector,'last_arm_evaluation',{}))
                    ad=connector.advertisement or {};budget=ad.get('admission_budget_ms',ad.get('budget_ms',0.))
                    possible=[k for k in checkpoints if cfg.get('remaining_cost_ms',{}).get(str(total-k),float('inf'))<=budget]
                    if cfg.get('source_shared_completion_filter'):
                        fresh=ad and time.monotonic()-ad['observed']<=cfg.get('advertisement_max_age_ms',50.)/1000
                        possible=([k for k in checkpoints if advertised_plan(ad,total-k).feasible]
                            if fresh and ad.get('available') else [])
                    if cfg.get('initial_token_budget_selection') and not possible:request.idle_armed=False
                    t.prefix=initial_token_prefix(len(request.prompt_token_ids),
                        possible if cfg.get('initial_token_budget_selection') else [checkpoints[0]],request.idle_armed)
                    request.native_staircase_prefix=t.prefix
                    log('TOKEN_INITIAL_BOUNDARY_SELECTION',tid=t.tid,seed=request.sampling_params.seed,
                        prefix=t.prefix,admission_budget_ms=budget,armed=request.idle_armed)
                if phase in prefill_phases and hasattr(request,'native_staircase_prefix'):
                    start=request.num_preluded_tokens if phase==P.PREFILLING_PRELUDE else request.num_prefilled_tokens
                    token_budget=prefix_budget(start,request.native_staircase_prefix,token_budget)
                return make(request,phase,token_budget)
            scheduler._make_scheduled_request=source_token_span
        if role=='decode':
            def shaped(request,phase,token_budget):
                choice=getattr(engine.connector,'current_choice',None) if engine.connector is not None else None
                width=getattr(request,'idle_warm_width',choice['tokens'] if choice else cfg.get('D_tile_tokens',cfg['length']))
                if cfg.get('qualification_tile_widths') and measured(request):
                    assert cfg['state_probe'],'Forced width coverage is only a state qualification'
                    width=cfg['qualification_tile_widths'][(request.sampling_params.seed-cfg['seed'])%len(cfg['qualification_tile_widths'])]
                if phase in prefill_phases:token_budget=min(token_budget,width)
                item=make(request,phase,token_budget)
                if cfg.get('execution_candidates') and phase in prefill_phases and not hasattr(request,'idle_warm_width'):
                    origin=getattr(request,'prefill_window_origin',0)
                    widths=[w for w in cfg['D_candidate_widths'] if w<=item.token_count
                        and (item.token_start-origin)%w==0]
                    if widths:item=replace(item,token_count=max(widths))
                if phase==P.PREFILLING_RECURRENT:
                    item=replace(item,prefill_hidden_row_offset=item.token_start)
                return item
            scheduler._make_scheduled_request=shaped
            def bounded(items):
                if all(item.phase==P.PREFILLING_RECURRENT for item in items):
                    legal=min(legal_loops(i.request.prefill_depth_ranges,i.token_start,
                        i.token_start+i.token_count,i.depth) for i in items)
                    choice=getattr(engine.connector,'current_choice',None) if engine.connector is not None else None
                    limit=choice['loops'] if choice else cfg.get('D_loop_quantum',1)
                    return min(legal,limit)
                if any(item.phase==P.PREFILLING_PRELUDE for item in items):return 0
                return planned(items)
            scheduler._planned_loop_count=bounded
            policy=scheduler.policy;old_schedule=policy.schedule
            def schedule(owner,*,prefer_recurrent=False):
                connector=engine.connector
                ready_loan=any(t.ready and (t.request.phase in prefill_phases or
                    t.request.phase==P.RECEIVING and getattr(t.request,'import_resume_phase',None) in prefill_phases)
                    for t in connector.transfers.values())
                connector.prefill_execution_grant=False
                if not ready_loan:
                    connector.current_choice=None
                    return old_schedule(owner,prefer_recurrent=prefer_recurrent)
                owner._admit()
                warm=[r for r in owner.requests.values() if hasattr(r,'idle_warm_width') and r.phase in prefill_phases]
                if warm:
                    assert not (folder/'measurement_begin.json').exists()
                    connector.current_choice=dict(tokens=warm[0].idle_warm_width,loops=1,
                        cost_ms=None,scope='EXACT_STARTUP_SHAPE_WARMUP_NOT_ONLINE_ADMISSION')
                    connector.prefill_execution_grant=True
                    for phase in prefill_phases:
                        if owner.has_ready(phase):return owner._take(phase)
                # A bounded, output-credit quantum can precede the next decode
                # only when the live estimator admits its non-preemptive cost.
                budget=connector.budget() if connector is not None else 0.
                candidates=cfg.get('execution_candidates')
                connector.current_choice=None
                if candidates:
                    if cfg.get('phase_aware_costs'):
                        for phase in prefill_phases:
                            safe=[]
                            for request in owner.requests.values():
                                if request.phase!=phase or not Base.can_schedule(connector,request):continue
                                for c in candidates:
                                    if c['cost_ms']+cfg.get('guard_ms',1.)>budget:continue
                                    if (phase==P.PREFILLING_PRELUDE)!=(c['operation']=='prelude'):continue
                                    item=make(request,phase,c['tokens'])
                                    legal=(legal_loops(request.prefill_depth_ranges,item.token_start,
                                        item.token_start+item.token_count,item.depth)
                                        if phase==P.PREFILLING_RECURRENT else 0)
                                    if not matches_quantum(c,'prelude' if phase==P.PREFILLING_PRELUDE else 'recurrent',
                                            item.token_count,item.depth or 0,legal,R):continue
                                    safe.append(dict(c,tid=request.request_id))
                            if safe:
                                if cfg.get('completion_reservations'):
                                    connector.current_choice=max(safe,key=lambda c:(
                                        -connector.completion_reservations.get(c['tid'],{}).get('deadline',float('inf')),
                                        c['saved_ms']/max(c['cost_ms'],.001),c['tokens']*max(1,c['loops'])))
                                else:
                                    connector.current_choice=max(safe,key=lambda c:(c['saved_ms']/max(c['cost_ms'],.001),
                                        c['tokens']*max(1,c['loops'])))
                                break
                    else:
                        safe=[c for c in candidates if c['cost_ms']+cfg.get('guard_ms',1.)<=budget]
                        connector.current_choice=max(safe,key=lambda c:(c['saved_ms']/max(c['cost_ms'],.001),
                            c['tokens']*c['loops']),default=None)
                eligible=budget>=cfg.get('minimum_quantum_ms',1.) and (not candidates or connector.current_choice is not None)
                if eligible:
                    if connector.current_choice is not None:
                        connector.current_choice=dict(connector.current_choice,budget_at_selection_ms=budget,
                            decode_batch_at_selection=len(connector.active_decode()),
                            decode_ready_at_selection=sum(len(owner.queues[p]) for p in decode_phases))
                    connector.prefill_execution_grant=True
                    batch=take_ready_batch(owner,prefill_phases)
                    if batch is not None:return batch
                    log('PREFILL_TAKE_EMPTY_FALLBACK_TO_NATIVE_POLICY',
                        budget_at_selection_ms=budget,budget_after_take_ms=connector.budget())
                connector.prefill_execution_grant=False
                return old_schedule(owner,prefer_recurrent=prefer_recurrent)
            policy.schedule=schedule
            step=engine.step
            def stepped():
                try:outputs=step()
                except Exception:
                    connector=engine.connector
                    try:
                        write(folder/f'scheduler_failure_{os.getpid()}.json',dict(
                            time=time.monotonic(),budget_ms=connector.budget(),suspended=connector.suspended,
                            choice=connector.current_choice,prefill_execution_grant=connector.prefill_execution_grant,
                            queues={str(phase):list(queue) for phase,queue in engine.scheduler.queues.items()},
                            requests=[dict(tid=r.request_id,phase=str(r.phase),prefilled=r.num_prefilled_tokens,
                                prelude=r.num_preluded_tokens,frontiers=r.prefill_depth_ranges,
                                origin=getattr(r,'prefill_window_origin',None))
                                for r in engine.scheduler.requests.values()]))
                        flush()
                    except Exception:pass
                    raise
                connector=engine.connector;now=time.monotonic()
                for output in outputs:
                    if not output.token_ids:continue
                    previous=connector.last_outputs.get(output.request_id)
                    if previous is not None:
                        interval=(now-previous)*1000;connector.intervals.append(interval)
                        batch=max(1,len(connector.active_decode()))
                        no_borrow=not (previous<=connector.last_quantum_submit_time<=now)
                        if no_borrow:connector.decode_intervals_by_batch[batch].append(interval)
                        if cfg.get('adaptive_decode_reserve'):
                            log('DECODE_INTERVAL_SAMPLE',batch=batch,interval_ms=interval,
                                used_for_native_reserve=no_borrow)
                        limit=65.08287642151119 if R==4 else 192.8212538175284
                        feedback_limit=min(limit,cfg.get('feedback_ITL_limit_ms',limit))
                        if (not cfg.get('qualification_ignore_observer_tail_latch') and not connector.suspended
                                and (interval>2*feedback_limit or sum(x>feedback_limit for x in connector.intervals)>=3)):
                            connector.suspended=True
                            connector.suspension_reason='DECODE_TAIL'
                            log('EXTRA_WORK_SUSPENDED',ITL_ms=interval,limit_ms=feedback_limit,
                                frozen_absolute_limit_ms=limit)
                    connector.last_outputs[output.request_id]=now;connector.credit=True
                    connector.first_outputs.setdefault(output.request_id,now)
                    connector.output_counts[output.request_id]=len(output.token_ids)
                    if output.finished:
                        connector.last_outputs.pop(output.request_id,None)
                        connector.first_outputs.pop(output.request_id,None)
                        connector.output_counts.pop(output.request_id,None)
                return outputs
            engine.step=stepped
        write(folder/f'intervention_contract_{os.getpid()}.json',dict(role=role,
            method=cfg['intervention_method'],checkpoints=checkpoints,
            D_tile_tokens=cfg.get('D_tile_tokens',cfg['length']),
            scope='Process-local hooks. Original recurrent progress and transport validity guards remain enabled.'))
        return result
    StageEngine.__init__=initialized_stage

    from vllm_rlt.worker.model_runner import ModelRunner
    submit=ModelRunner.submit
    def submitted(runner,scheduled_batch=None,*,batch):
        engine=getattr(runner,'_intervention_engine',None)
        prompt=([s for s in batch.spans if s.chunk_start is not None]
            if engine is not None and engine.stage_config.role=='decode' else [])
        submission_begin=time.monotonic() if prompt else None
        result=submit(runner,scheduled_batch,batch=batch)
        # ModelRunner does not own the connector. Associate it during the first
        # engine submission through the engine's stable model_runner reference.
        if engine is not None and engine.stage_config.role=='decode':
            if prompt:
                connector=engine.connector;connector.credit=False
                if prompt:connector.last_quantum_event=runner.events.get(prompt[-1].request.request_id)
                connector.pending_quantum_record=dict(submitted_host_time=submission_begin,
                    submit_host_ms=(time.monotonic()-submission_begin)*1000,timing_revision=2,
                    rows=batch.num_rows,loops=batch.loop_count,choice=connector.current_choice)
                connector.last_quantum_submit_time=connector.pending_quantum_record['submitted_host_time']
                log('D_QUANTUM_SUBMITTED',rows=batch.num_rows,loops=batch.loop_count,
                    choice=connector.current_choice,
                    spans=[dict(tid=s.request.request_id,start=s.token_start,count=s.token_count,depth=s.depth)
                        for s in prompt])
        return result
    ModelRunner.submit=submitted
    stage_installed=StageEngine.__init__
    def associate(engine,*args,**kwargs):
        result=stage_installed(engine,*args,**kwargs)
        engine.model_runner._intervention_engine=engine;return result
    StageEngine.__init__=associate
    close_stage=StageEngine.close
    def closed(engine):
        flush();return close_stage(engine)
    StageEngine.close=closed
