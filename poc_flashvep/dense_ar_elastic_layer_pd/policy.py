"""Select a prefix before submission using observed state and calibrated costs."""
from dataclasses import dataclass,field
import math


@dataclass
class Observation:
    now:float
    group_size:int
    P_backlog:int
    D_available:float
    pending_requests:int
    KV_free_blocks:int
    active_decode:int
    earliest_decode_deadline:float|None=None
    state_age_ms:float=0.
    pending_cuts:tuple=()
    admitted_D_requests:int=0


@dataclass
class Choice:
    layers:int=0
    reason:str='NATIVE_FALLBACK'
    candidates:list=field(default_factory=list)


class LayerPolicy:
    def __init__(self,cfg,costs):
        self.cfg=cfg;self.costs=costs;self.counter=0
        self.seen_requests=0;self.selected_requests=0;self.selected_layer_work=0
    def static_request_layers(self,index):
        if 'static_layers_by_request' in self.cfg:
            return self.cfg['static_layers_by_request'][index]
        fraction=self.cfg.get('migration_fraction',1.)
        block=index//self.cfg.get('static_request_block_size',4)
        return self.cfg['static_layers'] if math.floor((block+1)*fraction)>math.floor(block*fraction) else 0
    def decode_bound(self,batch):
        fallback=float(self.costs.get('decode_step_p95_ms',35.))
        if self.cfg.get('decode_prediction')!='BATCH_LOOKUP' or batch<=0:return fallback
        table=self.costs.get('decode_step_by_batch_p95_ms',{})
        sizes=sorted(map(int,table))
        if not sizes:return fallback
        key=next((n for n in sizes if n>=batch),None)
        if key is None:return fallback+self.cfg.get('decode_prediction_margin_ms',2.)
        return float(table[str(key)])+self.cfg.get('decode_prediction_margin_ms',2.)
    def _cost(self,kind,layers,batch):
        table=self.costs[kind];key=str(batch)
        scale=1
        if key not in table:
            largest=max(int(x) for x in table)
            if batch>largest:key=str(largest);scale=math.ceil(batch/largest)
            elif self.cfg.get('revised_revision',1)>=2:
                key=str(min(int(x) for x in table if int(x)>=batch))
            else:key=str(min((int(x) for x in table),key=lambda b:abs(b-batch)))
        return scale*float(table[key][str(layers)])
    def choose(self,state):
        mode=self.cfg['policy'];self.counter+=1
        if mode=='REVISED':
            self.seen_requests+=state.group_size
            return self.revised(state)
        if mode=='FORCED_QUALIFICATION':
            return Choice(self.cfg['forced_layers'][(self.counter-1)%len(self.cfg['forced_layers'])],
                'FORCED_CORRECTNESS_NOT_DYNAMIC',[])
        if mode=='NATIVE':return Choice(reason='NATIVE_CONTROL')
        if mode=='STATIC':
            fraction=self.cfg.get('migration_fraction',1.)
            selected=((self.counter-1)%100)/100<fraction
            # Spread the selected fraction across the run, not only its start.
            selected=math.floor(self.counter*fraction)>math.floor((self.counter-1)*fraction)
            return Choice(self.cfg['static_layers'] if selected else 0,'STATIC_FRACTION',[])
        if state.state_age_ms>self.cfg.get('maximum_state_age_ms',250):return Choice(reason='STALE_D_STATE')
        if state.P_backlog<self.cfg.get('minimum_P_backlog',1):return Choice(reason='NO_P_COMPUTE_BACKLOG')
        if state.KV_free_blocks<state.group_size*math.ceil((self.cfg['length']+self.cfg['cap'])/16):
            return Choice(reason='KV_CREDIT_BOUND')
        batch=state.group_size;full=self._cost('P_prefix_ms',0,batch)
        native_handoff=self._cost('handoff_ms',0,batch)
        native_finish=max(state.now+full/1000,state.D_available)+native_handoff/1000
        guard=self.cfg.get('safety_margin_ms',5.)
        D_busy=bool(state.active_decode or state.admitted_D_requests)
        negotiation=self.costs.get('negotiation_idle_ms',2.) if not D_busy else self.costs.get('negotiation_busy_ms',30.)
        candidates=[]
        for layers in self.cfg.get('layer_candidates',(2,4,8,12)):
            prefix=self._cost('P_prefix_ms',layers,batch)
            handoff=self._cost('handoff_ms',layers,batch)
            suffix=self._cost('D_suffix_ms',layers,batch)
            commit=self._cost('D_commit_ms',layers,batch) if 'D_commit_ms' in self.costs else 0.
            source_ready=state.now+(negotiation+prefix)/1000
            unblocked_ready=source_ready+handoff/1000
            ready=max(source_ready,state.D_available)+handoff/1000
            start=ready
            finish=start+(suffix+commit+guard)/1000
            idle_budget=max(0.,1000*(state.now+full/1000-state.D_available))
            accepted=suffix+guard<=idle_budget
            reason='IDLE_BUDGET' if accepted else 'INSUFFICIENT_IDLE_PREDICTION'
            rejections=[] if accepted else [reason]
            if mode in ('DEADLINE','COST'):
                # Execution also waits for the real Decode cohort to retire.
                # This predictive gate cannot certify unknown future arrivals.
                if D_busy and state.D_available+guard/1000>unblocked_ready:
                    accepted=False;reason='DECODE_BUSY_AT_PREDICTED_HANDOFF'
                    if state.earliest_decode_deadline is not None and ready+(suffix+guard)/1000>state.earliest_decode_deadline:
                        reason='DECODE_DEADLINE_BEFORE_PROPOSED_QUANTUM'
                    rejections.append(reason)
                # The execution path partitions ready work by its real cut.
                # Known pending cohorts already contribute to D_available.
                # A different pending cut is not an unconditional rejection.
            saving=1000*(native_finish-finish)
            if mode=='COST' and saving<=self.cfg.get('minimum_predicted_benefit_ms',0.):
                accepted=False;reason='NO_COMPLETION_BENEFIT'
                rejections.append(reason)
            candidates.append(dict(layers=layers,predicted_prefix_ms=prefix,predicted_handoff_ms=handoff,
                predicted_suffix_ms=suffix,predicted_D_ready=ready,predicted_D_start=start,predicted_D_finish=finish,
                predicted_P_cut_ready=source_ready,predicted_unblocked_D_ready=unblocked_ready,
                predicted_commit_ms=commit,
                predicted_Native_finish=native_finish,predicted_TTFT_saving_ms=saving,
                predicted_P_compute_saved_ms=full-prefix,idle_budget_ms=idle_budget,
                safety_margin_ms=guard,negotiation_bound_ms=negotiation,accepted=accepted,reason=reason,
                rejection_reasons=rejections))
        feasible=[c for c in candidates if c['accepted']]
        if not feasible:return Choice(reason='NO_FEASIBLE_CUT',candidates=candidates)
        best=max(feasible,key=lambda c:(c['predicted_P_compute_saved_ms'],c['predicted_TTFT_saving_ms']))
        return Choice(best['layers'],'ONLINE_STATE_AWARE',candidates)

    def revised(self,state):
        """Score observed-queue benefit inside an empirical execution bound."""
        if state.state_age_ms>self.cfg.get('maximum_state_age_ms',250):return Choice(reason='STALE_D_STATE')
        if state.P_backlog<self.cfg.get('minimum_P_backlog',1):return Choice(reason='NO_P_COMPUTE_BACKLOG')
        blocks=state.group_size*math.ceil((self.cfg['length']+self.cfg['cap'])/16)
        if state.KV_free_blocks<blocks:return Choice(reason='KV_CREDIT_BOUND')
        batch=state.group_size;full=self._cost('P_prefix_ms',0,batch)
        native_handoff=self._cost('handoff_ms',0,batch)
        native_finish=max(state.now+full/1000,state.D_available)+native_handoff/1000
        uncertainty=self.costs.get('uncertainty',{})
        if self.cfg.get('native_continuous_D_ingress'):
            # The Native counterfactual need not wait for cohort retirement.
            # Use its earlier estimate when testing migration economics.
            native_finish=state.now+(full+native_handoff)/1000
        guard=self.cfg.get('safety_margin_ms',uncertainty.get('completion_upper_margin_ms',110.))
        source_uncertainty=uncertainty.get('source_absolute_p95_ms',55.)
        source_bias=uncertainty.get('source_bias_ms',0.)
        drain_guard=uncertainty.get('known_drain_underprediction_p95_ms',70.)
        busy=bool(state.active_decode or state.admitted_D_requests or state.pending_requests)
        available_upper=state.D_available+(drain_guard/1000 if busy else 0.)
        negotiation=self.costs.get('negotiation_busy_ms',30.) if busy else self.costs.get('negotiation_idle_ms',2.)
        next_batch=min(state.P_backlog,self.cfg.get('P_group_size',4))
        next_frontier=min(self._cost('P_prefix_ms',n,next_batch)+self._cost('handoff_ms',n,next_batch)
            for n in (0,*self.cfg.get('layer_candidates',(2,4,8,12))))
        pressure=min(self.cfg.get('maximum_queue_groups',4.),state.P_backlog/batch)
        decode_reserve=self.decode_bound(max(1,state.admitted_D_requests))
        candidates=[];serialized=self.cfg.get('revised_revision',1)>=2
        for layers in self.cfg.get('layer_candidates',(2,4,8,12)):
            guard=self.cfg.get('safety_margin_by_layers_ms',{}).get(str(layers),
                self.cfg.get('safety_margin_ms',uncertainty.get('completion_upper_margin_ms',110.)))
            prefix=self._cost('P_prefix_ms',layers,batch);handoff=self._cost('handoff_ms',layers,batch)
            suffix=self._cost('D_suffix_ms',layers,batch)
            commit=self._cost('D_commit_ms',layers,batch) if 'D_commit_ms' in self.costs else 0.
            source=state.now+(negotiation+prefix+source_bias)/1000
            unblocked=source+handoff/1000;earliest_ready=unblocked-source_uncertainty/1000
            if serialized:
                # The coordinator does not submit a destination request while
                # an existing Decode cohort is active. Transfer cannot start
                # before that ingress gate opens.
                start=max(source,available_upper)+handoff/1000
                mean_start=max(source,state.D_available)+handoff/1000
                predicted_wait=max(0.,1000*(available_upper-(source-source_uncertainty/1000)))
            else:
                start=max(unblocked,available_upper);mean_start=start
                predicted_wait=max(0.,1000*(available_upper-earliest_ready))
            finish_mean=mean_start+(suffix+commit)/1000
            finish_upper=start+(suffix+commit+guard)/1000
            # The next queued request group is already observed. Use its
            # earliest calibrated legal frontier, not a future arrival time.
            next_ready=source+(next_frontier-source_uncertainty)/1000
            budget=1000*(next_ready-start)-decode_reserve
            rejections=[]
            if serialized:
                if predicted_wait>self.cfg.get('maximum_predicted_D_wait_ms',0.):
                    rejections.append('BOUNDED_ADMISSION_WAIT_EXCEEDED')
            elif available_upper>earliest_ready:rejections.append('KNOWN_DECODE_NOT_DRAINED_BEFORE_EARLIEST_HANDOFF')
            if suffix+commit+guard>budget:rejections.append('INSUFFICIENT_KNOWN_INGRESS_WINDOW')
            saved=full-prefix;personal=1000*(native_finish-finish_upper)
            queue_saving=self.cfg.get('queue_benefit_weight',1.)*pressure*saved
            gain=personal+queue_saving
            if gain<=self.cfg.get('minimum_predicted_benefit_ms',0.):rejections.append('NO_AGGREGATE_COMPLETION_BENEFIT')
            request_cap=(math.floor(self.seen_requests*self.cfg['request_budget_fraction'])
                if 'request_budget_fraction' in self.cfg else None)
            layer_cap=(math.floor(self.seen_requests*self.cfg['layer_budget_per_request'])
                if 'layer_budget_per_request' in self.cfg else None)
            if request_cap is not None and self.selected_requests+batch>request_cap:
                rejections.append('MATCHED_REQUEST_BUDGET_CAP')
            if layer_cap is not None and self.selected_layer_work+batch*layers>layer_cap:
                rejections.append('MATCHED_LAYER_WORK_BUDGET_CAP')
            # Account for suffix occupancy and positive incremental handoff.
            occupancy=suffix+commit+max(0.,handoff-native_handoff)+negotiation
            if self.cfg.get('exclude_negotiation_from_D_occupancy'):
                # Waiting for a control reply is already in P-cut readiness.
                # It is not additional D GPU compute occupancy.
                occupancy-=negotiation
            score=gain/max(1.,occupancy)
            candidates.append(dict(layers=layers,predicted_prefix_ms=prefix,predicted_handoff_ms=handoff,
                predicted_suffix_ms=suffix,predicted_commit_ms=commit,predicted_P_cut_ready=source,
                predicted_unblocked_D_ready=unblocked,predicted_D_ready=start,predicted_D_start=start,
                predicted_D_finish=finish_upper,predicted_D_finish_mean=finish_mean,
                predicted_Native_finish=native_finish,predicted_TTFT_saving_ms=personal,
                predicted_P_compute_saved_ms=saved,predicted_queue_saving_ms=queue_saving,
                predicted_aggregate_gain_ms=gain,score=score,idle_budget_ms=budget,
                predicted_D_incremental_cost_ms=occupancy,
                negotiation_counted_as_D_cost=not self.cfg.get('exclude_negotiation_from_D_occupancy',False),
                execution_budget_ms=budget,admission_budget_ms=1000*(native_finish-state.now),
                predicted_known_next_ingress=next_ready,D_available_upper=available_upper,
                source_earliest_ready=earliest_ready,safety_margin_ms=guard,
                predicted_admission_wait_upper_ms=predicted_wait,handoff_can_overlap_existing_decode=not serialized,
                seen_requests=self.seen_requests,selected_request_budget_cap=request_cap,selected_layer_work_budget_cap=layer_cap,
                negotiation_bound_ms=negotiation,earliest_existing_decode_deadline=state.earliest_decode_deadline,
                accepted=not rejections,reason=rejections[0] if rejections else 'EMPIRICAL_SAFE_QUEUE_BENEFIT',
                rejection_reasons=rejections,
                budget_scope='Past/current Host state and empirical bounds. Not physical idle or unknown-future deadline proof.'))
        feasible=[c for c in candidates if c['accepted']]
        if not feasible:return Choice(reason='NO_FEASIBLE_CUT',candidates=candidates)
        best=max(feasible,key=lambda c:(c['score'],c['predicted_aggregate_gain_ms'],-c['layers']))
        self.selected_requests+=batch;self.selected_layer_work+=batch*best['layers']
        return Choice(best['layers'],'ONLINE_BOUNDED_QUEUE_AWARE',candidates)
