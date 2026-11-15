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
    affected_deadlines:tuple=()
    own_TTFT_deadline:float|None=None
    maximum_group_output_tokens:int=0
    input_token_lengths:tuple=()
    P_ready_external:int=0
    P_ready_inside_macro:int=0
    D_preparation_overdue:bool=False
    own_ITL_limit_ms:float|None=None
    group_output_caps:tuple=()
    group_request_ids:tuple=()
    own_TTFT_deadlines:tuple=()
    D_available_nominal:float|None=None


@dataclass
class Choice:
    layers:int=0
    reason:str='NATIVE_FALLBACK'
    candidates:list=field(default_factory=list)


class LayerPolicy:
    def __init__(self,cfg,costs):
        self.cfg=cfg;self.costs=costs;self.counter=0
        self.seen_requests=0;self.selected_requests=0;self.selected_layer_work=0
        self.current_length=None
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
        model=self.costs
        if self.current_length is not None and self.current_length!=8000:
            model=self.costs['length_calibration'][str(self.current_length)]
        table=model[kind];key=str(batch)
        scale=1
        if key not in table:
            largest=max(int(x) for x in table)
            if batch>largest:key=str(largest);scale=math.ceil(batch/largest)
            elif self.cfg.get('revised_revision',1)>=2:
                key=str(min(int(x) for x in table if int(x)>=batch))
            else:key=str(min((int(x) for x in table),key=lambda b:abs(b-batch)))
        return scale*float(table[key][str(layers)])
    def destination_plan(self,layers,batch,cap):
        maximum=self.cfg.get('D_admission_batch',batch)
        sizes=[min(maximum,batch-offset) for offset in range(0,batch,maximum)]
        return dict(batches=sizes,
            handoff_ms=sum(self._cost('handoff_ms',layers,n) for n in sizes),
            suffix_ms=sum(self._cost('D_suffix_ms',layers,n) for n in sizes),
            commit_ms=sum(self._cost('D_commit_ms',layers,n) for n in sizes) if 'D_commit_ms' in self.costs else 0.,
            intercohort_decode_ms=sum(self.decode_bound(n)*max(0,cap-1) for n in sizes[:-1]),
            last_cohort_decode_ms=self.decode_bound(sizes[-1])*max(0,cap-1),
            maximum_decode_step_bound_ms=max(self.decode_bound(n) for n in sizes))
    def choose(self,state):
        mode=self.cfg['policy'];self.counter+=1
        if mode=='DIRECT':
            self.seen_requests+=state.group_size
            if state.D_preparation_overdue:return Choice(reason='D_COMMITTED_PREPARATION_OVERRUN_NATIVE_FALLBACK')
            lengths=set(state.input_token_lengths)
            self.current_length=next(iter(lengths)) if len(lengths)==1 else None
            if lengths and (len(lengths)!=1 or (self.current_length!=8000 and
                str(self.current_length) not in self.costs.get('length_calibration',{}))):
                return Choice(reason='UNCALIBRATED_INPUT_GEOMETRY_NATIVE_FALLBACK')
            from .deadline import direct_choice
            return direct_choice(self,state)
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
        if state.P_backlog<self.cfg.get('minimum_P_backlog',1) and not (
            self.cfg['policy']=='DIRECT' and self.cfg.get('allow_zero_queue_personal_benefit')):
            return Choice(reason='NO_P_COMPUTE_BACKLOG')
        blocks=state.group_size*math.ceil((self.cfg['length']+self.cfg['cap'])/16)
        if state.KV_free_blocks<blocks:return Choice(reason='KV_CREDIT_BOUND')
        batch=state.group_size;full=self._cost('P_prefix_ms',0,batch)
        native_handoff=self._cost('handoff_ms',0,batch)
        native_finish=max(state.now+full/1000,state.D_available)+native_handoff/1000
        planned=self.cfg.get('bounded_destination_batches') and self.cfg['policy']=='DIRECT'
        if planned:
            native_plan=self.destination_plan(0,batch,state.maximum_group_output_tokens)
            native_handoff=native_plan['handoff_ms']
            native_finish=max(state.now+full/1000,state.D_available)+(
                native_handoff+native_plan['intercohort_decode_ms'])/1000
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
        next_batch=max(1,min(state.P_backlog,self.cfg.get('P_group_size',4)))
        next_frontier=min(self._cost('P_prefix_ms',n,next_batch)+self._cost('handoff_ms',n,next_batch)
            for n in (0,*self.cfg.get('layer_candidates',(2,4,8,12))))
        benefit_backlog=state.P_ready_external if (self.cfg['policy']=='DIRECT' and
            self.cfg.get('queue_benefit_external_only')) else state.P_backlog
        pressure=min(self.cfg.get('maximum_queue_groups',4.),benefit_backlog/batch)
        decode_reserve=self.decode_bound(max(1,state.admitted_D_requests))
        candidates=[];serialized=self.cfg.get('revised_revision',1)>=2
        for layers in self.cfg.get('layer_candidates',(2,4,8,12)):
            guard=self.cfg.get('safety_margin_by_layers_ms',{}).get(str(layers),
                self.cfg.get('safety_margin_ms',uncertainty.get('completion_upper_margin_ms',110.)))
            prefix=self._cost('P_prefix_ms',layers,batch);handoff=self._cost('handoff_ms',layers,batch)
            suffix=self._cost('D_suffix_ms',layers,batch)
            commit=self._cost('D_commit_ms',layers,batch) if 'D_commit_ms' in self.costs else 0.
            intercohort=0.;plan=None
            if planned:
                plan=self.destination_plan(layers,batch,state.maximum_group_output_tokens)
                handoff=plan['handoff_ms'];suffix=plan['suffix_ms'];commit=plan['commit_ms']
                intercohort=plan['intercohort_decode_ms']
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
            finish_mean=mean_start+(suffix+commit+intercohort)/1000
            finish_upper=start+(suffix+commit+intercohort+guard)/1000
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
            occupancy=suffix+commit+intercohort+max(0.,handoff-native_handoff)+negotiation
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
            if planned:candidates[-1].update(destination_plan=plan,predicted_intercohort_decode_wait_ms=intercohort)
        feasible=[c for c in candidates if c['accepted']]
        if not feasible:return Choice(reason='NO_FEASIBLE_CUT',candidates=candidates)
        best=max(feasible,key=lambda c:(c['score'],c['predicted_aggregate_gain_ms'],-c['layers']))
        self.selected_requests+=batch;self.selected_layer_work+=batch*best['layers']
        return Choice(best['layers'],'ONLINE_BOUNDED_QUEUE_AWARE',candidates)

