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
    def __init__(self,cfg,costs):self.cfg=cfg;self.costs=costs;self.counter=0
    def static_request_layers(self,index):
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
            else:key=str(min((int(x) for x in table),key=lambda b:abs(b-batch)))
        return scale*float(table[key][str(layers)])
    def choose(self,state):
        mode=self.cfg['policy'];self.counter+=1
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
