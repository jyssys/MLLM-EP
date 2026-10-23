"""Minimal causal objective plug-in for the existing legal chunk controller."""
from collections import deque
import hashlib
import json
import time
from .objective_features import prepare_items
from .objective_families import choose
from ..ouro_budget_joint_cg.optimizer import BudgetSelector,enumerate_candidates
from ..ouro_budget_joint_cg.actions import ChunkActions,Choice
from ..ouro_budget_joint_cg.admission import BudgetAdmission

class DualSelector(BudgetSelector):
    def __init__(self,model,budget,normalization,mode='JOINT',fixed_C=2048,
            family='O2',weight=1.,endpoint='DUAL',risk_cap=.5,efficiency_floor=.5):
        super().__init__(model,budget,mode,fixed_C)
        self.norm=normalization;self.family=family
        self.kwargs=dict(weight=weight,endpoint=endpoint,risk_cap=risk_cap,efficiency_floor=efficiency_floor)

    def universe(self,state,budget):
        return enumerate_candidates(state,budget,self.model,'JOINT')

    def restrict(self,candidates,state):
        actual=min(self.fixed_C,state['prompt_remaining'],state['token_budget']) if state['kind']=='NEW_CHUNK' else state['C']
        return tuple(c for c in candidates if (self.mode!='G_ONLY' or c.C==actual) and
            (self.mode!='C_ONLY' or c.G==4-state.get('depth',0)))

    def choices(self,state):
        return self.restrict(self.universe(state,self.budget),state)

    def __call__(self,state):
        start=time.perf_counter_ns()
        universe=self.universe(state,self.budget)
        restricted=self.restrict(universe,state)
        fallback=not restricted
        if fallback:
            universe=self.universe(state,float('inf'))
            restricted=self.restrict(universe,state)
        assert restricted
        # Construct objective features on the same universe for every surface.
        # In particular, O1's projections cannot change when a control fixes C.
        items=prepare_items(universe,state,self.norm,self.model)
        allowed=set(restricted);items=[pair for pair in items if pair[0] in allowed]
        if fallback:
            selected=min(restricted,key=lambda c:max(c.current_ms,c.opening_ms));rank=None
        else:
            selected,rank=choose(self.family,items,**self.kwargs)
        row=dict(host_s=time.perf_counter(),state=state,
            state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            budget_ms=self.budget,mode=self.mode,family=self.family,objective_parameters=self.kwargs,
            candidates=[dict(c.__dict__,features=f) for c,f in items],selected=selected.__dict__,score=rank,
            reason='EXPERIMENTAL_OVERBUDGET_PROGRESS_FALLBACK' if fallback else f'{self.mode}_{self.family}_ENUMERATION',
            selector_ns=time.perf_counter_ns()-start)
        self.trace.append(row)
        return Choice(selected.C,selected.G,row['reason'])

class ObservedActions(ChunkActions):
    def __init__(self,engine,selector,telemetry):
        self.telemetry=telemetry;self.snapshot_calls=0;self.snapshot_states={}
        super().__init__(engine,selector)

    def snapshot(self,request,phase,token_budget):
        template,state=super().snapshot(request,phase,token_budget)
        self.snapshot_calls+=1
        if state is not None:
            now=time.perf_counter();t=self.telemetry
            issued=state['issued_decode']
            ages=[1000*(now-t.last[r['id']]) for r in issued if r['id'] in t.last]
            state.update(decode_age_max_ms=max(ages,default=0.),
                prompt_age_ms=max(0.,1000*(now-t.admitted.get(request.request_id,now))),
                TBT_target_ms=t.target_ms,past_TBT_samples=len(t.gaps),
                past_TBT_violation_fraction=sum(g>t.target_ms for g in t.gaps)/max(1,len(t.gaps)),
                previous_C=t.previous_C.get(request.request_id),previous_G=t.previous_G.get(request.request_id),
                telemetry_clock='perf_counter, output delivery from completed native step',
                causal_observed_host_s=now)
            key=(request.request_id,state['start'],state['depth'],phase.value,state['request_event_ready'])
            self.snapshot_states[key]=dict(state)
        return template,state

    def make(self,request,phase,token_budget):
        result=super().make(request,phase,token_budget)
        if phase.value.startswith('prefill') and result.token_count>0:
            self.telemetry.previous_C[request.request_id]=result.token_count
            if phase.value=='prefilling_recurrent': self.telemetry.previous_G[request.request_id]=result.loop_count
        return result

class DualControl:
    def __init__(self,engine,model,normalization,cell):
        self.engine=engine;self.target_ms=cell.get('TBT_target_ms',200.)
        self.admitted={};self.last={};self.counts={};self.gaps=deque(maxlen=256)
        self.previous_C={};self.previous_G={};self.output_events=[]
        self.selector=DualSelector(model,cell['budget_ms'],normalization,cell.get('surface','JOINT'),
            cell.get('fixed_C',2048),cell.get('objective','O2'),cell.get('weight',1.),
            cell.get('endpoint','DUAL'),cell.get('risk_cap',.5),cell.get('efficiency_floor',.5))
        self.actions=ObservedActions(engine,self.selector,self)
        self.admission=BudgetAdmission(engine,self.actions,self.selector)
        self.original_step=engine.step;self.original_add=engine.add_request
        engine.step=self.step;engine.add_request=self.add

    def add(self,request_id,*args,**kwargs):
        assert request_id not in self.admitted,'Reset request lifetime before ID reuse'
        result=self.original_add(request_id,*args,**kwargs)
        self.admitted[request_id]=time.perf_counter()
        return result

    def step(self,*args,**kwargs):
        outputs=self.original_step(*args,**kwargs)
        for output in outputs:
            now=time.perf_counter();rid=output.request_id;count=len(output.token_ids)
            if count>self.counts.get(rid,0):
                assert count==self.counts.get(rid,0)+1
                if rid in self.last: self.gaps.append(1000*(now-self.last[rid]))
                self.last[rid]=now;self.counts[rid]=count
                self.output_events.append(dict(id=rid,host_s=now,tokens=count))
            if output.finished:
                for mapping in (self.admitted,self.last,self.counts,self.previous_C,self.previous_G):mapping.pop(rid,None)
        return outputs

    def reset(self):
        assert not self.engine.has_unfinished_requests()
        for mapping in (self.admitted,self.last,self.counts,self.previous_C,self.previous_G):mapping.clear()
        self.gaps.clear();self.output_events.clear();self.selector.trace.clear();self.actions.decisions.clear()
        self.actions.snapshot_calls=0;self.actions.snapshot_states.clear()
        self.admission.waiting.clear();self.admission.trace.clear()
        self.admission.checks=0;self.admission.total_ns=0;self.admission.max_ns=0

    def close(self):
        if self.engine is None:return
        self.engine.step=self.original_step;self.engine.add_request=self.original_add
        self.admission.close();self.actions.close()
        self.engine=None;self.original_step=None;self.original_add=None
