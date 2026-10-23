"""Single-action 0–1 optimization by enumeration. Credit complete R4 work."""
from dataclasses import dataclass
import time
from .actions import Choice
from .common import CHUNKS,GROUPS

@dataclass(frozen=True)
class Candidate:
    C:int
    G:int
    current_ms:float
    opening_ms:float
    residual_chunk_ms:float
    groups_remaining:int
    complete_tokens_now:int
    utility:float
    path:str
    known:bool

def enumerate_candidates(state,budget,model,mode='JOINT',fixed_C=1024,objective='U2',yield_penalty_ms=0.):
    continuation=state['kind']=='CONTINUE_CHUNK'
    chunks=(state['C'],) if continuation else CHUNKS if mode!='G_ONLY' else (fixed_C,)
    groups=(4,) if mode=='C_ONLY' else GROUPS
    candidates=[];seen=set()
    for proposed_C in chunks:
        c=state['C'] if continuation else min(proposed_C,state['prompt_remaining'],state['token_budget'])
        if c<=0:
            continue
        depth=state['depth'] if continuation else 0
        for proposed_G in groups:
            g=min(proposed_G,4-depth)
            if (c,g) in seen:
                continue
            seen.add((c,g))
            path='atomic' if state.get('atomic',False) else 'staged_core'
            if path=='atomic' and (g!=4 or depth):
                continue
            context=state['start']+c
            current=model.estimate(path,c,g,context,depth)
            prelude=model.estimate('staged_prelude',c,0,context,0) if not continuation and path!='atomic' else (0.,True)
            full=current[0]+prelude[0]
            remaining=4-depth-g;at=depth+g;count=1;known=current[1] and prelude[1]
            while remaining:
                next_g=min(g,remaining)
                cost,supported=model.estimate('staged_core',c,next_g,context,at)
                full+=cost;known=known and supported
                remaining-=next_g;at+=next_g;count+=1
            complete=c if depth+g==4 else 0
            occupancy=max(current[0],prelude[0])
            # A staged opening has two native dispatch units. Both must fit B.
            # The full-chunk objective still pays their total and every missing loop.
            total_with_yields=full+max(0,count-1)*yield_penalty_ms
            if objective=='U0':
                utility=c*g/4 # Diagnostic partial-work-only objective.
            elif objective=='U1':
                utility=float(complete)+.01*c*g/4
            elif objective=='U2':
                utility=c/max(.001,total_with_yields)
            else:
                raise ValueError('Unregistered objective')
            if occupancy<=budget:
                candidates.append(Candidate(c,g,current[0],prelude[0],full,count,complete,utility,path,known))
    return tuple(candidates)

class BudgetSelector:
    def __init__(self,model,budget,mode='JOINT',fixed_C=1024,objective='U2',yield_penalty_ms=0.,starvation_ms=250.):
        self.model=model;self.budget=budget;self.mode=mode;self.fixed_C=fixed_C;self.objective=objective
        self.yield_penalty_ms=yield_penalty_ms;self.starvation_ms=starvation_ms;self.trace=[]

    def choices(self,state):
        return enumerate_candidates(state,self.budget,self.model,self.mode,self.fixed_C,self.objective,
            self.yield_penalty_ms if state.get('issued_decode') else 0.)

    def __call__(self,state):
        started=time.perf_counter_ns();candidates=self.choices(state)
        if not candidates:
            # The phase policy must normally defer this prefill and run decode.
            # A committed starvation/idle fallback is explicit and may overrun B.
            fallback=enumerate_candidates(state,float('inf'),self.model,self.mode,self.fixed_C,self.objective,0.)
            assert fallback,'No legal supported positive-span action'
            chosen=min(fallback,key=lambda c:max(c.current_ms,c.opening_ms))
            reason='EXPERIMENTAL_OVERBUDGET_PROGRESS_FALLBACK'
        else:
            chosen=max(candidates,key=lambda c:(c.utility,c.complete_tokens_now,-c.residual_chunk_ms,c.G))
            reason=f'{self.mode}_{self.objective}_FINITE_ENUMERATION'
        self.trace.append(dict(host_s=time.perf_counter(),state=state,budget_ms=self.budget,mode=self.mode,
            candidates=[c.__dict__ for c in candidates],selected=chosen.__dict__,reason=reason,
            selector_ns=time.perf_counter_ns()-started))
        return Choice(chosen.C,chosen.G,reason)
