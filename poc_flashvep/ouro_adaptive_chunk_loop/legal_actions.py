"""Offline state/action contracts. This module does not change RLT execution."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Frontier:
    request_id:str
    prompt_remaining:int
    depth:int
    staged_tokens:int
    token_budget:int
    free_token_capacity:int
    prefix_valid:bool
    one_chunk_lease:bool=True

@dataclass(frozen=True)
class Action:
    C:int
    G:int
    depth:int
    kind:str
    token_loop_progress:int
    completed_frontier_progress:int

def enumerate_actions(frontier,chunk_menu=(64,128,256,512,768,1024,1536,2048),group_menu=(1,2,3,4)):
    if not frontier.prefix_valid or not frontier.one_chunk_lease:
        return ()
    if not 0<=frontier.depth<4 or frontier.prompt_remaining<=0:
        return ()
    if frontier.staged_tokens:
        if frontier.staged_tokens>frontier.token_budget:
            # This PoC does not repartition an existing staged span.
            return ()
        chunks=(frontier.staged_tokens,)
        kind='PREFILL_CONTINUE'
    else:
        if frontier.depth!=0:
            return ()
        chunks=sorted(set(min(c,frontier.prompt_remaining,frontier.token_budget,frontier.free_token_capacity) for c in chunk_menu))
        kind='PREFILL_START'
    actions=[]
    seen=set()
    for c in chunks:
        if c<=0:
            continue
        for proposed in group_menu:
            g=min(proposed,4-frontier.depth)
            if g<=0 or (c,g) in seen:
                continue
            seen.add((c,g))
            actions.append(Action(c,g,frontier.depth,kind,c*g,c if frontier.depth+g==4 else 0))
    return tuple(actions)

@dataclass(frozen=True)
class DecodeNow:
    request_id:str
    phase:str
    completed_R:int
    deadline_ms:float
    KV_ready:bool=True

def complete_next_token_schedule(states,now_ms,phase_cost):
    """Decode-only bounded reference DAG. No future tokens or arrivals exist here.

    phase_cost receives an actual present batch and returns a declared cost.
    The caller must label estimated costs separately from measured execution.
    """
    pending={r.request_id:[r.phase,r.completed_R,r] for r in states if r.phase!='finished'}
    assert all(r.KV_ready for r in states),'Unpublished KV cannot be consumed'
    assert all(0<=r.completed_R<=4 for r in states)
    completions={}
    timeline=[]
    clock=float(now_ms)
    turns=0
    while pending:
        coda=[rid for rid,(phase,depth,r) in pending.items() if phase=='coda' or (phase=='recurrent' and depth==4)]
        prelude=[rid for rid,(phase,depth,r) in pending.items() if phase=='prelude']
        recurrent=[rid for rid,(phase,depth,r) in pending.items() if phase=='recurrent' and depth<4]
        if coda:
            kind,ids='coda',coda
        elif prelude:
            kind,ids='prelude',prelude
        elif recurrent:
            kind,ids='recurrent',recurrent
        else:
            raise ValueError('No legal next-token task in current state')
        cost=float(phase_cost(kind,tuple(pending[rid][2] for rid in ids)))
        assert cost>=0,'Negative durations are not legal'
        timeline.append(dict(phase=kind,ids=ids,start_ms=clock,end_ms=clock+cost))
        clock+=cost
        for rid in ids:
            if kind=='coda':
                completions[rid]=clock
                del pending[rid]
            elif kind=='prelude':
                pending[rid][0]='recurrent'
                pending[rid][1]=0
            else:
                pending[rid][1]+=1
        turns+=1
        assert turns<=3*len(states)+8,'Bounded next-token schedule did not converge'
    return completions,timeline

def candidate_feasible(states,now_ms,quantum_safe_ms,phase_cost,jitter_margin_ms=0.):
    completions,timeline=complete_next_token_schedule(states,now_ms+quantum_safe_ms,phase_cost)
    safe=all(completions[r.request_id]+jitter_margin_ms<=r.deadline_ms for r in states if r.phase!='finished')
    return safe,completions,timeline
