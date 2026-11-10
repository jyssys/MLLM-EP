"""Bounded empirical completion reservations. These are not future guarantees."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CompletionPlan:
    feasible: bool
    reason: str
    predicted_ms: float
    execution_ms: float
    quantum_ms: float
    future_quantum_ms: float
    queued_ms: float
    deadline_ms: float


def plan_completion(execution_ms,quantum_ms,current_budget_ms,future_budget_ms,
                    decode_reserve_ms,handoff_ms,queued_ms,deadline_ms):
    values=(execution_ms,quantum_ms,current_budget_ms,future_budget_ms,
            decode_reserve_ms,handoff_ms,queued_ms,deadline_ms)
    if not all(math.isfinite(v) and v>=0 for v in values):
        return CompletionPlan(False,'INVALID_COST',float('inf'),execution_ms,quantum_ms,future_budget_ms,queued_ms,deadline_ms)
    if future_budget_ms<quantum_ms or quantum_ms<=0:
        return CompletionPlan(False,'NO_REPEATABLE_QUANTUM',float('inf'),execution_ms,quantum_ms,future_budget_ms,queued_ms,deadline_ms)
    # Reserve a complete Decode interval between successive opportunities.
    # A small first opportunity does not imply unlimited later opportunities.
    first=min(execution_ms,math.floor(current_budget_ms/quantum_ms)*quantum_ms)
    remaining=max(0.,execution_ms-first)
    future_usable=math.floor(future_budget_ms/quantum_ms)*quantum_ms
    rounds=math.ceil(remaining/future_usable) if remaining else 0
    predicted=handoff_ms+queued_ms+execution_ms+rounds*decode_reserve_ms
    reason='COMPLETION_RESERVED' if predicted<=deadline_ms else 'COMPLETION_DEADLINE_INFEASIBLE'
    return CompletionPlan(predicted<=deadline_ms,reason,predicted,execution_ms,quantum_ms,future_budget_ms,queued_ms,deadline_ms)


def can_reconsider(previous_armed,depth,total,measured,monitor_boundaries):
    return bool(measured and depth<total and (previous_armed or monitor_boundaries))


def execution_margin(target_ms,age_ms,remaining_decode_ms,guard_ms,device_pending=False):
    # Suspension forbids new admissions elsewhere. It must not erase the
    # deadline-safe execution margin of already committed ownership.
    if device_pending:return 0.
    return max(0.,target_ms-age_ms-remaining_decode_ms-guard_ms)


def mean_deadline_remaining(first_output_time,delivered_tokens,now,limit_ms):
    """Price the next cumulative mean with observed output tokens only."""
    if delivered_tokens<1:raise ValueError('A first output is required')
    return max(0.,limit_ms*delivered_tokens-1000*(now-first_output_time))


def empty_epoch_recoverable(reason,active_decode,pending_continuations):
    return reason=='DECODE_TAIL' and active_decode==0 and pending_continuations==0


def suffix_execution_cost(candidates,loops,tokens,quantum_ms,guard_ms,include_prelude=False):
    stages=[('recurrent',False)]*max(0,loops-1)+[('recurrent',True)]
    if include_prelude:stages.insert(0,('prelude',False))
    total=0.;floor=0.
    for operation,terminal in stages:
        allowed=[c for c in candidates if c['operation']==operation and
            c['loops']==(0 if operation=='prelude' else 1) and
            (operation=='prelude' or c['terminal']==terminal) and c['tokens']<=tokens and
            tokens%c['tokens']==0 and c['cost_ms']+guard_ms<=quantum_ms]
        if not allowed:return float('inf'),float('inf')
        chosen=min(allowed,key=lambda c:c['cost_ms']/c['tokens'])
        total+=tokens/chosen['tokens']*chosen['cost_ms']
        floor=max(floor,chosen['cost_ms']+guard_ms)
    return total,floor
