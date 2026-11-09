"""Pure protocol and causal-frontier rules used by the GPU hooks."""
from dataclasses import dataclass


@dataclass
class Boundary:
    epoch: int=0
    depth: int=0
    status: str='running'
    expires: float=0.

    def offer(self, depth, complete, now, ttl):
        if self.status!='running' or not complete or depth<=self.depth:
            raise ValueError('Offer requires a new completed GPU boundary')
        self.epoch+=1;self.depth=depth;self.expires=now+ttl;self.status='offered'
        return self.epoch

    def prepared(self, epoch, now):
        if epoch!=self.epoch or self.status!='offered' or now>self.expires:return False
        self.status='committing';return True

    def committed(self, epoch):
        if epoch==self.epoch and self.status=='committed':return False
        if epoch!=self.epoch or self.status!='committing':raise ValueError('Unmatched cut commit')
        self.status='committed';return True

    def reject(self, epoch):
        if epoch!=self.epoch or self.status not in ('offered','committing'):return False
        self.status='running';return True


def legal_loops(ranges,start,end,depth):
    """A completed prefix and producer within this submission form a down-set."""
    if not 0<=start<end:return 0
    if ranges[depth][0]!=start or ranges[depth][1]<end:return 0
    count=1
    for left,right in ranges[depth+1:]:
        if left!=start:break
        count+=1
    return count


def choose(candidates,budget):
    """Candidates contain measured conservative execution costs, not row proxies."""
    valid=[c for c in candidates if c['cost_ms']<=budget]
    return max(valid,key=lambda c:(c['saved_ms']/max(c['cost_ms'],.001),c['loops']),default=None)


def prefix_budget(start,target,budget):
    result=min(budget,target-start)
    if result<=0:raise ValueError('Token prefix has no work before its committed checkpoint')
    return result


def matches_quantum(candidate,operation,tokens,depth,legal,total_depth):
    if candidate['operation']!=operation or candidate['tokens']!=tokens:return False
    loops=candidate['loops']
    if operation=='prelude':return loops==0
    return 1<=loops<=legal and candidate['terminal']==(depth+loops==total_depth)


def initial_token_prefix(prompt_tokens,candidates,armed):
    if not armed:return prompt_tokens
    prefix=min(candidates)
    if not 0<prefix<prompt_tokens:raise ValueError('A negotiated prefix must precede the actual prompt end')
    return prefix


def should_capture_state(cfg,seed,measurement_started):
    if not cfg.get('state_probe_measured_only'):return True
    return measurement_started and cfg['seed']<=seed<cfg['seed']+cfg['count']


def token_reservation_path(prefix,prompt_tokens):
    if prefix==prompt_tokens and prompt_tokens>0:return 'native'
    if not 0<prefix<prompt_tokens:raise ValueError('Invalid exact frontier from the coordinator')
    return 'partial'


def distinct_P_pressure(ready,pending,credit_clear_waiters=0):
    if credit_clear_waiters<0:raise ValueError('A waiting-request count cannot be negative')
    return len(set(ready)|set(pending))+credit_clear_waiters


def take_ready_batch(owner,phases):
    for phase in phases:
        if owner.has_ready(phase):
            batch=owner._take(phase)
            if batch is not None:return batch
    return None


def forecast_quantum(active_count,target,reserve,guard,maximum,empty_cap=None):
    if active_count==0 and empty_cap is not None:return min(maximum,max(0.,empty_cap))
    return min(maximum,max(0.,target-reserve-guard))


def offer_completed_boundary(depth,target,checkpoints):
    if depth<target:return False
    if depth!=target or depth not in checkpoints:raise ValueError('Unexpected source checkpoint')
    return True
