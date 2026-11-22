"""Limit new partial admission before allocation,not after model scheduling."""
import atexit
import time
from .common import ROOT,write


def capped_call(original,scheduler,cap,*args,**kwargs):
    prior=scheduler.max_num_running_reqs
    scheduler.max_num_running_reqs=min(prior,cap)
    try:
        return original(scheduler,*args,**kwargs)
    finally:
        scheduler.max_num_running_reqs=prior


def install(cfg,role,info,adapter):
    if not cfg.get('D_prefill_completion_guard') or role!='decode':
        return
    assert cfg.get('d_compute_throttle') and cfg.get('defer_D_remote_setup')
    from vllm.v1.core.sched.scheduler import Scheduler
    original=Scheduler.schedule
    rows=[]

    def left(request):
        computed=request.num_computed_tokens
        length=request.num_prompt_tokens
        if adapter is None:
            remote=bool((request.kv_transfer_params or {}).get('do_remote_prefill'))
            return max(0,length-1-computed)+1 if not remote else 1
        part=adapter.partition(request.request_id)
        if not part.upper:
            return max(0,part.length-max(computed,part.prefix)) if part.suffix else 1
        return max(0,part.length-computed)

    def can_complete(request):
        if adapter is None:
            return True
        part=adapter.partition(request.request_id)
        aligned=cfg.get('partition_aligned_chunks') and part.upper and part.suffix
        fused=cfg.get('fuse_short_hybrid') and part.length<=8192
        return not aligned or fused or request.num_computed_tokens>=part.prefix

    def schedule(scheduler,*args,**kwargs):
        running=list(scheduler.running)
        # Preserve the independent decode-first policy. Never increase admission.
        if not running or any(not r.is_prefill_chunk for r in running):
            return original(scheduler,*args,**kwargs)
        remaining=sum(left(r) for r in running)
        budget=scheduler.max_num_scheduled_tokens
        if remaining>=budget:
            return original(scheduler,*args,**kwargs)
        available=budget-remaining
        fit=0
        for request in list(scheduler.waiting)+list(scheduler.skipped_waiting):
            if request.status.name not in ('WAITING','PREEMPTED'):
                continue
            cost=left(request)
            if not can_complete(request) or cost>available:
                break
            available-=cost
            fit+=1
        cap=len(running)+fit
        if cap>=scheduler.max_num_running_reqs:
            return original(scheduler,*args,**kwargs)
        rows.append(dict(time=time.monotonic(),running_prefills=len(running),
            running_remaining_query_estimate=remaining,B=budget,new_complete_requests_estimate=fit,
            original_running_cap=scheduler.max_num_running_reqs,temporary_running_cap=cap))
        return capped_call(original,scheduler,cap,*args,**kwargs)

    Scheduler.schedule=schedule
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'D_prefill_completion_guard.json',dict(
        rows=rows,evidence='PRE_ALLOCATION_POLICY_AUDIT',
        scope='When partial prefills already run,limit new admissions to estimated whole-prefill completions in remaining B8192. '
            'Keep small fitting Token/Native candidates eligible. The rule applies to all axes. '
            'Remote-not-ready requests can make the estimate conservative or incomplete. '
            'It does not guarantee zero fragmentation. Measure actual query counts and latency. '
            'Do not change token budgets,allocated tokens,KV ownership,model outputs or client timestamps. '
            'The original running cap is restored even on exceptions.')))


def test():
    from types import SimpleNamespace
    s=SimpleNamespace(max_num_running_reqs=1024)
    def ok(value):
        assert value.max_num_running_reqs==2
        return 7
    assert capped_call(ok,s,2)==7 and s.max_num_running_reqs==1024
    def failed(value):
        raise RuntimeError('Expected fixture')
    try:
        capped_call(failed,s,2)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Propagate the original scheduler failure')
    assert s.max_num_running_reqs==1024
    print('PREFILL_COMPLETION_CAP_RESTORATION_PASS',flush=True)


if __name__=='__main__':
    test()
