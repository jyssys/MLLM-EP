"""Defer prefill compute through the installed scheduler's legal admission path."""
import atexit
import hashlib
import inspect
import time
from .common import ROOT,write


def call(original,scheduler,*a,**kw):
    assert len(a)<=1
    previous=scheduler.prefill_capacity_bound
    scheduler.prefill_capacity_bound=False
    try:return original(scheduler,**dict(kw,throttle_prefills=True))
    finally:scheduler.prefill_capacity_bound=previous


def install(cfg,role):
    if not cfg.get('d_compute_throttle'):return
    assert role in ('prefill','decode')
    if role!='decode':return
    assert not cfg.get('d_quiescent_admission'),'Keep admission interventions separate'
    from vllm.v1.core.sched.scheduler import Scheduler
    original=Scheduler.schedule;source=inspect.getsource(original)
    assert 'throttle_prefills' in inspect.signature(original).parameters
    assert 'defer_prefills' in source and 'load_kv_async' in source
    rows=[];last=[None]
    def schedule(scheduler,*a,**kw):
        state=(sum(not r.is_prefill_chunk for r in scheduler.running),
            sum(r.is_prefill_chunk for r in scheduler.running),
            len(scheduler.waiting)+len(scheduler.skipped_waiting))
        if state!=last[0]:
            rows.append(dict(time=time.monotonic(),running_nonchunk=state[0],
                running_prefill_chunks=state[1],waiting=state[2]))
            last[0]=state
        return call(original,scheduler,*a,**kw)
    Scheduler.schedule=schedule
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'D_compute_throttle.json',dict(rows=rows,
        scheduler_source_SHA256=hashlib.sha256(source.encode()).hexdigest(),
        original_allocation_and_lifetime_checks_preserved=True,
        uses_installed_throttle_prefills=True,capacity_escape_disabled=True,
        remote_KV_loads_still_allowed=True,evidence='RUNTIME_POLICY_AUDIT',
        scope='Use the installed pre-allocation prefill-throttle branches. Never alter allocated token counts afterward. '
            'Decode and remote-KV processing remain continuous. Native last-prompt-query recomputation remains eligible. '
            'This is compute deferral,not device-wide idle admission. It does not guarantee an ITL bound. '
            'Real query records,correctness and client-visible latency must test the policy.')))


def test():
    from types import SimpleNamespace
    scheduler=SimpleNamespace(prefill_capacity_bound=True)
    def original(value,throttle_prefills=False):
        assert throttle_prefills and not value.prefill_capacity_bound
        return 7
    assert call(original,scheduler)==7 and scheduler.prefill_capacity_bound is True
    def failed(value,throttle_prefills=False):raise RuntimeError('Expected policy-path fixture')
    try:call(failed,scheduler)
    except RuntimeError:pass
    else:raise AssertionError('The real call must propagate an exception')
    assert scheduler.prefill_capacity_bound is True
    print('THROTTLE_CALL_AND_EXCEPTION_RESTORATION_PASS',flush=True)


if __name__=='__main__':test()
