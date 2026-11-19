"""Bound observer cost before using any online GPU event enclosure."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,write,bind_legacy,resources,preserve
from .research_campaign import performance,freeze
from .run import execute


def main():
    bind_legacy();preserve()
    while not (ROOT/'state/expanded_state_complete.json').exists():
        assert not (ROOT/'state/research_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(10)
    from .online_audit import audit
    try:
        widths=[8000]+[s for lo,hi in ((432,448),(880,896),(1776,1792),(2656,2672)) for s in (lo,hi,8000-lo,8000-hi)]
        cfg=performance('Native',0,16,2.,2026111904,count=36,stage='isolated_actual_serving_API')
        cfg.update(auxiliary_only=True,entry_module='poc_flashvep.dense_ar_axis_sweetspot.isolated_serving_api',
            isolated_widths=sorted(set(widths)),isolated_skip_attention=True,
            isolated_profile_widths=[432,448,880,896,1776,1792,2656,2672,8000],
            wall_timeout=900)
        freeze('isolated_actual_API',[cfg],'Actual serving Linear API, original BF16 weights. Operator-only CUDA traces with independent observer gates. No online service-time inference.')
        assert execute(cfg['name']),cfg['name']
        controls=[]
        for m in (4,8):
            for a in ('Layer','Token'):
                values=[]
                configs=[performance(a,m,16,2.,2026111905,count=36,stage='event_observer_'+mode)
                    for mode in ('before','observed','after')]
                configs[1]['execution_events']=True
                freeze(f'event_observer_{a}_{m}',configs,'A/B/A outer CUDA events. Both clean-control drift and observed perturbation must stay within5% for each latency/throughput metric. Events still include launch gaps and stream waits.')
                for c in configs:
                    if time.time()>DEADLINE-3600:
                        return
                    assert execute(c['name']),c['name']
                    values.append(audit(c['name']))
                tests={}
                for key in ('TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s'):
                    b,o,f=[v['metric_summary'][key] for v in values]
                    tests[key]=dict(before=b,observed=o,after=f,observed_over_controls=o/((b+f)/2),
                        after_over_before=f/b,passed=abs(o/((b+f)/2)-1)<=.05 and abs(f/b-1)<=.05)
                controls.append(dict(method=a,layers=m,names=[c['name'] for c in configs],tests=tests,
                    passed=all(t['passed'] for t in tests.values()),
                    evidence='DIAGNOSTIC',
                    scope='Same prompt/seed/actions. CUDA stream-event enclosure, not exclusive kernel time. The observed run is never an independent performance confirmation.'))
                write(ROOT/'analysis/online_event_observer_gates.json',controls)
        write(ROOT/'state/diagnostic_complete.json',dict(time=time.time(),online_event_gates=controls))
    except BaseException as error:
        write(ROOT/'state/diagnostic_stopped.json',dict(time=time.time(),error=repr(error)))
        raise
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            owner=resources()
            if not owner.processes():
                owner.restore(list(range(8)))
        preserve()


if __name__=='__main__':
    main()
