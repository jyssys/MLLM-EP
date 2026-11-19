"""Try one no-queue observer control, then stop profiling regardless of outcome."""
import fcntl
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,write,bind_legacy,preserve,resources
from .research_campaign import performance,freeze
from .run import execute


def main():
    bind_legacy();preserve()
    while not (ROOT/'state/diagnostic_complete.json').exists():
        assert not (ROOT/'state/diagnostic_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(10)
    from .online_audit import audit
    controls=[]
    try:
        for axis in ('Layer','Token'):
            configs=[]
            for mode in ('before','observed','after'):
                cfg=performance(axis,4,16,2.,2026111906,count=36,
                    stage='event_observer_serial_'+mode)
                cfg.update(name=cfg['name'].replace('_q2_','_SERIAL_'),
                    calibration_serial=True,calibration_serial_group_size=1,
                    arrival_offsets=[0.]*36,arrival_rate=None,
                    execution_events=mode=='observed',
                    arrival_scope='CLOSED_LOOP_SERIAL_MECHANISM_CONTROL_NOT_POISSON')
                configs.append(cfg)
            freeze('serial_observer_'+axis,configs,
                'After all four online event-observer gates failed, test one fixed-cut pair without arrival queue amplification. One request completes before the next arrives. No GPU sleep, kernel, SLO or partition changes. Never use this as an online QPS baseline. No further profiler refinement is authorized by this driver.')
            values=[]
            for cfg in configs:
                assert execute(cfg['name']),cfg['name']
                values.append(audit(cfg['name']))
            tests={}
            for key in ('TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s'):
                before,observed,after=[r['metric_summary'][key] for r in values]
                tests[key]=dict(before=before,observed=observed,after=after,
                    observed_over_controls=observed/((before+after)/2),
                    after_over_before=after/before,
                    passed=abs(observed/((before+after)/2)-1)<=.05 and abs(after/before-1)<=.05)
            controls.append(dict(method=axis,layers=4,names=[c['name'] for c in configs],tests=tests,
                passed=all(v['passed'] for v in tests.values()),
                evidence='DIAGNOSTIC',
                scope='Serial actual P→D execution. If the observer passes, only CUDA stream-event enclosures qualify. They remain distinct from exclusive kernels, Poisson Serving performance and sustainable capacity.'))
            write(ROOT/'analysis/serial_event_observer_gates.json',controls)
        write(ROOT/'state/serial_observer_complete.json',dict(time=time.time(),controls=controls,
            scope='One bounded alternative completed. Do not expand profiling if these gates fail.'))
    except BaseException as error:
        write(ROOT/'state/serial_observer_stopped.json',dict(time=time.time(),error=repr(error)))
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
