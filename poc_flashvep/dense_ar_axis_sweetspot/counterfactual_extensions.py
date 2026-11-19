"""Use negative controls and an Output-only intervention to test the cause."""
import fcntl
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,bind_legacy,preserve,write,resources
from .research_campaign import performance,freeze,run


def wait_for(marker):
    while not (ROOT/'state'/marker).exists():
        for failure in ('research_stopped.json','batch_extension_stopped.json'):
            assert not (ROOT/'state'/failure).exists(),failure
        assert time.time()<DEADLINE-3600
        time.sleep(10)


def main():
    bind_legacy();preserve()
    window=[performance(a,m,16,6.,2026111901,batch=2,window=w,count=72,
        stage='source_window_control') for m in (4,12)
        for w,axes in ((32,('Layer','Token')),(4,('Token','Layer'))) for a in axes]
    freeze('source_window_negative_positive_controls',window,
        'Last4 is the P-bound negative control. Last12 is the strongly D-bound positive control. The guard/queue-location explanation predicts much larger wait relocation atLast12. Source window is the only changed limit within each axis/cut pair. Same fresh seed1901 asLast8.')
    output=[]
    for rep,seed in enumerate((2026111940,2026111941,2026111942)):
        for cap in ((16,64) if rep%2==0 else (64,16)):
            arms=[('Native',0),('Layer',4),('Token',4)]
            arms=arms[rep:]+arms[:rep]
            output.extend(performance(a,m,cap,6.,seed,batch=2,count=72,
                stage='output_residency_counterfactual') for a,m in arms)
    freeze('output_residency_counterfactual',output,
        'Keep Last4/Token880-896, actual prep1/Decode2, B/S, Input and arrival trace fixed. Change only controlled Output16 to64. Isolated prefill GEMM predicts no ranking change. Stage-balance hypothesis predicts that the D Decode residency can moveLast4 from P-bound to D-bound. Three paired fresh starts per setting, without a post-hoc SLO change.')
    qps=[performance(a,m,16,3.,seed,batch=2,count=72,
        stage='new_QPS_confirmation') for seed in (2026111940,2026111941,2026111942)
        for a,m in (('Native',0),('Layer',4),('Token',4))]
    freeze('new_QPS_confirmation',qps,
        'QPS3 was absent from historical Decode2 discovery. Validate the frozen pressure rule at this new offered rate. Same arrival seeds and prompts. Do not alter the rule after this test.')
    try:
        wait_for('source_window_complete.json')
        if not run('source_window_extended',window):
            return
        wait_for('batch_extension_complete.json')
        if not run('output_residency_counterfactual',output):
            return
        if time.time()<DEADLINE-2*3600:
            run('new_QPS_confirmation',qps)
        else:
            write(ROOT/'state/new_QPS_confirmation_deferred.json',dict(time=time.time(),status='NOT_RUN',
                reason='Keep the one-hour publication reserve and prioritize existing independent confirmation.'))
        write(ROOT/'state/counterfactual_extensions_complete.json',dict(time=time.time()))
    except BaseException as error:
        write(ROOT/'state/counterfactual_extensions_stopped.json',dict(time=time.time(),error=repr(error)))
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
