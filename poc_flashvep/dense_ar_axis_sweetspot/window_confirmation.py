"""Confirm Last12 queue relocation after one-start service drift appeared."""
import fcntl
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,bind_legacy,preserve,write,resources
from .research_campaign import performance,freeze,run


def main():
    bind_legacy();preserve()
    configs=[]
    for seed,windows in ((2026111950,(4,32)),(2026111951,(32,4))):
        for index,w in enumerate(windows):
            axes=('Layer','Token') if index==0 else ('Token','Layer')
            configs.extend(performance(a,12,16,6.,seed,batch=2,window=w,count=72,
                stage='source_window_control') for a in axes)
    freeze('source_window_Last12_confirmation',configs,
        'The initial Last12 intervention relocated seconds of wait, but its service-time changes differed across axes. Two new paired arrival seeds reverse the treatment order. Preserve the initial seed1901. Do not infer equivalence from one start or discard service drift.')
    while not (ROOT/'state/source_window_extended_complete.json').exists():
        assert not (ROOT/'state/counterfactual_extensions_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(10)
    try:
        run('source_window_Last12_confirmation',configs)
    except BaseException as error:
        write(ROOT/'state/window_confirmation_stopped.json',dict(time=time.time(),error=repr(error)))
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
