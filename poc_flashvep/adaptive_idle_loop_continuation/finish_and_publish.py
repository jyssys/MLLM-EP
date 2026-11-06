"""Finish a bounded evidence chain. Never leave a monitor as the research owner."""
import signal
import subprocess
import time
from .common import *

WAIT_FOR={'mixed_qualification','finish_controls','verify_collector_build'}


def active_controllers():
    expected={'poc_flashvep.adaptive_idle_loop_continuation.'+s for s in WAIT_FOR}
    result=[]
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdecimal() or proc.stat().st_uid!=os.getuid():continue
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if expected&set(words):result.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError,UnicodeError):continue
    return result


def main():
    initialize();state=ROOT/'state/finish_and_publish.json';records=[]
    try:
        while active_controllers():
            assert time.time()<GPU_DEADLINE+30,'GPU controller did not retire at the fixed deadline'
            write(state,dict(status='WAITING_FOR_REGISTERED_CONTROLLERS',time=time.time(),PIDs=active_controllers(),records=records))
            time.sleep(5)
        from ..looped_pd_autonomous_headroom.common import PYTHON
        tasks=[('close_gpu',()),('cpu_regression',()),('analyze',()),('primitive_analysis',()),
            ('decode_service_diagnosis',()),('opportunity_oracle',()),('inventory',()),
            ('figures',()),('report',()),('publish',('archive',)),('publish',('publish','--push'))]
        with (ROOT/'state/finish_and_publish.log').open('a') as log:
            for module,args in tasks:
                assert time.time()<DEADLINE-30,'No time remains for another closeout step'
                command=[str(PYTHON),'-m','poc_flashvep.adaptive_idle_loop_continuation.'+module,*args]
                write(state,dict(status='RUNNING',time=time.time(),step=module,args=args,records=records))
                begin=time.time()
                child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=max(1,DEADLINE-time.time()-15))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10);raise
                records.append(dict(step=module,args=args,returncode=code,begin=begin,end=time.time()))
                write(state,dict(status='RUNNING',time=time.time(),step=module,records=records))
                assert code==0,(module,args,code)
        write(state,dict(status='COMPLETE',time=time.time(),records=records))
    except BaseException as error:
        write(state,dict(status='FAILED',time=time.time(),error=repr(error),records=records))
        raise


if __name__=='__main__':main()
