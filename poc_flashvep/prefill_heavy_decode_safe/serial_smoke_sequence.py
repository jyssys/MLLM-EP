"""Test the documented lower-instrumentation mode before Native use."""
import fcntl,json,signal,time
from .common import *


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources();rows=[];namespace='collector_serial_api26_v1'
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);owner.stop(list(range(8)))
        try:
            assert not owner.processes()
            for mode in ('baseline','early','serial'):
                command=['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
                    'poc_flashvep.prefill_heavy_decode_safe.collector_smoke',mode,'--namespace',namespace]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',LD_PRELOAD='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1')
                with (ROOT/'logs'/(namespace+'_'+mode+'.log')).open('w') as log:
                    process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        assert os.getsid(process.pid)==process.pid
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20);code=124
                rows.append(dict(mode=mode,returncode=code,time=time.time(),GPU=0,command=command,
                    scope='Synthetic one-stream graph self-test. Serial mode changes concurrency. This is not Native overhead qualification.'))
                write(ROOT/'state/serial_smoke_progress.json',dict(results=rows,closed=False))
                print('SERIAL_COLLECTOR_SELFTEST',mode,code,flush=True)
                if code:break
        finally:
            assert not owner.processes();owner.restore(list(range(8)))
            write(ROOT/'state/serial_smoke_progress.json',dict(results=rows,closed=True))


if __name__=='__main__':main()
