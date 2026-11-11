"""Run bounded collector self-tests on one exclusively owned GPU."""
import fcntl,json,signal,time
from .common import *


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources();records=[]
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        owner.stop(list(range(8)))
        try:
            assert not owner.processes(),'Do not displace an unknown GPU owner'
            for mode in ('baseline','early','late'):
                command=['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m','poc_flashvep.prefill_heavy_decode_safe.collector_smoke',mode]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');path=ROOT/'logs'/('collector_smoke_'+mode+'.log')
                with path.open('w') as log:
                    process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        assert os.getsid(process.pid)==process.pid
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                        code=124
                records.append(dict(mode=mode,command=command,returncode=code,GPU=0,time=time.time(),
                    source_SHA256=digest(SOURCE/'collector_smoke.py')))
                write(ROOT/'state/collector_smoke_progress.json',dict(results=records,closed=False))
                print('COLLECTOR_SELFTEST',mode,code,flush=True)
                if code:break
        finally:
            assert not owner.processes(),'Inspect a remaining worker before restoration'
            owner.restore(list(range(8)))
            write(ROOT/'state/collector_smoke_progress.json',dict(results=records,closed=True))


if __name__=='__main__':main()
