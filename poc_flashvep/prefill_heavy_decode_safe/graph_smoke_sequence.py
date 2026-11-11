"""Check graph coverage and eager coverage before any bound claim."""
import fcntl,json,os,signal,time
from .common import *


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    initialize();predecessor=3083454
    while alive(predecessor):
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources();code=None
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);owner.stop(list(range(8)))
        try:
            assert not owner.processes()
            command=['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
                'poc_flashvep.prefill_heavy_decode_safe.collector_smoke','graph','--namespace','collector_graph_api26_v1']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',LD_PRELOAD='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1')
            with (ROOT/'logs/collector_graph_api26_v1.log').open('w') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    assert os.getsid(child.pid)==child.pid and Path('/proc',str(child.pid)).stat().st_uid==os.getuid()
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);code=124
        finally:
            assert not owner.processes();owner.restore(list(range(8)))
    write(ROOT/'state/graph_smoke_receipt.json',dict(returncode=code,time=time.time(),passed=code==0,
        scope='Synthetic graph and eager coverage only. Not real model performance, exact kernel busy, or online capacity.'))
    print('GRAPH_COLLECTOR_SELFTEST',code,flush=True)


if __name__=='__main__':main()
