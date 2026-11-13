"""Continue the baseline campaign after its exactness gate completes."""
import fcntl
from .common import *
from .prepare import base
from .run import execute


def main():
    initialize();deadline=time.monotonic()+1200
    while True:
        path=ROOT/'state/phase_A_initial_progress.json'
        initial=json.loads(path.read_text()) if path.exists() else {}
        assert not initial.get('failed_job') and not initial.get('stop'),initial
        if len(initial.get('results',[]))==8:break
        assert time.monotonic()<deadline,'Initial qualification did not complete'
        time.sleep(3)
    c=base('capacity_D_graph_all_caps_B416_v2',count=16,cap=200,dataset='calibration.json')
    c.update(module='capacity',capacity_mode='PREPARED_D',gpus=[1],reuse_prefill_token=False,
        serialize_D_imports=False,cohort_barrier=False,decode_only_graph_roles=['capacity'],wall_timeout=1500,
        jobs=[dict(batch=b,cap=cap,repeats=2) for cap in (16,32,64,128,200) for b in ((1,4,8,16) if cap in (16,200) else (4,16))],
        phase='MATCHED_DECODE_GRAPH_AND_TEXT_CAPACITY',
        scope='Actual Full Decode Graph policy and exact text priming. Prepared local capacity excludes all migration costs. The earlier v1 capacity used eager Decode and remains separate.')
    try:
        if not execute(save(c),restore=False):return
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))
    from .discovery import main as discover
    discover()
    from .calibrate_slo import main as calibrate
    calibrate()


if __name__=='__main__':main()
