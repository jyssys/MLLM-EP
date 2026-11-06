"""Close all experiment launches before restoring verified approved burners."""
import fcntl
import signal
import subprocess
import time
from .common import *

CONTROLLERS={'mixed_qualification','finish_controls','quantum_calibration',
    'verify_collector_build',
    'phase_a','slack_confirmation','final_flush_confirmation','drained_flush_trials',
    'submission_event_trials','host_observer_controls','native_repeat_controls'}


def words(pid):
    try:return Path('/proc',str(pid),'cmdline').read_bytes().decode().split('\0')
    except (FileNotFoundError,ProcessLookupError):return []


def main():
    initialize();prefix='poc_flashvep.adaptive_idle_loop_continuation.'
    running=[]
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdecimal() or proc.stat().st_uid!=os.getuid():continue
            if set(words(proc.name)) & {prefix+s for s in CONTROLLERS}:running.append(proc.name)
        except (FileNotFoundError,ProcessLookupError):continue
    assert not running,('A controller remains active. Leave it running.',running)
    with (ROOT/'state/quiet_lane.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        active=json.loads((ROOT/'state/active.json').read_text())
        assert active.get('status')=='IDLE','A serving job still owns the registered lane'
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources();owned=owner.burners()
        for row in owner.processes():
            assert os.getsid(row['pid'])==owned.get(row['gpu']),'An unrelated GPU process remains untouched'
        write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),status='GPU_EXPERIMENTS_CLOSED',
            adaptive_policy='NOT_IMPLEMENTED_GATE_1_UNQUALIFIED',
            scope='No further GPU experiment launches. Analysis, report, and ordinary push remain.'))
        retired=[]
        for name,file in (('monitor','monitor_owner.json'),('deadline_guard','deadline_guard.json')):
            value=json.loads((ROOT/'state'/file).read_text());pid=value['pid']
            if prefix+name in words(pid):
                assert Path('/proc',str(pid)).stat().st_uid==os.getuid()
                os.kill(pid,signal.SIGTERM);retired.append(pid)
        samples=[];last_attempt=0.
        for _ in range(48):
            owned=owner.burners();processes=owner.processes()
            for row in processes:
                assert os.getsid(row['pid'])==owned.get(row['gpu']),'Do not interfere with an unrelated GPU process'
            missing=sorted(set(range(8))-set(owned))
            if missing and time.monotonic()-last_attempt>20:
                owner.restore(missing);last_attempt=time.monotonic()
            query=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                '--format=csv,noheader'],text=True)
            present={r['gpu'] for r in processes}
            nonzero=present==set(range(8)) and all(int(line.split(',')[1].strip().split()[0])>0
                for line in query.splitlines())
            verified=set(owned)==present==set(range(8))
            samples.append(dict(time=time.time(),owned_burners=owned,GPU_processes=processes,GPU_query=query))
            write(ROOT/'state/final_resource_restoration.json',dict(time=time.time(),verified=verified,
                all_GPU_utilization_nonzero=nonzero,samples=samples,retired_guards=retired,
                script='/home/esjung/vllm-ep/run_utilize.sh',explicit_GPU_selection='0,1,2,3,4,5,6,7',
                scope='Actual approved GPU sessions, not only worker-launch acknowledgments. Missing engines retry without stopping successful burners.'))
            if verified and nonzero:
                print(json.dumps(dict(restored=True,GPUs=list(range(8)))));return
            time.sleep(5)
        raise RuntimeError('All eight burner engines did not become active. Retain logs and retry only missing engines.')


if __name__=='__main__':main()
