"""Close the completed GPU campaign and restore only approved burner sessions."""
import subprocess
import time
import signal
from .common import *


def main():
    initialize()
    state=json.loads((ROOT/'state/recovery_controller.json').read_text())
    assert state['stage']=='COMPLETE','The serial controller still has pending work'
    active=json.loads((ROOT/'state/active.json').read_text())
    assert active.get('status')=='IDLE','A registered worker remains active'
    observers=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if '-m' not in words:continue
            module=words[words.index('-m')+1]
            if not module.startswith('poc_flashvep.d_local_cross_depth_batching.'):continue
            role=module.rsplit('.',1)[1]
            assert role in ('close_gpu','deadline_guard','watch_campaign'),('A campaign process remains',proc.name,module)
            if role!='close_gpu':observers.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError,UnicodeError):continue
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources()
    assert not owner.processes(),'A GPU process remains. Leave its ownership unchanged.'
    stamp=time.time()
    write(ROOT/'state/experiments_finished.json',dict(time=stamp,start=START,
        elapsed_hours=(stamp-START)/3600,GPU_deadline=GPU_DEADLINE,
        scope='All registered serial jobs returned. No new GPU experiment follows this record.'))
    write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=stamp,reason='REGISTERED_PLAN_COMPLETE'))
    for pid in observers:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    write(ROOT/'state/observers_closed.json',dict(time=time.time(),PIDs=observers,
        scope='Owned namespace watchdog and budget observer only. The serial GPU plan already returned.'))
    owner.restore(list(range(8)))
    samples=[]
    for _ in range(18):
        owned=owner.burners();processes=owner.processes()
        for row in processes:
            assert os.getsid(row['pid'])==owned.get(row['gpu']),'An unrelated process appeared. Do not signal it.'
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
            '--format=csv,noheader'],text=True)
        samples.append(dict(time=time.time(),owned_burners=owned,GPU_processes=processes,GPU_query=query))
        all_executing=set(owned)==set(range(8)) and {r['gpu'] for r in processes}==set(range(8))
        nonzero=all(int(line.split(',')[1].strip().split()[0])>0 for line in query.splitlines())
        write(ROOT/'state/final_resource_restoration.json',dict(time=time.time(),verified=all_executing,
            all_GPU_utilization_nonzero=all_executing and nonzero,samples=samples,
            script='/home/esjung/vllm-ep/run_utilize.sh',explicit_GPU_selection='0,1,2,3,4,5,6,7',
            scope='Only approved burner workers and their GPU process sessions. Burners remain running after publication.'))
        if all_executing and nonzero:
            initialize()
            print(json.dumps(dict(restored=True,all_GPU_utilization_nonzero=True,GPUs=list(range(8)))))
            return
        time.sleep(5)
    assert all_executing,'Not all eight approved burner GPU sessions became visible'
    print(json.dumps(dict(restored=True,all_GPU_utilization_nonzero=False,
        scope='All eight GPU sessions exist. Nonzero utilization was not observed on every device in one sample.')))


if __name__=='__main__':main()
