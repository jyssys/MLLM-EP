"""Retry only a missing approved burner after an isolated startup failure."""
import os
import subprocess
import time
from .common import *


def main():
    initialize()
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources()
    initial=ROOT/'state/final_resource_restoration.json'
    if initial.exists() and not (ROOT/'state/initial_resource_restoration_attempt.json').exists():
        write(ROOT/'state/initial_resource_restoration_attempt.json',json.loads(initial.read_text()))
    before=owner.processes();owned=owner.burners()
    for row in before:
        assert os.getsid(row['pid'])==owned.get(row['gpu']),'Leave unrelated GPU processes untouched'
    missing=sorted(set(range(8))-set(owned))
    assert not any(r['gpu'] in missing for r in before),'A missing worker still owns a GPU process'
    write(ROOT/'state/burner_startup_retry.json',dict(time=time.time(),missing=missing,
        reason='GPU 1 initial vLLM TCPStore startup reported EADDRINUSE on port 39901.',
        existing_sessions_left_running=owned,
        scope='Post-experiment resource recovery only. No listener or existing burner is terminated.'))
    owner.restore(missing)
    samples=[]
    for _ in range(30):
        owned=owner.burners();active=owner.processes()
        for row in active:
            assert os.getsid(row['pid'])==owned.get(row['gpu']),'Leave unrelated GPU processes untouched'
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
            '--format=csv,noheader'],text=True)
        samples.append(dict(time=time.time(),owned_burners=owned,GPU_processes=active,GPU_query=query))
        verified=set(owned)==set(range(8)) and {v['gpu'] for v in active}==set(range(8))
        nonzero=verified and all(int(line.split(',')[1].strip().split()[0])>0 for line in query.splitlines())
        write(initial,dict(time=time.time(),verified=verified,all_GPU_utilization_nonzero=nonzero,
            samples=samples,script='/home/esjung/vllm-ep/run_utilize.sh',explicit_GPU_selection='0,1,2,3,4,5,6,7',
            initial_failed_attempt='state/initial_resource_restoration_attempt.json',
            scope='Verified owned burner sessions only. Existing sessions remain running.'))
        if nonzero:
            initialize();print(json.dumps(dict(restored=True,GPUs=list(range(8)),all_utilization_nonzero=True)));return
        time.sleep(5)
    assert verified,'Not all eight approved burner GPU sessions became visible'


if __name__=='__main__':main()
