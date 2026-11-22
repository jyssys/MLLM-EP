"""Move an owned queue parent at a completed GPU-job boundary."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write


def main():
    pid=3764359;path=Path('/proc',str(pid),'cmdline')
    expected=b'poc_flashvep.dense_ar_longbench400_pd.queue\0motivation400_discovery_v1.json'
    assert path.stat().st_uid==os.getuid() and expected in path.read_bytes()
    lease=ROOT/'state/pair_lease_0_1.json';current=json.loads(lease.read_text())
    assert current['name'].startswith('Motive400_') and current['status']=='RUNNING'
    receipt=ROOT/'state/main_queue_migration_v1.json';assert not receipt.exists()
    write(receipt,dict(old_pid=pid,current_job=current['name'],status='WAITING_FOR_CURRENT_JOB',time=time.time()))
    os.kill(pid,signal.SIGSTOP);stopped=True
    try:
        result=ROOT/'raw'/current['name']/'dispatcher_result.json'
        began=time.time()
        while not result.exists() or json.loads(lease.read_text()).get('status')!='RELEASED':
            assert time.time()-began<2400,'Do not interrupt a live GPU job for queue migration'
            time.sleep(1)
        assert expected in path.read_bytes()
        # The stopped parent cannot dispatch another job. Its GPU child already ended.
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False
        log_path=ROOT/'logs/background_motivation_queue_v2.log'
        command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.queue','motivation400_discovery_v1.json']
        with log_path.open('w') as log:
            child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(receipt,dict(old_pid=pid,new_pid=child.pid,current_job=current['name'],status='MIGRATED',time=time.time(),
            scope='Only the owned queue parent changed. The active GPU job drained normally. Existing receipts prevent duplicate GPU execution. '
                'The frozen configuration,arrival traces,source capsule and user checkout remain unchanged.'))
    finally:
        if stopped and path.exists():
            assert expected in path.read_bytes();os.kill(pid,signal.SIGCONT)


if __name__=='__main__':main()
