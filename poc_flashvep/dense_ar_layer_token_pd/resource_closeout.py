"""Restore approved burners and verify positive utilization on all eight GPUs."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .common import ROOT,QUIET_LOCK,resources,preserve,write,digest


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    target=ROOT/'state/final_GPU_resource_restoration.json'
    assert not target.exists(),'Preserve the original closeout receipt'
    preserve()
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        owner=resources();active=owner.processes();approved=owner.burners()
        assert all(os.getsid(row['pid'])==approved.get(row['gpu']) for row in active),active
        owner.restore(list(range(8)))
        samples=[];consecutive=0;started=time.time()
        while time.time()-started<120:
            result=subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
                '--format=csv,noheader,nounits'],text=True)
            values=[]
            for line in result.splitlines():
                index,uuid,utilization,memory,power,temp,clock=[s.strip() for s in line.split(',')]
                values.append(dict(gpu=int(index),UUID=uuid,utilization_pct=int(utilization),
                    memory_MiB=int(memory),power_W=float(power),temperature_C=int(temp),SM_clock_MHz=int(clock)))
            approved=owner.burners();active=owner.processes()
            ownership=(set(approved)==set(range(8)) and
                {r['gpu'] for r in active}==set(range(8)) and
                all(os.getsid(r['pid'])==approved.get(r['gpu']) for r in active))
            positive=all(r['utilization_pct']>0 for r in values) and len(values)==8
            samples.append(dict(time=time.time(),GPUs=values,approved_sessions=approved,
                active=active,ownership_passed=ownership,all_utilization_positive=positive))
            consecutive=consecutive+1 if ownership and positive else 0
            if consecutive>=3:
                break
            time.sleep(2)
        assert consecutive>=3,'All-eight positive utilization/ownership did not stabilize'
        snapshot=ROOT/'state/final_burner_log_snapshot.txt'
        assert not snapshot.exists()
        shutil.copyfile(ROOT/'burner.log',snapshot)
        write(target,dict(time=time.time(),passed=True,samples=samples,
            command='bash /home/esjung/vllm-ep/run_utilize.sh',
            explicit_GPU_environment='VLLM_UTILIZE_GPUS=0,1,2,3,4,5,6,7',
            burner_log_snapshot_SHA256=digest(snapshot),consecutive_positive_samples=consecutive,
            scope='Only the approved burner facade restores unoccupied devices. Three consecutive samples verify real positive NVML utilization and verified owned worker sessions on GPUs0-7. The burner remains running after this check. No unrelated process is signaled.'))
    preserve();print('ALL_EIGHT_BURNERS_RESTORED_AND_ACTIVE',flush=True)


if __name__=='__main__':
    main()
