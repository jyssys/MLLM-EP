"""Close dispatch, wait for owned work, and restore approved burners."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .common import ROOT,REPO,QUIET_LOCK,digest,preserve,resources,write

CONTROLLERS={'baseline_v3','async_qualification','continuation_costs','controlled_discovery',
    'confirmation','replay_campaign','fast_control_profile','context_boundary_qualification',
    'output_replay_control','static_qps_extension','matched_reference','minimal_rule','coalescing_screen','coalescing_followup','cut12_confirmation','low_qps_followup'}


def live_controllers():
    result=[]
    names={'poc_flashvep.dense_ar_bi0_shortanswer_pd.'+name for name in CONTROLLERS}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid==os.getuid() and names.intersection(words):
                result.append(dict(pid=int(proc.name),module=next(iter(names.intersection(words)))))
        except (OSError,UnicodeError):pass
    return result


def main():
    preserve()
    write(ROOT/'state/STOP_REQUESTED.json',dict(time=time.time(),reason='Campaign closeout. Let the current owned job finish. Pending cells remain NOT_RUN.'))
    # Release the barrier before waiting. Queued dispatchers must acquire it
    # once to observe STOP and exit without starting another GPU worker.
    with QUIET_LOCK.open('a+') as barrier:
        fcntl.flock(barrier,fcntl.LOCK_EX)
        current=json.loads((ROOT/'state/GPU_lease_current.json').read_text())
        assert current.get('status')!='RUNNING','Let the current owned GPU job finish'
    until=time.monotonic()+120
    while live_controllers() and time.monotonic()<until:time.sleep(2)
    assert not live_controllers(),live_controllers()
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        current=json.loads((ROOT/'state/GPU_lease_current.json').read_text())
        assert current.get('status')!='RUNNING','Do not freeze evidence while a GPU job is active'
        assert not live_controllers(),live_controllers()
        owner=resources();owned=owner.burners();active=owner.processes()
        assert all(os.getsid(r['pid'])==owned.get(r['gpu']) for r in active),'Unrelated GPU work remains untouched'
        owner.restore(list(range(8)))
        until=time.monotonic()+40
        while time.monotonic()<until:
            owned=owner.burners();active=owner.processes()
            if len(active)==8 and {r['gpu'] for r in active}==set(range(8)):break
            time.sleep(1)
        assert set(owned)==set(range(8)) and len(active)==8
        assert all(os.getsid(r['pid'])==owned[r['gpu']] and (Path('/proc')/str(r['pid'])).stat().st_uid==os.getuid() for r in active)
        utilization=[]
        for _ in range(20):
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
            utilization=[dict(index=int(parts[0]),UUID=parts[1].strip(),GPU_utilization_percent=int(parts[2]),memory_MiB=int(parts[3]))
                for parts in (line.split(',') for line in raw.splitlines())]
            if len(utilization)==8 and all(r['GPU_utilization_percent']>0 for r in utilization):break
            time.sleep(1)
        previous=json.loads((ROOT/'state/prior_report_hashes.json').read_text())
        assert all(digest(REPO/'poc_flashvep/reports'/name)==sha for name,sha in previous.items())
        from .runtime_audit import main as runtime_audit
        runtime_audit();preserve()
        log=ROOT/'burner.log'
        if log.exists():shutil.copyfile(log,ROOT/'state/burner_closeout_snapshot.log')
        write(ROOT/'state/burner_final_state.json',dict(time=time.time(),owned_sessions=owned,active_GPU_processes=active,
            GPU_inventory=utilization,run_utilize_SHA256=digest(Path('/home/esjung/vllm-ep/run_utilize.sh')),
            all8_running=True,user_UID=os.getuid(),scope='Only approved burner sessions. No experiment worker remains on a GPU.'))
        write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),passed=True,
            remaining_controllers=[],prior_reports_unchanged=True,user_checkout_index_external_runtime_preserved=True,
            burners_restored=True,scope='No later GPU dispatch. Raw evidence is ready for final CPU analysis and publication.'))
    print('CAMPAIGN_CLOSED_BURNERS_RESTORED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--close',action='store_true',required=True)
    parser.parse_args();main()
