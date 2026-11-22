"""Close owned work only,then restore and check all eight approved burners."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .common import ROOT,QUIET_LOCK,write,digest,preserve,resources


def processes():
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            if entry.stat().st_uid!=os.getuid():
                continue
            command=(entry/'cmdline').read_bytes().split(b'\0')
            modules=[c.decode() for c in command if c.startswith(b'poc_flashvep.dense_ar_longbench400_pd.')]
            if modules:
                found.append(dict(pid=int(entry.name),modules=modules))
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    return found


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--close',action='store_true')
    parser.add_argument('--resume-burner-verification',action='store_true')
    args=parser.parse_args()
    assert not args.resume_burner_verification or args.close
    preserve()
    monitors={'poc_flashvep.dense_ar_longbench400_pd.guardian','poc_flashvep.dense_ar_longbench400_pd.watch_analysis'}
    current=processes()
    active=[r for r in current if not set(r['modules'])<=monitors]
    if not args.close:
        print(json.dumps(dict(active_work=active,monitors=[r for r in current if r not in active])),flush=True)
        return
    assert not active,'Do not interrupt an unfinished controller or GPU job'
    accounting=json.loads((ROOT/'state/final_work_accounting.json').read_text())
    assert accounting['all_required_work_accounted'] and not accounting['unresolved_mandatory_work']
    # Do not trust a stale green receipt after any reviewed report changed.
    from .final_work_accounting import main as refresh_accounting
    refresh_accounting()
    accounting=json.loads((ROOT/'state/final_work_accounting.json').read_text())
    assert accounting['all_required_work_accounted'] and not accounting['unresolved_mandatory_work']
    assert json.loads((ROOT/'analysis/cpu_final_regression_v1.json').read_text())['passed']
    assert (ROOT/'analysis/final_continuous_strong_confirmation_v1.json').exists()
    for path in (ROOT/'state').glob('pair_lease_*.json'):
        assert json.loads(path.read_text()).get('status')!='RUNNING',path
    write(ROOT/'state/STOP_DISPATCH.json',dict(time=time.time(),reason='COMPLETED_OR_SCIENTIFICALLY_ACCOUNTED_CAMPAIGN'))
    write(ROOT/'state/STOP_ANALYSIS_WATCHER.json',dict(time=time.time(),reason='FREEZE_FINAL_EVIDENCE'))
    write(ROOT/'state/STOP_GUARDIAN.json',dict(time=time.time(),reason='ALL_OWNED_GPU_JOBS_DRAINED'))
    deadline=time.monotonic()+180
    while processes():
        remaining=processes()
        assert all(set(r['modules'])<=monitors for r in remaining),remaining
        assert time.monotonic()<deadline,'Keep the evidence. Inspect a monitor that did not stop normally.'
        time.sleep(1)
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        owner=resources()
        active_gpu=owner.processes()
        burners=owner.burners()
        assert all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active_gpu), 'Leave unknown GPU owners untouched'
        script=Path('/home/esjung/vllm-ep/run_utilize.sh')
        original=json.loads((ROOT/'state/environment_initial.json').read_text())
        assert digest(script)==original['approved_script_SHA256']
        # Restart only the owned approved sessions. One fresh all-eight launch
        # keeps the requested burners active during CPU publication and afterward.
        if args.resume_burner_verification:
            # The approved eight-model burner can need more than60s to load,
            # compile and capture graphs. Do not restart an already ready set.
            records=[json.loads(p.read_text()) for p in (ROOT/'state').glob('burner_*.json')]
            records=[r for r in records if isinstance(r,dict) and r.get('GPUs')==list(range(8)) and isinstance(r.get('supervisor'),int)]
            record=max(records,key=lambda r:r['time'])
            last_gpu=json.loads((ROOT/'state/eager_compile_confirmation_v2_complete_v1.json').read_text())['time']
            assert record['time']>last_gpu,'Resume only the fresh closeout launch'
            process=Path('/proc')/str(record['supervisor'])
            assert process.stat().st_uid==os.getuid()
            assert owner.SCRIPT in (process/'cmdline').read_bytes().decode().split('\0')
            assert set(owner.burners())==set(range(8))
        else:
            owner.stop(list(range(8)))
            owner.restore(list(range(8)))
        samples=[]
        for _ in range(150):
            data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'],text=True)
            values=[dict(GPU=int(a),UUID=b,utilization_percent=int(c),memory_MiB=int(d))
                for a,b,c,d in (line.split(', ') for line in data.strip().splitlines())]
            active_gpu=owner.processes()
            burners=owner.burners()
            owned=all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active_gpu)
            present={r['gpu'] for r in active_gpu}==set(range(8))
            samples.append(dict(time=time.time(),GPUs=values))
            if owned and present and all(r['utilization_percent']>=70 for r in values):
                break
            time.sleep(2)
        assert owned and present and all(r['utilization_percent']>=70 for r in values)
        if (ROOT/'burner.log').exists():
            shutil.copyfile(ROOT/'burner.log',ROOT/'state/burner_closeout_log_snapshot.txt')
        preserve()
        write(ROOT/'state/final_GPU_resource_restoration.json',dict(time=time.time(),passed=True,
            approved_script=str(script),script_SHA256=digest(script),GPUs=list(range(8)),
            active_processes=active_gpu,burner_sessions=burners,utilization_samples=samples,
            no_experiment_processes=True,user_checkout_and_index_preserved=True,
            resumed_existing_fresh_burner_verification=args.resume_burner_verification,
            scope='A fresh approved all-eight burner launch is running. Do not stop these sessions after publication. '
                'No unknown GPU process was terminated. Utilization is an operational closeout check,not a research GPU-busy metric.'))
        write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),status='GPU_WORK_CLOSED',
            scientific_success_not_implied=True,new_GPU_dispatch_allowed=False,
            scope='All required work is completed or has an explicit scientific stop reason. '
                'Failed and NOT_RUN cells remain in the evidence. CPU archive and publication still remain.'))
    print('LONG BENCH400 GPU CLOSEOUT PASS: ALL8 APPROVED BURNERS ACTIVE',flush=True)


if __name__=='__main__':
    main()
