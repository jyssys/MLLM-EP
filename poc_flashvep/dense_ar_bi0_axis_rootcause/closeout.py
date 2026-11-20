"""Close only this campaign and verify all eight approved burners."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
from .common import ROOT,QUIET_LOCK,resources,preserve,write,digest


def experiment_processes():
    selected=[]
    modules={b'poc_flashvep.dense_ar_bi0_axis_rootcause.'+m for m in
        (b'serving',b'causal_serving',b'standard_worker',b'causal_worker')}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            command=(entry/'cmdline').read_bytes().split(b'\0')
            if any(value in modules for value in command):selected.append(dict(pid=int(entry.name),command=[v.decode(errors='replace') for v in command if v]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return selected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--close',action='store_true');args=parser.parse_args()
    preserve()
    if not args.close:
        print(json.dumps(dict(active_experiment_processes=experiment_processes(),
            lease=json.loads((ROOT/'state/GPU_lease_current.json').read_text()))));return
    core=('last12_confirmation_complete.json','credit_sweep_complete.json','pilot50_all_cuts_complete.json',
        'full200_allcuts_confirmation_complete.json','input_bucket_controls_complete.json',
        'continuous_bridge_complete.json','C3_bridge_followup_complete.json',
        'KV_block_sensitivity_complete.json','historical_runtime_bridge_complete.json',
        'historical_one_factor_controls_complete.json','historical_block16_confirmation_complete.json',
        'historical_block16_extra_holdout_complete.json','historical_coalescing_intervention_complete.json',
        'coalescing_paired_repetition_complete.json')
    missing=[name for name in core if not (ROOT/'state'/name).exists()]
    assert not missing,dict(incomplete_controllers=missing)
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not experiment_processes(),'Never close or archive an active experiment'
        owner=resources();active=owner.processes();burners=owner.burners()
        assert all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active),'Leave unknown GPU owners untouched'
        write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),status='GPU_WORK_CLOSED',
            core_flags_checked=list(core),new_GPU_dispatch_allowed=False,
            scope='A completed controller can contain NOT_RUN or failed cells. Closure is not a scientific GO. Raw evidence remains local.'))
        write(ROOT/'state/progress.json',dict(time=time.time(),status='GPU_WORK_CLOSED',
            phase='CPU_VERIFICATION_AND_PUBLICATION',new_GPU_dispatch_allowed=False))
        if set(burners)!=set(range(8)) or {r['gpu'] for r in active}!=set(range(8)):owner.restore(list(range(8)))
        samples=[]
        for _ in range(20):
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
            rows=[dict(GPU=int(a),UUID=b,utilization_percent=int(c),memory_MiB=int(d))
                for a,b,c,d in (line.split(', ') for line in output.strip().splitlines())]
            samples.append(dict(time=time.time(),GPUs=rows))
            active=owner.processes();burners=owner.burners()
            owned=all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active)
            present={r['gpu'] for r in active}==set(range(8))
            if owned and present and all(r['utilization_percent']>=70 for r in rows):break
            time.sleep(2)
        assert owned and present,'Approved burner ownership must cover all eight GPUs'
        assert all(r['utilization_percent']>=70 for r in rows),'Burner utilization did not stabilize'
        script=Path('/home/esjung/vllm-ep/run_utilize.sh')
        log=ROOT/'burner.log'
        if log.exists():
            # Operational snapshot only. The live log remains in place.
            snapshot=ROOT/'state/burner_closeout_log_snapshot.txt';shutil.copyfile(log,snapshot)
        preserve()
        write(ROOT/'state/final_GPU_resource_restoration.json',dict(time=time.time(),passed=True,
            approved_script=str(script),script_SHA256=digest(script),active_processes=active,
            burner_sessions=burners,utilization_samples=samples,no_experiment_processes=True,
            user_checkout_and_index_preserved=True,
            scope='All eight approved burners are running. Do not stop them after publication. No unowned process was terminated.'))
    print('CLOSEOUT_PASS_ALL8_BURNERS')


if __name__=='__main__':main()
