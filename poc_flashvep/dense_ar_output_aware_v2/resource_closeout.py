"""Close drained v2 GPU work and verify the approved persistent all-eight burner."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,QUIET_LOCK,write,digest,preserve,resources


def owned_processes():
    result=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            words=(entry/'cmdline').read_bytes().decode().split('\0')
            if any(w.startswith('poc_flashvep.dense_ar_output_aware_v2.') for w in words):
                result.append(dict(pid=int(entry.name),words=words))
        except (OSError,UnicodeError):continue
    return result


def close():
    preserve()
    assert json.loads((ROOT/'analysis/cpu_final_regression_v2.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/raw_metric_recalculation_v2.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/final_configuration_and_gate_inventory_v2.json').read_text())['passed']
    assert (ROOT/'analysis/scientific_verdict_v2.json').exists()
    assert all(json.loads(p.read_text()).get('status')!='RUNNING' for p in (ROOT/'state').glob('pair_lease_*.json'))
    active=owned_processes()
    assert not any('poc_flashvep.dense_ar_output_aware_v2.runner' in r['words'] for r in active),'Wait for queued pair runners to finish'
    write(ROOT/'state/STOP_DISPATCH.json',dict(time=time.time(),reason='FINAL_SCIENTIFIC_AND_BUDGET_CLOSEOUT'))
    write(ROOT/'state/STOP_EXECUTION.json',dict(time=time.time(),reason='ALL_OWNED_GPU_JOBS_DRAINED'))
    write(ROOT/'state/STOP_GUARDIAN.json',dict(time=time.time(),reason='ALL_OWNED_PAIR_LEASES_RELEASED'))
    # These two processes only sample state. They cannot own a GPU actor.
    stopped=[]
    for r in active:
        if any(module in r['words'] for module in (
            'poc_flashvep.dense_ar_output_aware_v2.monitor','poc_flashvep.dense_ar_output_aware_v2.environment_sampler')):
            proc=Path('/proc')/str(r['pid'])
            assert not (proc/'task'/str(r['pid'])/'children').read_text().strip()
            os.kill(r['pid'],signal.SIGTERM);stopped.append(r)
    until=time.monotonic()+60
    while json.loads((ROOT/'state/guardian.json').read_text())['status']!='RELEASED':
        assert time.monotonic()<until,'Inspect the guardian without terminating other work'
        time.sleep(1)
    with QUIET_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        owner=resources();running=owner.processes();approved=owner.burners()
        assert all(os.getsid(r['pid'])==approved.get(r['gpu']) for r in running),'Leave unrelated GPU owners untouched'
        environment=json.loads((ROOT/'state/environment_initial.json').read_text())
        assert digest(Path(owner.SCRIPT))==environment['burner_script_SHA256']
        owner.stop(list(range(8)));owner.restore(list(range(8)))
        samples=[];deadline=time.monotonic()+300
        while True:
            value=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'],text=True)
            devices=[dict(GPU=int(a),UUID=b.strip(),utilization_percent=int(c),memory_MiB=int(d))
                for a,b,c,d in (line.split(',') for line in value.strip().splitlines())]
            running=owner.processes();approved=owner.burners()
            owned=all(Path('/proc',str(r['pid'])).stat().st_uid==os.getuid() and
                os.getsid(r['pid'])==approved.get(r['gpu']) for r in running)
            complete={r['gpu'] for r in running}==set(range(8))==set(approved)
            samples.append(dict(time=time.time(),devices=devices))
            if owned and complete and all(r['utilization_percent']>=70 for r in devices):break
            assert time.monotonic()<deadline,'Preserve the all-eight launch and inspect a slow burner startup'
            time.sleep(2)
        preserve()
        write(ROOT/'state/resource_closeout_v2.json',dict(time=time.time(),passed=True,GPUs=list(range(8)),
            active_processes=running,burner_sessions=approved,samples=samples,
            burner_script=owner.SCRIPT,burner_script_SHA256=digest(Path(owner.SCRIPT)),
            bash_wrapper='/home/esjung/vllm-ep/run_utilize.sh',
            bash_wrapper_SHA256=digest(Path('/home/esjung/vllm-ep/run_utilize.sh')),
            stopped_readonly_monitors=stopped,user_checkout_and_index_preserved=True,
            unknown_GPU_processes_terminated=False,
            scope='All owned GPU jobs drained before closeout. The approved bash wrapper launched persistent burners on all8 GPUs. '
                'Ownership and utilization were checked. This operational utilization is not a research kernel-busy measurement. '
                'Do not stop these burner sessions after publication. Scientific failures and NOT_RUN points remain in the evidence.'))
    print('ALL8_APPROVED_BURNERS_RESTORED_AND_VERIFIED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--close',action='store_true');args=parser.parse_args()
    if args.close:close()
    else:print(json.dumps(owned_processes()),flush=True)
