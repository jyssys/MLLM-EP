"""Close only drained owned work and verify persistent all-eight burners."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from contextlib import ExitStack
from .common import ROOT,QUIET_LOCK,write,digest,preserve,resources


def active_gpu_actors():
    rows=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if '-m' not in words:continue
            module=words[words.index('-m')+1]
            if module not in ('poc_flashvep.dense_ar_longbench400_pd.serving',
                'poc_flashvep.dense_ar_p_critical_path.attention_capture',
                'poc_flashvep.dense_ar_p_critical_path.attention_benchmark'):continue
            if any(word.startswith(str(ROOT)+'/') for word in words):rows.append(dict(pid=int(proc.name),words=words))
        except (OSError,UnicodeError,IndexError):continue
    return rows


def close():
    preserve()
    for relative in ('analysis/cpu_regression_audit.json','analysis/raw_metric_recalculation.json',
        'analysis/configuration_and_gate_inventory.json','analysis/gpu_budget_accounting.json','state/science_completion_audit.json'):
        value=json.loads((ROOT/relative).read_text());assert value['passed'],relative
    assert json.loads((ROOT/'analysis/configuration_and_gate_inventory.json').read_text())['final']
    assert json.loads((ROOT/'analysis/raw_metric_recalculation.json').read_text())['final']
    assert not active_gpu_actors(),'Wait for every owned standard and Attention-PoC GPU actor'
    assert all(json.loads(path.read_text()).get('status')!='RUNNING' for path in (ROOT/'state').glob('pair_lease_*.json'))
    write(ROOT/'state/STOP_DISPATCH.json',dict(time=time.time(),reason='RESEARCH_AND_GPU_GATES_AUDITED'))
    # No dispatcher can enter a new pair while these locks are held.
    with ExitStack() as stack:
        for left in (0,2,4,6):
            lock=stack.enter_context((ROOT/'state'/f'pair_{left}_{left+1}.lock').open('a+'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not active_gpu_actors()
        write(ROOT/'state/STOP_EXECUTION.json',dict(time=time.time(),reason='ALL_OWNED_GPU_ACTORS_DRAINED'))
        write(ROOT/'state/STOP_GUARDIAN.json',dict(time=time.time(),reason='ALL_PAIR_LEASES_RELEASED'))
        deadline=time.monotonic()+60
        while json.loads((ROOT/'state/guardian.json').read_text())['status']!='RELEASED':
            assert time.monotonic()<deadline,'Inspect the guard without terminating unrelated work'
            time.sleep(1)
        with QUIET_LOCK.open('a+') as global_lock:
            fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            owner=resources();processes=owner.processes();approved=owner.burners()
            assert all(os.getsid(row['pid'])==approved.get(row['gpu']) for row in processes)
            initial=json.loads((ROOT/'state/environment_initial.json').read_text())
            assert digest(Path(owner.SCRIPT))==initial['burner_script_SHA256']
            owner.stop(list(range(8)));owner.restore(list(range(8)))
            samples=[];consecutive=0;until=time.monotonic()+300
            while consecutive<2:
                text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'],text=True)
                devices=[dict(GPU=int(a),UUID=b.strip(),utilization_percent=int(c),memory_MiB=int(d))
                    for a,b,c,d in (line.split(',') for line in text.strip().splitlines())]
                processes=owner.processes();approved=owner.burners()
                owned=all(Path('/proc',str(row['pid'])).stat().st_uid==os.getuid()
                    and os.getsid(row['pid'])==approved.get(row['gpu']) for row in processes)
                all_devices={row['gpu'] for row in processes}==set(range(8))==set(approved)
                valid=owned and all_devices and all(row['utilization_percent']>=70 for row in devices)
                samples.append(dict(time=time.time(),devices=devices,owned=owned,all_devices=all_devices))
                consecutive=consecutive+1 if valid else 0
                assert time.monotonic()<until,'Leave the approved launch running and inspect slow startup'
                if consecutive<2:time.sleep(2)
            write(ROOT/'state/CAMPAIGN_GPU_WORK_CLOSED.json',dict(time=time.time(),passed=True,
                reason='No owned GPU actor remains. All pair leases released before approved burner restoration.'))
            write(ROOT/'state/resource_closeout.json',dict(time=time.time(),passed=True,GPUs=list(range(8)),
                active_processes=processes,burner_sessions=approved,samples=samples,
                burner_script=owner.SCRIPT,burner_script_SHA256=digest(Path(owner.SCRIPT)),
                bash_wrapper='/home/esjung/vllm-ep/run_utilize.sh',
                bash_wrapper_SHA256=digest(Path('/home/esjung/vllm-ep/run_utilize.sh')),
                all_eight_explicit_environment=True,two_consecutive_owned_utilization_samples=True,
                user_checkout_and_index_preserved=True,unknown_processes_terminated=False,
                scope='Stop only approved burner sessions after all research GPU work drains. '
                    'The existing bash wrapper launches GPUs0–7 through explicit VLLM_UTILIZE_GPUS and the approved Python. '
                    'Keep these persistent burners alive after publication. Operational utilization is not kernel-busy research evidence.'))
    preserve();print('P_CRITICAL_ALL8_BURNERS_VERIFIED',flush=True)


if __name__=='__main__':close()
