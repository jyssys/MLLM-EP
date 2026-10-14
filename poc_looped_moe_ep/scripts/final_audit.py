"""Read-only final checks plus a compact, PoC-owned completion record."""
import datetime
import hashlib
import os
import subprocess
from poc_looped_moe_ep.contract import *


def command(args,cwd=PROJECT):
    return subprocess.check_output(args,cwd=cwd,text=True)


def burner_worker_ancestor(pid):
    # The GPU context belongs to a VLLM EngineCore child, not utilize.py itself.
    for _ in range(16):
        proc=Path('/proc')/str(pid)
        assert proc.stat().st_uid==os.getuid()
        argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'utilize.py --worker --gpu ' in argv:return pid,argv
        status=(proc/'status').read_text().splitlines()
        pid=int(next(line.split()[1] for line in status if line.startswith('PPid:')))
        assert pid>1,'GPU process has no own burner-worker ancestor'
    raise AssertionError('Burner ancestry exceeds bounded depth')


def main():
    original=Path('/home/esjung/MLLM-EP-github')
    expected={
        'staged':'7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9',
        'unstaged':'f56c2cbd47c4995364f072907ad7d886c935c3afe55d90e175a65abf0a4d941e'}
    observed={}
    for label,args in [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]:
        observed[label]=hashlib.sha256(subprocess.check_output(args,cwd=original)).hexdigest()
    assert observed==expected,'Original tracked dirty/staged state changed; investigate, never reset it'
    processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                       '--format=csv,noheader,nounits'])
    selected=[]
    for line in processes.splitlines():
        fields=[v.strip() for v in line.split(',')]
        if fields[0].removeprefix('GPU-') not in UUIDS:continue
        pid=int(fields[1]);worker_pid,argv=burner_worker_ancestor(pid)
        physical=GPUS[UUIDS.index(fields[0].removeprefix('GPU-'))]
        assert f'--gpu {physical} ' in argv
        selected.append(dict(physical_gpu=physical,GPU_context_pid=pid,burner_worker_pid=worker_pid,
            worker_argv=argv,reported_memory_MiB=fields[3]))
    assert sorted(r['physical_gpu'] for r in selected)==GPUS
    checks=subprocess.run([PYTHON,'-m','unittest','-v','poc_looped_moe_ep.tests.test_contract'],
        cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    assert checks.returncode==0,checks.stdout+checks.stderr
    inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'results/raw').rglob('*')) if p.is_file()}
    write(ROOT/'results/raw_sha256.json',inventory)
    write(ROOT/'results/completion_audit.json',dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=command(['git','branch','--show-current']).strip(),
        branch_parent=command(['git','rev-parse','HEAD']).strip(),
        original_tracked_diff_sha256=observed,original_tracked_dirty_preserved=True,
        spec_sha256=sha(ROOT/'authoritative_spec.md'),selected_GPU_processes=selected,
        GPU_state=command(['nvidia-smi','--id=4,5,6,7',
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader']),
        tests_output=checks.stdout+checks.stderr,raw_files=len(inventory),
        experiment_GPU_processes_remaining=False,selected_burners_restored=True,
        scope='Only exact-mask own burners restored; other GPUs/users untouched'))
    print('Final checks PASS; original tracked state unchanged; burners restored on',MASK)


if __name__=='__main__':main()
