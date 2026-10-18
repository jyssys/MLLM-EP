"""Check evidence, unrelated tracked changes, and permitted burner ancestry."""
import argparse
import datetime
import hashlib
import os
import subprocess
from poc_decode_first_depth_aware_dp.contract import *
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    frozen=read(ROOT/'artifacts/preregistration.json')
    original=Path('/home/esjung/MLLM-EP-github')
    observed={label:hashlib.sha256(subprocess.check_output(command,cwd=original)).hexdigest()
        for label,command in [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]}
    assert observed==frozen['original_tracked_diff_sha256']
    assert sha(SPEC)==sha(ROOT/'reports/spec.md')==frozen['spec_sha256']
    selected=[]
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True)
    for line in inventory.splitlines():
        uuid,pid,used=[v.strip() for v in line.split(',')]
        uuid=uuid.removeprefix('GPU-')
        if uuid not in UUIDS:continue
        worker,argv=burner_worker_ancestor(int(pid));physical=GPUS[UUIDS.index(uuid)]
        assert f'--gpu {physical} ' in argv
        selected.append(dict(physical_gpu=physical,context_pid=int(pid),verified_burner_worker=worker,memory_MiB=used))
    assert sorted(r['physical_gpu'] for r in selected)==[4,5,6,7]
    test=subprocess.run([REFERENCE_PYTHON,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    assert test.returncode==0,test.stdout+test.stderr
    payloads={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in ROOT.rglob('*.npz')}
    write(ROOT/'artifacts/local_array_manifest.json',dict(files=payloads,
        scope='Local reference arrays, not model weights. Reproduce with the recorded physical commands.'))
    raw={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'outputs').rglob('*'))
        if p.is_file() and p.suffix in ['.json','.csv','.gz','.log','.zip'] and p.name!='clean_start.json'}
    write(ROOT/'artifacts/raw_sha256.json',raw)
    write(args.output,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip(),
        recorded_parent=frozen['parent'],spec_sha256=frozen['spec_sha256'],
        original_tracked_diff_sha256=observed,original_tracked_dirty_preserved=True,
        final_GPU_contexts=selected,burners_restored=True,experiment_processes_remaining=False,
        test_output=test.stdout+test.stderr,raw_files=len(raw),
        scope='Read-only final GPU audit. No unknown or foreign process is signalled.'))
    print('FINAL_AUDIT_PASS',len(selected),'verified own burners; dirty worktree unchanged')


if __name__=='__main__':main()
