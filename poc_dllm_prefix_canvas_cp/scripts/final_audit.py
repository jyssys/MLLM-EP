"""Check preserved inputs, completed evidence and restored own burners."""
import datetime
import hashlib
import os
import subprocess
from poc_dllm_prefix_canvas_cp.contract import *
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor


def command(args,cwd=PROJECT):
    return subprocess.check_output(args,cwd=cwd,text=True)


def main():
    initial=json.loads((ROOT/'results/environment.json').read_text())
    observed={}
    for label,args in [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]:
        observed[label]=hashlib.sha256(subprocess.check_output(args,cwd=SPEC.parents[2])).hexdigest()
        assert observed[label]==initial[f'original_{label}_sha256'],'Original changes changed. Never reset them.'
    assert sha(SPEC)==sha(ROOT/'authoritative_spec.md')==initial['spec_sha256']
    diff=subprocess.check_output(['git','diff'],cwd=DINFER)
    assert hashlib.sha256(diff).hexdigest()==initial['dinfer_dirty_diff_sha256']
    assert command(['git','rev-parse','HEAD'],DINFER).strip()==initial['dinfer_revision']
    for path,digest in initial['dinfer_source_hashes'].items():
        assert sha(DINFER/'python/dinfer'/path)==digest
    # Preserve the pre-existing runtime patch as provenance, not a new change.
    patch=ROOT/'patches/dinfer_pre_existing_native.diff'
    patch.parent.mkdir(exist_ok=True)
    if patch.exists():assert patch.read_bytes()==diff
    else:patch.write_bytes(diff)
    processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'])
    selected=[]
    for line in processes.splitlines():
        fields=[v.strip() for v in line.split(',')]
        uuid=fields[0].removeprefix('GPU-')
        if uuid not in UUIDS:continue
        pid=int(fields[1]);worker_pid,argv=burner_worker_ancestor(pid)
        physical=GPUS[UUIDS.index(uuid)]
        assert f'--gpu {physical} ' in argv
        selected.append(dict(physical_gpu=physical,GPU_context_pid=pid,burner_worker_pid=worker_pid,
            worker_argv=argv,reported_memory_MiB=fields[3]))
    assert sorted(row['physical_gpu'] for row in selected)==GPUS
    checks=subprocess.run([PYTHON,'-m','unittest','discover','-s','poc_dllm_prefix_canvas_cp/tests','-v'],
        cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    assert checks.returncode==0,checks.stdout+checks.stderr
    manifest=json.loads((ROOT/'results/frozen_manifest.json').read_text())
    held_out={row['request_id'] for row in manifest['requests'] if row['split']=='held_out'}
    assert not any(any(p.name.startswith(rid+'.') for rid in held_out) for p in (ROOT/'results/raw').rglob('*.json.gz'))
    stage_b=ROOT/'results/raw/stage_b_refinement_order_v3'
    assert json.loads((stage_b/'guard_status.json').read_text())['success']
    references=[]
    for rank in range(4):
        rows=json.loads((stage_b/f'results.rank{rank}.json').read_text())['results']
        assert len(rows)==3
        for row in rows:
            assert all(e['pass_all'] for e in row['correctness'])
            references.extend(row['FP32_reference'])
    assert len(references)==24
    assert all(r['max_abs']<=.03125 and r['relative_L2']<=.01 for r in references)
    inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'results/raw').rglob('*')) if p.is_file()}
    write(ROOT/'results/raw_sha256.json',inventory)
    write(ROOT/'results/completion_audit.json',dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=command(['git','branch','--show-current']).strip(),
        branch_parent=initial['git_parent'],original_tracked_diff_sha256=observed,
        original_tracked_dirty_preserved=True,external_runtime_unchanged=True,
        runtime_patch_sha256=hashlib.sha256(diff).hexdigest(),spec_sha256=sha(SPEC),
        selected_GPU_processes=selected,
        GPU_state=command(['nvidia-smi','--id=4,5,6,7',
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader']),
        tests_output=checks.stdout+checks.stderr,raw_files=len(inventory),
        experiment_GPU_processes_remaining=False,selected_burners_restored=True,
        held_out_execution='NOT_RUN: all24 frozen request IDs absent from raw model-run files',
        FP32_reference_checks=len(references),FP32_reference_max_abs=max(x['max_abs'] for x in references),
        FP32_reference_max_relative_L2=max(x['relative_L2'] for x in references),
        scope='Read-only GPU checks. Only own exact-mask burners remain. No other GPU or user process was changed.'))
    print('Final checks PASS. Original changes and runtime preserved. Burners restored on',MASK)


if __name__=='__main__':main()
