"""Read-only ownership and provenance checks. Never signal a GPU process here."""
import datetime
import hashlib
import os
import subprocess
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor

def command(args,cwd=PROJECT):
    return subprocess.check_output(args,cwd=cwd,text=True)

def main():
    original='/home/esjung/MLLM-EP-github'
    continuity=read(ROOT/'artifacts/continuity.json')
    staged=hashlib.sha256(command(['git','diff','--cached','--binary'],original).encode()).hexdigest()
    unstaged=hashlib.sha256(command(['git','diff','--binary'],original).encode()).hexdigest()
    preserved=staged==continuity['original_staged_sha256'] and unstaged==continuity['original_unstaged_sha256']
    assert preserved,'Original tracked changes differ. Do not reset them.'
    for source,digest in continuity['sources'].items():assert sha(OLD_ROOT/source)==digest
    assert command(['git','rev-parse','HEAD'],CDB).strip()==CDB_REVISION
    assert command(['git','status','--porcelain'],CDB)==''
    for source,digest in continuity['specs'].items():assert sha(Path(source))==digest
    for i,spec in enumerate(SPECS,2):
        assert sha(spec)==sha(ROOT/f'specifications/authoritative_v{i}.md')
    allowed={uuid:gpu for uuid,gpu in zip(UUIDS,GPUS)}
    selected=[]
    for line in command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits']).splitlines():
        fields=[v.strip() for v in line.split(',')]
        uuid=fields[0].removeprefix('GPU-')
        if uuid not in allowed:continue
        pid=int(fields[1])
        ancestor,argv=burner_worker_ancestor(pid)
        gpu=allowed[uuid]
        assert f'--gpu {gpu} ' in argv
        selected.append(dict(physical_gpu=gpu,logical_burner_device=0,GPU_context_pid=pid,
            owned_burner_worker_pid=ancestor,process_name=fields[2],memory_MiB=int(fields[3])))
    assert sorted(r['physical_gpu'] for r in selected)==GPUS
    checks=subprocess.run([REFERENCE_PYTHON,'-m','unittest','-v','poc_decode_first_depth_aware_dp_v3.tests.test_contract'],
        cwd=PROJECT,text=True,capture_output=True)
    assert checks.returncode==0,checks.stdout+checks.stderr
    guards={p.parent.name:read(p)['success'] for p in (ROOT/'outputs').glob('*/guard_status.json')}
    required=['live_trace_v1','live_clean_v1','hindsight_confirmation_v1','hindsight_confirmation_c64_v1',
        'batched_kv_probe_v2','cost_surface_v1']
    assert all(guards.get(r) is True for r in required)
    assert guards.get('batched_kv_probe_v1') is False,'The failed observer log must remain visible.'
    write(ROOT/'artifacts/final_audit.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=command(['git','branch','--show-current']).strip(),base_commit=BASE_COMMIT,
        original_staged_sha256=staged,original_unstaged_sha256=unstaged,
        original_tracked_changes_preserved=preserved,previous_v2_sources_preserved=True,
        selected_GPU_processes=selected,burners_restored=True,experiment_GPU_processes_remaining=False,
        GPU_state=command(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader']),
        guard_success=guards,tests=checks.stdout+checks.stderr,
        scope='Only physical GPUs 4,5,6,7 operated. Other users and GPUs were not signalled.',
        final_decision=read(ROOT/'artifacts/final_decision.json')))
    inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'outputs').rglob('*'))
        if p.is_file() and p.name!='clean_start.json' and p.suffix!='.pyc'
        and not p.relative_to(ROOT).match('outputs/*/rank_*/*.json')}
    write(ROOT/'artifacts/raw_sha256.json',inventory)
    print('FINAL_AUDIT_PASS: old work preserved; allowed burners restored; no experiment GPU process remains.')

if __name__=='__main__':main()
