"""Check scope, all required rows, raw evidence, and the final burner state."""
import datetime
import gzip
import hashlib
import os
import subprocess
from poc_depth_cdb_factorial.contract import *
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor

def command(args,cwd=PROJECT):
    return subprocess.check_output(args,cwd=cwd,text=True)

def main():
    c=read(ROOT/'artifacts/continuity.json')
    original='/home/esjung/MLLM-EP-github'
    staged=hashlib.sha256(command(['git','diff','--cached','--binary'],original).encode()).hexdigest()
    unstaged=hashlib.sha256(command(['git','diff','--binary'],original).encode()).hexdigest()
    assert staged==c['original_staged_sha256'] and unstaged==c['original_unstaged_sha256']
    for source,digest in c['sources'].items():assert sha(PROJECT/source)==digest
    assert sha(SPEC)==c['spec_sha256']==sha(ROOT/'reports/spec.md')
    assert command(['git','rev-parse','HEAD'],CDB).strip()==CDB_REVISION
    assert command(['git','status','--porcelain'],CDB)==''
    rows=read(ROOT/'artifacts/metrics_raw.json')
    clean=[r for r in rows if r['kind']=='speed' and not r['trace'] and not r['validation']]
    assert len(clean)==125
    for C in [4,8,16,32,64]:
        for arm in ARMS:
            assert sorted(r['repeat'] for r in clean if r['C']==C and r['arm']==arm)==[0,1,2,3,4]
    quality=read(ROOT/'artifacts/quality_summary.json')
    assert len(quality)==10 and all(r['N']==128 for r in quality)
    for path in (ROOT/'outputs/smoke_validation_v1').glob('rank_*/*.json'):
        raw=read(path)
        assert all(r['exact'] for r in raw['checks'])
        assert raw['layer_calls']==24*(4*raw['stats']['prefill_batches']+raw['stats']['recurrent_batches'])
        if not raw['refill']:assert raw['stats']['scheduler_refills']==0
    required=['speed_A0_r0_v1','speed_A0_r1_v1','speed_A0_r2_v1','speed_DP4_r0_v1','speed_DP4_r1_v1','speed_DP4_r2_v1',
        'speed_A0_r3_v1','speed_A0_r4_v1','speed_DP4_r3_v1','speed_DP4_r4_v1',
        'quality_A0_v1','quality_DP4_v1','trace_A0_v1','trace_DP4_v1','smoke_validation_v1','sensitivity_DP4_v1']
    required.append('order_control_DP4_v1')
    guards={p.parent.name:read(p)['success'] for p in (ROOT/'outputs').glob('*/guard_status.json')}
    assert all(guards.get(name) for name in required),guards
    for row in read(ROOT/'artifacts/raw_compression_manifest.json'):
        assert sha(ROOT/row['original'])==row['sha256']
        assert hashlib.sha256(gzip.decompress((ROOT/row['published']).read_bytes())).hexdigest()==row['sha256']
    allowed={uuid:gpu for uuid,gpu in zip(UUIDS,GPUS)}
    burners=[]
    for line in command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']).splitlines():
        fields=[x.strip() for x in line.split(',')]
        uuid=fields[0].removeprefix('GPU-')
        if uuid not in allowed:continue
        pid=int(fields[1])
        ancestor,argv=burner_worker_ancestor(pid)
        assert f'--gpu {allowed[uuid]} ' in argv
        burners.append(dict(physical_GPU=allowed[uuid],context_pid=pid,worker_pid=ancestor))
    assert sorted(r['physical_GPU'] for r in burners)==GPUS
    tests=subprocess.run([REFERENCE_PYTHON,'-m','unittest','-v','poc_depth_cdb_factorial.tests.test_contract'],cwd=PROJECT,text=True,capture_output=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    reports=['environment_and_reuse','five_arm_functional_parity','quality_five_arms','latency_five_arms',
        'cdb_interaction_analysis','concurrency_and_memory','final_report']
    assert all((ROOT/'reports'/f'{name}.md').exists() for name in reports)
    write(ROOT/'artifacts/final_audit.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=command(['git','branch','--show-current']).strip(),base_commit=BASE_COMMIT,
        original_dirty_staged_preserved=True,preceding_evidence_preserved=True,all_125_primary_cohorts_measured=True,
        quality_questions_per_arm_per_domain=128,guards=guards,burners_restored=burners,
        GPU_process_policy='Only owned burners remain on physical GPUs 4,5,6,7. Other users were not signalled.',
        tests=tests.stdout+tests.stderr,final_decision=read(ROOT/'artifacts/final_decision.json')))
    print('FINAL_AUDIT_PASS')

if __name__=='__main__':main()
