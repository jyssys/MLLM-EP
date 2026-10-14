"""Check provenance, completed evidence, parity, CPU tests, and burner ownership."""
import datetime
import gzip
import os
import subprocess
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor


def command(args,cwd=PROJECT):
    return subprocess.check_output(args,cwd=cwd)


def load(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:return json.load(stream)
    return json.loads(path.read_text())


def main():
    environment=load(ROOT/'manifests/environment.json')
    original=Path('/home/esjung/MLLM-EP-github')
    observed={}
    for label,args in [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]:
        observed[label]=hashlib.sha256(command(args,original)).hexdigest()
        assert observed[label]==environment[f'original_{label}_sha256'], 'Original tracked work changed. Never reset it.'
    assert command(['git','rev-parse','HEAD'],DINFER).decode().strip()==environment['dinfer_revision']
    assert hashlib.sha256(command(['git','diff'],DINFER)).hexdigest()==environment['dinfer_diff_sha256']
    for source,digest in environment['source_hashes'].items():
        assert sha(DINFER/'python/dinfer'/source)==digest,source
    assert sha(SPEC)==sha(ROOT/'authoritative_spec.md')==environment['spec_sha256']
    geometry=load(ROOT/'manifests/model_geometry.json')
    assert sha(MODEL/'config.json')==geometry['config_sha256']
    for split in ['calibration','held_out']:
        manifest=load(ROOT/f'manifests/workload_{split}.json')
        assert sha(PROJECT/manifest['source'])==manifest['source_sha256']
        assert sha(PROJECT/manifest['token_artifact'])==manifest['token_sha256']
    frozen=load(ROOT/'manifests/frozen_control.json')
    assert sha(ROOT/'results/architecture_decision.json')==frozen['architecture_screen_sha256']
    assert sha(ROOT/'manifests/profile_validity.json')==frozen['profile_validity_sha256']
    validity=load(ROOT/'manifests/timing_validity.json')
    assert validity['all_parity'] and len(validity['checks'])==896
    assert all(row['pass_all'] for row in validity['checks'])
    control=load(ROOT/'results/control_summary.json')
    assert all(row['pass_all'] for row in control['checks'])
    completed=[]
    campaign=['native_p512_v2','native_p4096_v1','native_p8192_v1','native_p16384_v1',
        'compact_calibration_p16384_v1','compact_heldout_p16384_v1','compact_calibration_p4096_v1']
    for name in campaign:
        directory=ROOT/'results/raw'/name
        assert load(directory/'guard_status.json')['success'],name
        assert load(directory/'status.json')['success'],name
        for rank in range(4):assert load(directory/f'worker_complete.rank{rank}.json')['success'],(name,rank)
        if name.startswith('compact_'):
            expected=24 if 'heldout' in name else 8
            assert control['groups'][name]['N']==expected
            for rank in range(4):
                parity=load(directory/f'parity.rank{rank}.json');assert len(parity)==expected
                for row in parity:
                    assert row['output_acceptance_NFE_exact'] and row['regular_graph_logits_exact']
                    assert len(row['regular_graph_KV_exact'])==40 and all(row['regular_graph_KV_exact'])
                    if 'independent_output_storage' in row:
                        assert len(row['independent_output_storage'])==40 and all(row['independent_output_storage'])
                    else:
                        assert name=='compact_calibration_p16384_v1', 'Held-out storage proof must be explicit'
        completed.append(name)
    for record in load(ROOT/'results/external_profile_inventory.json'):
        path=Path(record['path'])
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],str(path)
    processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits']).decode()
    selected=[]
    for line in processes.splitlines():
        fields=[v.strip() for v in line.split(',')]
        if fields[0].removeprefix('GPU-') not in UUIDS:continue
        physical=GPUS[UUIDS.index(fields[0].removeprefix('GPU-'))]
        worker_pid,argv=burner_worker_ancestor(int(fields[1]))
        assert f'--gpu {physical} ' in argv
        selected.append(dict(physical_gpu=physical,GPU_context_pid=int(fields[1]),
            burner_worker_pid=worker_pid,worker_argv=argv,reported_memory_MiB=fields[3]))
    assert sorted(row['physical_gpu'] for row in selected)==GPUS
    tests=subprocess.run([PYTHON,'-m','unittest','discover','-v','-s',str(ROOT/'tests')],
        cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    subprocess.run([PYTHON,'-m','compileall','-q',str(ROOT)],cwd=PROJECT,check=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    inventory={str(path.relative_to(ROOT)):sha(path)
        for path in sorted((ROOT/'results/raw').rglob('*')) if path.is_file()}
    assert max(path.stat().st_size for path in ROOT.rglob('*') if path.is_file())<95*1024**2
    write(ROOT/'results/raw_sha256.json',inventory)
    write(ROOT/'results/completion_audit.json',dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=command(['git','branch','--show-current']).decode().strip(),
        branch_parent=environment['parent'],original_tracked_diff_sha256=observed,
        original_tracked_dirty_preserved=True,external_runtime_source_unchanged=True,
        frozen_spec_model_cohort_control_hashes_unchanged=True,completed_campaigns=completed,
        all_native_lane_rank_parity_checks=len(validity['checks']),
        all_control_lane_rank_parity_checks=len(control['checks']),
        exact_control_KV_logits_output_acceptance_NFE=True,
        candidate_output_storage_independent='Explicitly checked in held-out and secondary P4096 calibration. Not recorded in initial P16384 calibration.',
        tests_output=tests.stdout+tests.stderr,
        raw_files=len(inventory),external_Nsight_originals_hash_verified=True,
        selected_GPU_processes=selected,
        GPU_state=command(['nvidia-smi','--id=4,5,6,7',
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader']).decode(),
        experiment_GPU_processes_remaining=False,selected_burners_restored=True,
        verdict='GENERIC_ONLY',fine_critical_DAG='PARTIAL',official_benchmark_accuracy='NOT_RUN',
        scope='Read-only preservation and GPU ownership checks. Only PoC-owned audit files written.'))
    print('Audit PASS. Frozen evidence and original changes preserved. Tests PASS. Burners4/5/6/7 restored.')


if __name__=='__main__':main()
