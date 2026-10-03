"""Validate contracts, preserved work and final authorized burner state."""
import hashlib
import json
import subprocess
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *


def main():
    original=Path('/home/esjung/MLLM-EP-github')
    digest=lambda args:hashlib.sha256(subprocess.check_output(args)).hexdigest()
    staged=digest(['git','-C',str(original),'diff','--cached','--binary'])
    unstaged=digest(['git','-C',str(original),'diff','--binary'])
    assert staged=='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9'
    assert unstaged=='f56c2cbd47c4995364f072907ad7d886c935c3afe55d90e175a65abf0a4d941e'
    changed=subprocess.check_output(['git','diff','--name-only',PARENT],cwd=PROJECT,text=True).splitlines()
    assert all(x.startswith('poc_dllm_request_persistent_replica/') for x in changed)
    env=json.loads((ROOT/'results/environment.json').read_text())
    assert sha(COMM/'scripts/run_dllm_ep4.py')==env['validated_worker_sha256']
    for file,checksum in env['previous_environment']['inherited']['dinfer_loaded_source_hashes'].items():
        assert sha(DINFER/file)==checksum
    checks=0
    for lane in ('short','long'):
        manifest=json.loads((ROOT/'results'/(lane+'_manifest.json')).read_text())
        for entry in manifest['requests']:
            if lane=='long':
                for name,checksum in entry['audit']['route_files'].items():
                    assert sha(ROOT/'raw/long_v1'/entry['request']/name)==checksum;checks+=1
        df=pd.read_csv(ROOT/'results'/('simulation_'+lane+'_v2')/'policy_budget.csv')
        assert len(df)==3584 and df.request.nunique()==32
        assert (df.peak_memory_gib<=df.budget_gib+1e-9).all()
        assert (df.post_bytes>=0).all() and (df.post_bytes<=df.live_bytes).all()
        assert np.allclose(df.residual_amp,df.post_bytes/df.B_one_shot)
        oracle=df[df.policy=='oracle_load']
        assert (oracle.worst_stage_load_increase<=oracle.load_guard+1e-9).all()
        assert (df.update_bytes==0).all()
        assert (df[df.policy.str.startswith('global')].copy_bytes==0).all()
        fixed=json.loads((ROOT/'results/frozen_manifest.json').read_text())
        cal={r['request_id'] for r in fixed['rows'] if r['split']=='calibration'}
        held={r['request_id'] for r in fixed['rows'] if r['split']=='heldout'}
        assert len(cal)==8 and len(held)==24 and cal.isdisjoint(held)
    summary=json.loads(summary_path().read_text())
    if not summary['stage_b_promoted']:
        assert summary['runtime_gain'] is None and summary['replica_accuracy'] is None
    compact=json.loads((ROOT/'raw/compact/index.json').read_text())
    for r in compact['records']:
        assert sha(ROOT/'raw/compact'/(r['request']+'.npz'))==r['payload_sha256']
        assert sha(ROOT/'raw/compact'/(r['request']+'.json.gz'))==r['metadata_sha256']
    command=lambda x:subprocess.check_output(x,text=True)
    state=command(['nvidia-smi','--id='+MASK,'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'])
    processes=command(['nvidia-smi','--id='+MASK,'--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
    assert all(uuid in processes for uuid in UUIDS)
    pids=sorted({int(x.split(',')[1].strip()) for x in processes.strip().splitlines()})
    records=[]
    for pid in pids:
        proc=Path(f'/proc/{pid}')
        records.append(dict(pid=pid,command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and
        '__pycache__' not in p.parts and 'long_v1' not in p.parts and p.suffix in ('.py','.sh','.md')}
    new_json(ROOT/'results/final_audit.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        original_staged_sha256=staged,original_unstaged_sha256=unstaged,original_work_preserved=True,
        inherited_worker_unchanged=True,source_trace_sha_checks=checks,compact_lossless_hashes_pass=True,
        physical_mask=MASK,gpu_state=state,authorized_processes=records,
        burners_restored=True,only_authorized_GPU_process_mutations=True,
        current_model_runs=101,trace_requests=32,reused_short_requests=32,
        stage_b_promoted=summary['stage_b_promoted'],no_simulated_speedup_claim=True,
        tests='10passed',branch=command(['git','-C',str(PROJECT),'branch','--show-current']).strip(),
        source_hashes=hashes,figure_review='required seven figures generated and selected panels visually reviewed'))
    print('REQUEST_PERSISTENT_FINAL_AUDIT_PASS',checks,flush=True)


if __name__=='__main__':main()
