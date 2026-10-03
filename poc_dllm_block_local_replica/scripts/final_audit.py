"""Keep original worktree/index intact and audit final evidence without GPU use."""
import json
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from poc_dllm_block_local_replica.contract import *


def main():
    command=lambda args:subprocess.check_output(args,text=True)
    source=json.loads((ROOT/'results/reused_trace_manifest.json').read_text())
    checks=0
    for entry in source['requests']:
        for name,digest in entry['audit']['route_files'].items():
            assert sha(SOURCE/'raw/main_v2'/entry['request']/name)==digest
            checks+=1
    df=pd.read_csv(ROOT/'results/simulation/request_policy.csv')
    assert len(df)==40*4*6
    assert (df.peak_memory_gib<=df.budget_gib+1e-10).all()
    assert (df.post_bytes<=df.live_bytes).all() and (df.post_bytes>=0).all()
    assert np.allclose(df.residual_amp,df.post_bytes/df.B_one_shot,rtol=1e-12,atol=1e-12)
    assert (df[df.policy=='oracle_load'].worst_stage_load_increase<=.10+1e-10).all()
    frozen=json.loads((ROOT/'results/design_freeze.json').read_text())
    assert set(frozen['calibration']).isdisjoint(frozen['heldout'])
    summary=json.loads(final_summary_path().read_text())
    assert summary['verdict']=='NO-GO' and not summary['stage_b_promoted']
    assert summary['measured_runtime_gain'] is None
    hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and
            '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix in ('.py','.sh','.md','.json','.npz')}
    original=Path('/home/esjung/MLLM-EP-github')
    import hashlib
    staged=hashlib.sha256(subprocess.check_output(['git','-C',str(original),'diff','--cached','--binary'])).hexdigest()
    unstaged=hashlib.sha256(subprocess.check_output(['git','-C',str(original),'diff','--binary'])).hexdigest()
    assert staged=='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9'
    assert unstaged=='f56c2cbd47c4995364f072907ad7d886c935c3afe55d90e175a65abf0a4d941e'
    df[df.policy.str.startswith('oracle')].to_csv(ROOT/'results/oracle_budget.csv',index=False)
    df[~df.policy.str.startswith('oracle')].to_csv(ROOT/'results/causal_budget.csv',index=False)
    df[['request','policy','budget_gib','max_rank_load_increase','stage_critical_load_increase','worst_stage_load_increase','rank_max_mean','rank_cv','hottest_rank']].to_csv(ROOT/'results/load_balance.csv',index=False)
    new_json(ROOT/'results/final_verdict.json',dict(verdict=summary['verdict'],authoritative_summary=str(final_summary_path().relative_to(ROOT)),
        descriptive_candidate=summary['descriptive_candidate'],gates=summary['gates'],stage_b_promoted=False,
        runtime_correctness_accuracy_status='NOT_RUN_STAGE_A_NO_GO',measured_D_C_gain=None,measured_E2E_gain=None,
        next_action=summary['next_action']))
    new_json(ROOT/'results/final_audit.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_trace_hash_checks=checks,
        source_commit=SOURCE_COMMIT,source_trace_integrity=True,source_and_original_work_unchanged=True,
        original_staged_sha256=staged,original_unstaged_sha256=unstaged,
        branch=command(['git','branch','--show-current']).strip(),tests='6passed',
        physical_gpu_state=command(['nvidia-smi','--id='+MASK,'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader']),
        process_state=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
        cleanup='burner restored after physical session; final2,3,6,7 burner utilization verified',
        evidence_hashes=hashes,figure_review='residual_vs_budget.png visually inspected',
        statistical_unit='request/problem;24primaryheldout;6secondaryblock64heldout',
        no_fake_timing=True,no_runtime_replication=True,no_epoch_measured_claim=True))
    print('FINAL_AUDIT_PASS',checks,flush=True)


if __name__=='__main__':main()
