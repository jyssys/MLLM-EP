"""Final read-only checks. Write only this campaign's generated audit."""
import hashlib
import subprocess
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read


def main():
    initial=read(ROOT/'results/env_and_pins.json');progress=read(ROOT/'results/campaign_progress.json')
    checks=dict(all_24_configs_complete=len(progress)==24 and all(r['status']=='COMPLETE' for r in progress),
        spec_unchanged=sha(ROOT/'authoritative_spec.md')==initial['spec_sha256']==sha(SPEC),
        exactly_expected_model_pins=initial['models']==MODELS,
        only_own_tracked_changes=all(name.startswith(ROOT.name+'/') for name in subprocess.check_output(
            ['git','diff','HEAD','--name-only'],cwd=PROJECT,text=True).splitlines()))
    source=read(ROOT/'results/source_and_model_audit.json')
    checks['installed_runtime_source_unchanged']=all(sha(row['path'])==row['sha256'] for row in source['sources'])
    hashes=read(ROOT/'results/hot_source_hashes.json')
    checks['one_primary_hot_policy']=len(hashes)==3 and all(len(v)==1 for v in hashes.values())
    checks['hot_policy_matches_current_code']=all(sha(ROOT/key)==values[0] for key,values in hashes.items())
    parity=read(ROOT/'results/parity_and_kv_validation.json')
    checks['all_126_same_TP_R_fields_bitwise']=len(parity)==126 and all(row['bitwise_equal'] for row in parity)
    diagnostic=read(ROOT/'results/diagnostic_progress_v4.json')
    rechecks=read(ROOT/'results/recheck_progress.json')
    checks['bounded_diagnostic_queue_complete']=len(diagnostic)==10 and all(r.get('exit_code')==0 for r in diagnostic)
    checks['all_four_same_policy_rechecks_complete']=len(rechecks)==4 and all(r.get('exit_code')==0 for r in rechecks)
    verdict=read(ROOT/'results/verdict.json')
    natural=read(ROOT/'results/natural_native_tied_parity.json')
    checks['natural_failures_do_not_become_positive_claims']=all(r['same_token_IDs'] for r in natural) or verdict['verdict']=='CONTROL_BLOCKED'
    checks['frozen_zero_tolerance_preserved']=verdict['tolerance']==PARITY==initial['parity_contract']
    checks['all_primary_recheck_kwargs_match']=all(r['exact_engine_kwargs_match'] for r in read(ROOT/'results/same_policy_recheck_contracts.json'))
    required=['report.md','architecture_matrix.md','workload_contract.md','timing_contract.md','tied_unroll_contract.md','prior_overlap.md','reproduce.md']
    checks['required_reports_exist']=all((ROOT/name).exists() for name in required)
    report_text=(ROOT/'report.md').read_text()
    checks['report_has_no_draft_placeholders']='PENDING_' not in report_text and 'DRAFT_COLLECTION_RUNNING' not in report_text
    original=Path('/home/esjung/MLLM-EP-github')
    now={k:hashlib.sha256(subprocess.check_output(cmd,cwd=original)).hexdigest() for k,cmd in [
        ('staged',['git','diff','--cached']),('unstaged',['git','diff'])]}
    checks['original_staged_unstaged_unchanged']=now==initial['original_diffs']
    manifest=read(ROOT/'results/heavy_evidence_manifest.json')
    checks['archived_heavy_evidence_exact']=all(Path(row['external_path']).exists() and sha(row['external_path'])==row['sha256'] for row in manifest['files'])
    masks=[]
    for path in (ROOT/'results').rglob('run_contract.json'):
        value=read(path);mask=value['mask'];masks.append(mask)
        assert mask in ['4','4,5','4,5,6,7'],(path,mask)
    checks['all_recorded_GPU_masks_permitted']=bool(masks)
    checks['final_scoped_burners_running']=read(ROOT/'results/final_burner_restore.json')['verification']=='RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs'
    report=dict(checks=checks,all_pass=all(checks.values()),original_diff_hashes_now=now,
        GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True),
        selected_GPU_status=subprocess.check_output(['nvidia-smi','--id=4,5,6,7','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        limitation='Read-only checks of other GPUs do not authorize process control. Final selected-device burner ownership also requires process inspection.')
    write(ROOT/'results/final_audit.json',report);print(json.dumps(checks,indent=2))
    assert report['all_pass'],'Do not claim full completion while an audit check fails'


if __name__=='__main__':main()
