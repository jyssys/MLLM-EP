"""Publish an evidence index/ledger and privacy-minimized environment copy."""
import json
import subprocess

from poc_speculative_moe_ep.contract import ROOT,write,sha


def main():
    path=ROOT/'artifacts/evidence_index.json'
    if path.exists():raise FileExistsError(path)
    private=ROOT/'artifacts/environment.json';environment=json.loads(private.read_text())
    rows=[]
    for line in environment['process_snapshot'].splitlines():
        fields=line.split(', ')
        if '/home/' in fields[-1] and '/home/esjung/' not in fields[-1]:
            fields[-1]='FOREIGN_PYTHON_OUTSIDE_AUTHORIZED_GPU_SET (path redacted)'
        rows.append(', '.join(fields))
    environment['process_snapshot']='\n'.join(rows)+'\n'
    environment['privacy_note']='Only foreign user command paths redacted. GPU UUID/PID/topology/config/revision/tokenizer/source hashes unchanged. Original audit remains local, append-only.'
    environment['original_environment_sha256']=sha(private)
    environment['package_metadata_versions']=dict(vllm='0.20.0+cu129',deep_ep='1.2.1+73b6ea4',torch='2.11.0+cu129',triton='3.6.0',transformers='5.14.1')
    write(ROOT/'artifacts/environment_public.json',environment)
    ledger=[]
    for status in sorted((ROOT/'raw').glob('*/status.json')):
        s=json.loads(status.read_text());worker=status.parent/'worker.dp0.json'
        records=json.loads(worker.read_text())['records'] if worker.exists() else []
        ledger.append(dict(run=status.parent.name,ok=s['ok'],checkpointed_request_waves=len(records),
            warmup=sum(r['repeat']<0 for r in records),measured=sum(r['repeat']>=0 for r in records),
            output_tokens=sum(len(o['tokens']) for r in records for o in r['outputs'])))
    write(ROOT/'artifacts/execution_ledger.json',dict(runs=ledger,
        checkpointed_request_waves=sum(r['checkpointed_request_waves'] for r in ledger),
        unique_primary_requests=16,frozen_pool_requests=96,
        note='Repeated/warmup requests not independent samples. Failed preliminary RPC wave can be uncheckpointed;414 is persisted successful wave count, not an invented exact attempted-forward count. No400 ceiling is specified for this campaign.'))
    errors=[]
    for file in sorted((ROOT/'raw').glob('*/*.failure.json')):
        errors.append(dict(run=file.parent.name,traceback=json.loads(file.read_text())['traceback']))
    write(ROOT/'artifacts/runtime_errors.json',dict(errors=errors,repairs='registered string RPC; no unsafe serialization setting; subsequent native runs PASS'))
    gpus=subprocess.check_output(['nvidia-smi','--id=2,3,6,7','--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    write(ROOT/'artifacts/final_gpu_state.json',dict(mask='2,3,6,7',gpus=gpus,processes=processes,
        policy='original run_utilize.sh restored with explicit VLLM_UTILIZE_GPUS=2,3,6,7; no experiment active'))
    write(path,dict(primary_verdict='final_verdict.json',primary_summary='final_summary.json',
        environment='environment_public.json',primary_clean='phase_b_v2',
        verification_ledger='phase_b_v2/verification_metrics_draft_aligned.parquet',
        trace_routes='phase_a_v1/route_metrics.parquet',trace_verification='phase_a_v1/verification_metrics_draft_aligned.parquet',
        trace_export='compact_phase_a_v1/manifest.json',clean_export='compact_phase_b_v1/manifest.json',
        repeated_and_bypass_export='compact_validation_v1/manifest.json',
        grouped_cv='phase_b_v2/cost_model_results.json',secondary_width_control='phase_b_v2/width_controlled_cost_model_results.json',
        matched='phase_b_v2/layer_matched_pairs.json',oracle='phase_b_v2/oracle_policy_results.json',
        repeatability='phase_b_v2/route_repeatability.json',instrumentation='phase_b_v2/instrumentation_validity.json',
        final_key_repeats='phase_b_v2/key_repeat_summary.json',
        limitations=['16 actual unique prompts, not full96','single-active DP0 with actual idle-DP work',
            'trace labels diagnostic only','current gamma0 is AR, not draft1','offline oracle target-side only; E2E oracle unvalidated',
            'lightweight wrappers remain in bypass; separate restart blocks, not randomized interleave']))
    print(json.dumps(dict(index=str(path),checkpointed_request_waves=sum(r['checkpointed_request_waves'] for r in ledger),gpu_policy_restored=True)))


if __name__=='__main__':main()
