"""Freeze the policy/split contract before any replica outcome calculation."""
import json
import random
import subprocess
from datetime import datetime, timezone
from poc_dllm_block_local_replica.contract import *


def main():
    original = json.loads((SOURCE/'results/frozen_manifest.json').read_text())
    rows = original['rows']
    calibration = []
    for domain in ('gsm8k', 'humaneval'):
        ids = sorted(r['request_id'] for r in rows if r['domain'] == domain)
        random.Random(8611).shuffle(ids)
        calibration.extend(ids[:4])
    heldout = sorted(r['request_id'] for r in rows if r['request_id'] not in calibration)
    new_json(ROOT/'results/design_freeze.json', dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), source_commit=SOURCE_COMMIT,
        spec_sha256=sha(SPEC), calibration=sorted(calibration), heldout=heldout,
        split_rule='metadata-only seeded8611 permutation;4/domain calibration,12/domain heldout; replication outcomes not inspected',
        primary='HumanEval communication-exposed B1 block32; GSM8K secondary; balanced pooled continuity also reported',
        prior_trace_caveat='Previously analyzed for amplification, not fresh model outputs; replica policy holdout is newly frozen.',
        budgets_gib_per_rank=BUDGETS, expert_bytes=EXPERT_BYTES,
        memory_rule='one shared budget per source rank across ALL19 sparse layers, renewed per native block',
        profiling_cost='first1/2 complete refinement iterations retain baseline traffic; replicas apply only after their completion',
        policies=['none','global','first_frequency','first_destination','first2_destination','oracle','oracle_load'],
        oracle='future-truth destination-bundle greedy; report LP dual upper bound where computed; never claim exact global optimality',
        load_rule='load-aware oracle caps each iteration/layer max owner assignment at1.10x baseline maximum',
        copy_rule='cold per-block full expert copies, no cross-block reuse; full6MiB per layer/expert; byte ROI and measured P2P transfer estimates',
        gates=dict(oracle_min=.35,causal_recovery_min=.50,causal_must_beat_global=True,
                   avoided_dc_bytes_per_copy_min=2.,load_preferred_max_increase=.10,load_warning=.20),
        bootstrap='2000 request/problem bootstrap; no layer/iteration pseudo-replication',
        no_methods_before_stage_a=True, no_epoch_measured_claim=True))
    (ROOT/'references').mkdir(exist_ok=True)
    target = ROOT/'references/authoritative_spec.md'
    if target.exists(): raise FileExistsError(target)
    target.write_bytes(SPEC.read_bytes())
    command = lambda a: subprocess.check_output(a, text=True)
    previous = json.loads((SOURCE/'results/environment.json').read_text())
    new_json(ROOT/'results/environment.json', dict(
        inherited=previous, source_commit=SOURCE_COMMIT,
        gpus=command(['nvidia-smi','--id='+MASK,'--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader']),
        topology=command(['nvidia-smi','topo','-m']), logical_physical=dict(enumerate(GPUS)),
        processes_read_only=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
        current_dinfer_sha=command(['git','-C',str(DINFER),'rev-parse','HEAD']).strip(),
        actual_ownership=[json.loads((SOURCE/f'raw/main_v2/ownership.rank{r}.json').read_text()) for r in range(4)],
        free_memory_reaudit='burner-free physical session audit follows', no_other_gpu_mutation=True))
    print('DESIGN_FROZEN',len(calibration),len(heldout),flush=True)


if __name__ == '__main__': main()
