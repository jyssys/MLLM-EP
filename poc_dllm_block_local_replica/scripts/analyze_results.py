"""Request-level paired bootstrap and explicit stage-promotion gate."""
import json
import numpy as np
import pandas as pd
from poc_dllm_block_local_replica.contract import *
from poc_dllm_block_local_replica.metrics import bootstrap_mean

METRICS=('baseline_amp','residual_amp','dc_reduction','oracle_recovery','byte_roi','peak_memory_gib',
         'max_rank_load_increase','stage_critical_load_increase','worst_stage_load_increase')


def main():
    df=pd.read_csv(ROOT/'results/simulation/request_policy.csv')
    costs=pd.read_csv(ROOT/'results/copy_cost/request_copy_roi.csv')
    df=df.merge(costs.drop(columns=['copy_bytes','byte_roi']),on=['request','policy','budget_gib'],validate='one_to_one')
    groups=dict(continuity_b32_all=df[df.block_length==32],
        calibration_b32_all=df[(df.block_length==32)&(df.split=='calibration')],
        heldout_b32_all=df[(df.block_length==32)&(df.split=='heldout')],
        heldout_b32_humaneval=df[(df.block_length==32)&(df.split=='heldout')&(df.domain=='humaneval')],
        heldout_b32_gsm8k=df[(df.block_length==32)&(df.split=='heldout')&(df.domain=='gsm8k')],
        heldout_b64_all=df[(df.block_length==64)&(df.split=='heldout')])
    stats={};pairs={};flat=[]
    extra=('copy_serial_expert_ms','copy_parallel_expert_est_ms','copy_serial_bulk_est_ms','copy_parallel_bulk_est_ms')
    for name,rows in groups.items():
        stats[name]={};pairs[name]={}
        for (policy,budget),r in rows.groupby(['policy','budget_gib']):
            key=f'{policy}@{budget:g}'
            values={m:bootstrap_mean(r[m].tolist()) for m in METRICS+extra}
            stats[name][key]=values
            flat.append(dict(cohort=name,policy=policy,budget_gib=budget,**{m:values[m]['mean'] for m in values}))
        for budget in BUDGETS:
            pivot=rows[rows.budget_gib==budget].pivot(index='request',columns='policy',values='dc_reduction')
            for candidate,control in [('first_destination','global'),('first2_destination','global'),
                                       ('first_destination','first_frequency'),('first2_destination','first_frequency')]:
                pairs[name][f'{candidate}-{control}@{budget:g}']=bootstrap_mean((pivot[candidate]-pivot[control]).tolist())
    # Only calibration chooses the descriptive candidate. This is NOT a
    # runtime operating point; no useful point can pass a failed copy gate.
    cal=stats['calibration_b32_all']
    candidates=[key for key in cal if key.startswith(('first_destination@','first2_destination@'))]
    chosen=max(candidates,key=lambda k:cal[k]['dc_reduction']['mean'])
    policy,budget=chosen.split('@');budget=float(budget)
    target=stats['heldout_b32_all'][chosen]
    oracle=stats['heldout_b32_all'][f'oracle@{budget:g}']
    contrast=pairs['heldout_b32_all'][f'{policy}-global@{budget:g}']
    upper=json.loads((ROOT/'results/copy_roi_upper_bound.json').read_text())
    gates=dict(oracle_headroom=oracle['dc_reduction']['mean']>=.35,
        causal_recovery=target['oracle_recovery']['mean']>=.50,
        causal_beats_static=contrast['ci95'][0]>0,
        copy_byte_roi=target['byte_roi']['mean']>=2.,
        equivalent_measured_timing_roi='UNAVAILABLE',
        preferred_compute_balance=(target['max_rank_load_increase']['mean']<=.10 and
                                   target['worst_stage_load_increase']['mean']<=.10),
        worst_stage_compute_warning=target['worst_stage_load_increase']['mean']>.20)
    # A failed central copy gate is not rescued with a hypothetical bytes->ms
    # conversion. Negative conclusion is scoped to this cold B1 regime.
    if not gates['copy_byte_roi'] or not gates['causal_beats_static']:
        verdict='NO-GO';promote=False
    elif all(gates[k] for k in ('oracle_headroom','causal_recovery','preferred_compute_balance')):
        verdict='CONDITIONAL';promote=True
    else:verdict='CONDITIONAL';promote=False
    result=dict(verdict=verdict,stage_b_promoted=promote,descriptive_candidate=chosen,
        descriptive_selection='maximum calibration pooled D+C reduction over frozen first/first2 destination and budgets; NOT a deployed operating point',
        gates=gates,stats=stats,paired_contrasts=pairs,copy_policy_independent_bound=upper,
        oracle_caveat='future-truth bundle greedy achievable reference; LP relaxation bound available at0.5/4GiB; not exact integer-optimal oracle',
        measured_runtime_gain=None,correctness='baseline revalidation PASS; runtime replica NOT_IMPLEMENTED because stageA gate',
        accuracy='runtime replica benchmark accuracy NOT_RUN; offline placement accounting cannot measure quality',
        final_scope='cold block-local exact BF16 replication, B1 source-SP4, native block32 primary/block64 secondary; persistent cross-block replicas and large batch not ruled out',
        next_action='Do not build block-local runtime. If revisiting, first characterize reuse across many blocks/requests or larger batch so expert copy can amortize; not a method claim.')
    destination=ROOT/'results/summary.json'
    # Result generation is append-only; an audited correction is stored in
    # a new version rather than silently replacing the first calculation.
    if destination.exists():destination=ROOT/'results/summary_v2.json'
    new_json(destination,result)
    if not (ROOT/'results/policy_summary.csv').exists():
        pd.DataFrame(flat).to_csv(ROOT/'results/policy_summary.csv',index=False)
    if (ROOT/'results/bootstrap_ci.json').exists():
        print('AUDITED_GATE_CORRECTION',destination,flush=True)
        return
    new_json(ROOT/'results/bootstrap_ci.json',dict(stats=stats,paired=pairs,unit='request/problem',repeats=2000))
    new_json(ROOT/'results/correctness.json',dict(physical_baseline='PASS: exact current/prior token,acceptance,NFE,router-ID/weight/buffer parity on two probes',
        structural_invariants='PASS: unchanged top8 path count; layer-aware tensors; same original one-shot denominator; causal profiling charged; per-source joint memory budget',
        replica_runtime='NOT_IMPLEMENTED_STAGE_A_NO_GO',final_token_parity='NOT_TESTED_FOR_REPLICA_RUNTIME',
        acceptance_parity='NOT_TESTED_FOR_REPLICA_RUNTIME',NFE_parity='NOT_TESTED_FOR_REPLICA_RUNTIME',
        replica_tensor_identity='NOT_TESTED_FOR_REPLICA_RUNTIME'))
    new_json(ROOT/'results/accuracy.json',dict(status='NOT_RUN_STAGE_A_NO_GO',gsm8k_exact_match=None,
        humaneval_deterministic_pass_at_1=None,accuracy_delta=None,
        reason='No replica runtime promoted; bounded prior outputs are not full benchmark completions; no fabricated accuracy parity.'))
    pd.DataFrame([dict(method=m,status='NOT_RUN_STAGE_A_NO_GO',D_C_ms=None,E2E_ms=None,speedup=None)
                  for m in ('Global replication','Block-local replication')]).to_csv(ROOT/'results/runtime_timing.csv',index=False)
    print('FINAL_GATE',verdict,chosen,gates,flush=True)


if __name__=='__main__':main()
