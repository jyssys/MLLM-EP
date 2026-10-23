"""Reproduce old plans and audit legal choices. No counterfactual latency claims."""
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from .common import ROOT, PRIOR, PARENT, REPO, SPEC, write, digest
from ..ouro_budget_joint_cg.cost_predictor import ServiceLookup, csv_file
from ..ouro_budget_joint_cg.optimizer import enumerate_candidates

def descriptors(data):
    result = defaultdict(list)
    for row in data['chunk_decisions']:
        s = row['state']
        result[s['id']].append((s['start'], row['chosen_C'], s['depth'], row['chosen_G'], row['phase']))
    return dict(result)

def pinned_inputs():
    names = ['analysis.md', 'online_policy.md', 'online_policy_contract.md', 'optimization.md',
        'phase_gates.md', 'cost_interaction.md', 'budget_action_trace.csv.gz',
        'G1_J1_actual_plan_equivalence.csv', 'G1_J1_CPU_overhead.csv', 'opportunity_frequency.csv',
        'budget_overruns.csv', 'budget_stall_relationship.md', 'fairness.md', 'predictor.md',
        'quality.md', 'quality_baseline_correct_to_wrong.csv', 'main_table.csv',
        'main_paired_pooled_CI.csv', 'request_events.csv.gz', 'source_family_manifest.json',
        'registration.json', 'registration.md', 'runtime_patch.diff', 'reproduce.md',
        'gpu_contention.md', 'pilot_training_selection.json']
    result = []
    for name in names:
        p = PRIOR / name
        committed = subprocess.check_output(['git', 'show', f'{PARENT}:{p.relative_to(REPO)}'], cwd=REPO)
        assert p.read_bytes() == committed, ('Historical evidence changed', name)
        result.append(dict(path=str(p.relative_to(REPO)), sha256=digest(p), bytes=p.stat().st_size))
    return result

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    inputs = pinned_inputs()
    lookup = ServiceLookup(json.loads((PRIOR / 'cost_model_artifacts/service_lookup_v1.json').read_text()))
    rows, candidates, equality, snapshots = [], [], [], []
    fixed_values = set()
    for folder in sorted((PRIOR / 'raw').glob('cb_p5_confirm_*')):
        marker = folder / 'completed.json'
        assert marker.exists(), folder
        provenance = json.loads(marker.read_text())
        for item in provenance['artifacts']:
            assert digest(folder / item['path']) == item['sha256'], item
        identity = json.loads((folder / 'identity.json').read_text())
        for trial in identity['job']['trials']:
            left = json.loads((folder / f'G1_T{trial}.json').read_text())
            right = json.loads((folder / f'J1_T{trial}.json').read_text())
            lp, rp = descriptors(left), descriptors(right)
            fixed_C = left['cell']['fixed_C']
            fixed_values.add(fixed_C)
            left_requests = {r['id']:r for r in left['requests']}
            for request in right['requests']:
                rid = request['id']
                equality.append(dict(run_id=folder.name, trial=trial, request_id=rid,
                    ordered_native_plan_equal=lp.get(rid)==rp.get(rid),
                    tokens_equal=left_requests[rid]['output_ids']==request['output_ids'],
                    input_tokens=request['input_tokens'], G_only_fixed_C=fixed_C,
                    J_decisions=len(rp.get(rid,[])), G_decisions=len(lp.get(rid,[])),
                    evidence='LIVE_MEASURED_HISTORICAL_NATIVE_SCHEDULE_DESCRIPTORS'))
            for index, decision in enumerate(right['optimizer_decisions']):
                state = decision['state']; B = decision['budget_ms']
                checksum = hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert checksum == decision['state_sha256']
                legal = enumerate_candidates(state, math.inf, lookup)
                feasible = enumerate_candidates(state, B, lookup)
                assert [c.__dict__ for c in feasible] == decision['candidates'], ('Replay mismatch', folder, trial, index)
                selected = decision['selected']
                fallback = not feasible
                choice = min(legal,key=lambda c:max(c.current_ms,c.opening_ms)) if fallback else max(
                    feasible,key=lambda c:(c.utility,c.complete_tokens_now,-c.residual_chunk_ms,c.G))
                assert choice.__dict__ == selected, ('Selected action mismatch',folder,trial,index)
                only = enumerate_candidates(state,B,lookup,'G_ONLY',fixed_C)
                only_fallback = enumerate_candidates(state,math.inf,lookup,'G_ONLY',fixed_C)
                only_choice = max(only,key=lambda c:(c.utility,c.complete_tokens_now,-c.residual_chunk_ms,c.G)) if only else min(
                    only_fallback,key=lambda c:max(c.current_ms,c.opening_ms))
                actual_cs = sorted({c.C for c in legal}); affordable_cs = sorted({c.C for c in feasible})
                fresh = state['kind']=='NEW_CHUNK'
                labels = []
                if not fresh: labels.append('no_fresh_boundary')
                if len(actual_cs)==1 and fresh:
                    labels.append('token_budget_limit' if state['token_budget']<min(64,state['prompt_remaining']) else 'same_clipped_span')
                if len(affordable_cs)<len(actual_cs): labels.append('cost_infeasible')
                if fallback: labels.append('fallback')
                if fresh and len(affordable_cs)>1 and choice.C==only_choice.C: labels.append('fixed_C_already_optimal')
                if choice.C==only_choice.C: labels.append('utility_prefers_same_C')
                scored = sorted(feasible,key=lambda c:c.utility,reverse=True)
                other = [c for c in scored if c.C!=choice.C]
                margin = choice.utility-max(c.utility for c in other) if other else None
                ties = [c for c in scored if abs(c.utility-choice.utility)<=1e-12]
                if len({c.C for c in ties})>1: labels.append('tie_or_hysteresis')
                # Packing is a shared restriction, not an independently established cause.
                labels.append('single_span_pack_limit_shared_context')
                primary_labels = list(labels)
                if len(affordable_cs)>1:
                    # Removing some G alternatives does not prove C was blocked.
                    primary_labels = [s for s in labels if s!='cost_infeasible']
                primary = next((s for s in ('no_fresh_boundary','same_clipped_span','token_budget_limit',
                    'fallback','cost_infeasible','tie_or_hysteresis','fixed_C_already_optimal',
                    'utility_prefers_same_C') if s in primary_labels),'unknown')
                meta = dict(run_id=folder.name,trial=trial,decision=index,GPU=identity['physical_GPU'],
                    P=right['P'],N=right['configured_N'],budget_ms=B,request_id=state['id'],
                    phase=state['phase'],fresh=fresh,start=state['start'],depth=state['depth'],
                    prompt_remaining=state['prompt_remaining'],token_budget=state['token_budget'],
                    legal_distinct_C=len(actual_cs),feasible_distinct_C=len(affordable_cs),
                    legal_C=json.dumps(actual_cs),feasible_C=json.dumps(affordable_cs),
                    chosen_C=choice.C,chosen_G=choice.G,G_only_C=only_choice.C,G_only_G=only_choice.G,
                    C_differs_from_G_only=choice.C!=only_choice.C,known_shape=choice.known,
                    score_margin_other_C=margin,top_C_ties=len({c.C for c in ties}),
                    primary_reason=primary,all_labels=json.dumps(labels),
                    source=str((folder/f'J1_T{trial}.json').relative_to(REPO)),
                    evidence='HISTORICAL_OBSERVED_STATE_WITH_OFFLINE_SELECTION_REPLAY')
                rows.append(meta)
                snapshots.append(dict(meta=meta,state=state,candidates=[c.__dict__ for c in legal],
                    historical_role='Consumed FINAL_ONLINE: development diagnostics only, not a fresh blind test'))
                for candidate in legal:
                    candidates.append(dict(run_id=folder.name,trial=trial,decision=index,
                        fresh=fresh,budget_ms=B,feasible=max(candidate.current_ms,candidate.opening_ms)<=B,
                        selected=candidate==choice,**candidate.__dict__))
    assert len(equality)==144 and all(r['ordered_native_plan_equal'] and r['tokens_equal'] for r in equality)
    csv_file(ROOT/'prior_plan_reproduction.csv',equality)
    csv_file(ROOT/'decision_opportunity_funnel.csv',rows)
    csv_file(ROOT/'candidate_score_audit.csv.gz',candidates)
    with gzip.open(ROOT/'historical_snapshots.jsonl.gz','wt') as stream:
        for row in snapshots: stream.write(json.dumps(row)+'\n')
    groups=defaultdict(list)
    for row in rows: groups[(row['P'],row['budget_ms'])].append(row)
    summary=[]
    for (P,B),values in groups.items():
        fresh=[r for r in values if r['fresh']]
        summary.append(dict(P=P,B_ms=B,decisions=len(values),fresh=len(fresh),
            multiple_legal_C=sum(r['legal_distinct_C']>1 for r in fresh),
            multiple_feasible_C=sum(r['feasible_distinct_C']>1 for r in fresh),
            different_C=sum(r['C_differs_from_G_only'] for r in fresh),
            continuation=len(values)-len(fresh),unknown_selected=sum(not r['known_shape'] for r in values),
            primary_reasons=json.dumps(dict(Counter(r['primary_reason'] for r in values)))))
    csv_file(ROOT/'eligibility_collapse.csv',summary)
    write(ROOT/'phase_a_provenance.json',dict(parent_commit=PARENT,spec_sha256=digest(SPEC),inputs=inputs,
        quiet_source_sha256={str(p.relative_to(REPO)):digest(p) for p in sorted((PRIOR/'raw').glob('cb_p5_confirm_*/J1_T*.json'))},
        fixed_C_values=sorted(fixed_values),plan_pairs=len(equality),selector_decisions=len(rows),
        replay='Exact candidate and selected-action equality under the frozen old selector and lookup',
        missing='Rejected admission states have no exhaustive snapshot log. Quiet lane has native descriptors, not heavy GPU span timings.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
