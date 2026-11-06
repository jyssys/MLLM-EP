"""Compute cost-fit bounds, not a validated serving or Safe Slack oracle."""
import csv
from .common import *
from .analyze import table,distribution
from ..native_loop_2d_oracle.dependency import Frontier


def action(row):
    samples=list(row['streams']['decode']['per_request_sum_ms'].values())
    if not samples or not row['passed']:return None
    cfg=json.loads((ROOT/'configs'/(row['name']+'.json')).read_text())
    frontier=cfg.get('native_frontier') or [cfg.get('native_partial_cut',cfg['depth']),0]
    return dict(name=row['name'],model=row['model'],remaining_loops=row['remaining_loops'],frontier=row.get('frontier'),
        structural_accounting=Frontier(cfg['length'],cfg['depth'],*frontier).accounting(row['model']),
        whole_suffix_best_observed_ms=min(samples),whole_suffix_largest_observed_ms=max(samples),
        measured_handoff_p50_ms=row['handoff']['p50_ms'],measured_handoff_max_ms=row['handoff']['maximum_ms'],
        actual_payload_bytes=row['counters']['prefill']['transport_bytes_sent']/row['count'],
        scope='Eight serial requests in one start. Best observed cost is not a certified lower bound. Maximum observed cost is not WCET.')


def main():
    initialize()
    primitive=json.loads((ROOT/'full_prompt_primitive_costs.json').read_text())
    quantum_path=ROOT/'equal_quantum_primitive_costs.json'
    quantum=json.loads(quantum_path.read_text()) if quantum_path.exists() else []
    actions=[a for row in primitive+quantum if (a:=action(row)) is not None]
    findings=[];decisions=[]
    for path in sorted((ROOT/'raw').glob('*/observer_qualification.json')):
        qualification=json.loads(path.read_text());name=path.parent.name
        if qualification.get('observer')!='graph':continue
        upper=path.parent/'compute_free_upper_envelope_gaps.csv'
        coverage=qualification.get('activity_coverage',[])
        if not upper.exists() or not coverage or not all(v['passed'] for v in coverage):continue
        gaps=list(csv.DictReader(upper.open()))
        gaps=[g for g in gaps if g['same_run_P_backlog']=='True']
        for a in actions:
            if a['model']!=qualification['model']:continue
            fits=[g for g in gaps if float(g['duration_ms'])>=a['whole_suffix_best_observed_ms']]
            findings.append(dict(name=name,model=a['model'],action=a['name'],
                instrumentation_qualified=qualification['qualified'],P_pressure_upper_envelopes=len(gaps),
                complete_suffix_compute_only_fit_count=len(fits),
                best_observed_complete_suffix_ms=a['whole_suffix_best_observed_ms'],
                largest_optimistic_idle_envelope_ms=max((float(g['duration_ms']) for g in gaps),default=None),
                evidence='CLAIRVOYANT_RELAXED_COST_FIT_BOUND_NOT_VALIDATED_SERVING',
                transfer_cost_in_admission='IGNORED_OPTIMISTICALLY',
                legal_frontier_timing='NOT_MEASURED_RELAXED',KV_workspace_admission='RELAXED',
                future_Decode_safety='UNVALIDATED',TTFT_speedup='NOT_PREDICTED',
                scope='All graph compute is removed from the trace. Handoff and frontier timing are relaxed equally. A positive fit is not a feasible or online-safe offload.'))
        full=[a for a in actions if a['model']==qualification['model'] and a['frontier'] is None]
        for gap in gaps:
            budget=float(gap['duration_ms'])
            choices=[a for a in full if a['whole_suffix_best_observed_ms']<=budget]
            selected=max(choices,key=lambda a:a['remaining_loops']) if choices else None
            decisions.append(dict(name=name,model=qualification['model'],opportunity_start=float(gap['start']),
                optimistic_budget_ms=budget,selected_loops=selected['remaining_loops'] if selected else 0,
                instrumentation_qualified=qualification['qualified'],
                scope='Offline clairvoyant relaxed upper-envelope fit. Not an actual policy decision, transfer, or accepted serving request.'))
    table(ROOT/'relaxed_opportunity_oracle.csv',findings)
    table(ROOT/'offline_budget_loop_decisions.csv',decisions)
    write(ROOT/'opportunity_oracle_status.json',dict(actions=actions,records=len(findings),
        validated_feasible_oracle=False,validation_error='NOT_MEASURED',
        Token_by_Loop_oracle='NOT_MEASURED_NO_NEW_VALIDATED_TILE_COST_AND_FRONTIER_TIMING',
        simulated_TTFT_E2E_goodput='NOT_PREDICTED',
        scope='Cost-fit diagnostics only. These results cannot establish online headroom or a speedup.'))
    print(json.dumps(dict(actions=len(actions),diagnostics=len(findings),offline_decisions=len(decisions))))


if __name__=='__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('relaxed_opportunity_cost_fit'):main()
