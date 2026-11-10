"""Keep diagnostic kernel measurements separate from qualified Oracle capacity."""
import math
from .common import *
from .serving_audit import csv_write


def main():
    configure()
    out=ROOT/'idle_slack_oracle'
    costs=json.loads((ROOT/'refreshed_primitive_groups.json').read_text())
    summary=[]
    requested=[]
    for model,depth,length,target,cuts in (
            ('ouro',4,4096,500,[1,2,3]),('huginn',32,2048,4000,[24,28,30,31])):
        path=out/f'oracle_decode_only_{model}_full128_calibration.json'
        diagnostic=json.loads(path.read_text())
        observed=diagnostic['roles']['decode']
        recurrent=[r for r in costs if r['model']==model and r['role']=='decode'
            and r['operation']=='recurrent' and r['tokens']==length and r['loops']==1]
        core=min(r['p50_ms'] for r in recurrent if not r['terminal'])
        terminal=min(r['p50_ms'] for r in recurrent if r['terminal'])
        extended=[]
        if model=='huginn':
            for cut in (16,21):
                qualification=ROOT/'raw'/f'earlier_huginn_loop{cut}_state8'/'qualification_with_frozen_reference.json'
                if qualification.exists() and json.loads(qualification.read_text()).get('passed'):
                    extended.append(cut)
        caps=384*(depth-min(cuts))
        extended_cap=384*(depth-min(cuts+extended))
        expanded_packing=[dict(tokens=p['tokens'],
            full_loop_equivalents=min(p['ready_free_quantum_count']//p['tiles_per_full_loop'],128*(depth-min(cuts+extended))),
            scope='One intrusive N128 trace and a ready-free median-cost model. Not qualified physical or Decode-safe capacity.')
            for p in observed['physical_packing_bounds'] if 'ready_free_quantum_count' in p]
        projected_requests=math.ceil(target/(depth-min(cuts+extended)))
        summary.append(dict(model=model,requested_requests=384,target_loops=target,total_prefill_loops=384*depth,
            target_fraction=target/(384*depth),historical_qualified_cuts=cuts,historical_structural_cap=caps,
            newly_N8_qualified_cuts=extended,extended_structural_cap=extended_cap,
            expanded_frontier_relaxed_N128_packing=expanded_packing,
            fastest_nonterminal_median_ms=core,fastest_terminal_median_ms=terminal,
            optimistic_median_cost_projection_seconds=(target*core+projected_requests*max(0.,terminal-core))/1000,
            projection_request_count=projected_requests,
            projection_scope='Conditional serial-median cost projection. Not a guaranteed minimum, physical capacity, or SLO-safe capacity. It excludes transfer, queueing, and launch gaps.',
            requested_clean_N384_kernel_busy='NOT_MEASURED',requested_clean_N384_physical_idle='NOT_MEASURED',
            requested_clean_N384_Idle_windows='NOT_MEASURED',
            qualified_physical_packing_oracle='NOT_MEASURED',decode_safe_oracle='NOT_MEASURED',
            completion_feasible_oracle='NOT_MEASURED',
            diagnostic_run=diagnostic['name'],diagnostic_arrival_seed=diagnostic['arrival_seed'],
            diagnostic_N128=observed,diagnostic_5_percent_gate=diagnostic['profiler_5_percent_gate'],
            diagnostic_interference=diagnostic['instrumentation_comparison'],
            target_verdict=('STRUCTURALLY_IMPOSSIBLE_WITH_HISTORICAL_CUTS. Extended cuts do not establish capacity.'
                if target>caps else 'NOT_ESTABLISHED. The target fits the cut-count ceiling only.'),
            principal_uncertainty='The full collector perturbs Decode tails. Both-stage captures also fail progress. Clean request-ready and transfer-constrained kernel capacity remains unqualified.'))
        for seed in (2026110911,2026110912,2026110913):
            requested.append(dict(model=model,arrival_seed=seed,count=128,
                physical_idle='NOT_MEASURED',decode_safe='NOT_MEASURED',completion_feasible='NOT_MEASURED',
                reason='The calibration collector failed the interference gate. Do not expand an unqualified collector to confirmation seeds.'))
    write(out/'qualified_capacity_status.json',summary)
    write(out/'requested_three_seed_status.json',requested)
    csv_write(out/'requested_three_seed_status.csv',requested)
    captures=[]
    for path in sorted((ROOT/'configs').glob('oracle_*.json')):
        cfg=json.loads(path.read_text())
        if not cfg.get('full_kernel_trace'):continue
        receipt=ROOT/'raw'/cfg['name']/'dispatcher_result.json'
        item=dict(name=cfg['name'],model=cfg['model'],configuration_sha256=digest(path),
            receipt=json.loads(receipt.read_text()) if receipt.exists() else dict(status='NOT_RUN'))
        derived=out/(cfg['name']+'.json')
        if derived.exists():
            v=json.loads(derived.read_text())
            item['profiler_5_percent_gate']=v.get('profiler_5_percent_gate')
            item['eligible_as_clean_native_capacity_evidence']=v.get('eligible_as_clean_native_capacity_evidence',False)
        captures.append(item)
    write(out/'capture_attempt_inventory.json',captures)
    print(json.dumps([dict(model=r['model'],historical_cap=r['historical_structural_cap'],
        extended_cap=r['extended_structural_cap'],projection_seconds=r['optimistic_median_cost_projection_seconds']) for r in summary]))


if __name__=='__main__':
    main()
