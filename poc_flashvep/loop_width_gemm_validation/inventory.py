"""Audit retained cells, observer coverage, and actual quantum aliases."""
from collections import defaultdict,Counter
from .common import *


def main():
    initialize();cells=[];qualification=[];failures=[];core=[];aliases=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():continue
        path=ROOT/'configs'/(folder.name+'.json');receipt=folder/'dispatcher_result.json'
        if not path.exists() or not receipt.exists():continue
        cfg=json.loads(path.read_text());result=json.loads(receipt.read_text())
        cells.append(dict(name=folder.name,model=cfg['model'],complete=result.get('complete'),
            configuration_SHA256=digest(path),phase=cfg.get('phase'),gpus=cfg['gpus']))
        if not result.get('complete'):
            failures.append(dict(name=folder.name,receipt=result,measured_window_exists=(folder/'measurement_begin.json').exists(),
                scope='A startup observer failure is not a measured model-performance failure. The complete log remains in the evidence bundle.'))
        q=folder/'qualification_with_frozen_reference.json'
        if q.exists():
            v=json.loads(q.read_text());qualification.append(dict(name=folder.name,model=cfg['model'],count=cfg['count'],
                passed=v.get('campaign_gate_passed',False),frozen_output=v['frozen_output']['passed'],
                frozen_full_state=v['frozen_full_state']['passed'],quantum_required=cfg.get('require_realized_quantum_contract',False)))
        if cfg.get('observe_core_graph_instance') and result.get('complete'):
            expected_seeds=set(range(cfg['seed'],cfg['seed']+cfg['count']))
            for p in folder.glob('core_graph_execution_*.json'):
                value=json.loads(p.read_text());depths=defaultdict(list)
                for row in value['rows']:
                    if row['seed'] in expected_seeds:depths[row['seed']].append(row['actual_depth'])
                frontier=cfg.get('native_frontier');cut=cfg.get('native_partial_cut')
                c=cfg['depth'] if cut is None else cut
                expected=list(range(cfg['depth'])) if value['role']=='prefill' and frontier else (
                    list(range(c)) if value['role']=='prefill' else list(range(frontier[0],cfg['depth'])) if frontier else list(range(c,cfg['depth'])))
                passed=all(sorted(depths.get(seed,[]))==expected for seed in expected_seeds)
                core.append(dict(name=folder.name,role=value['role'],expected_depths=expected,
                    requests=cfg['count'],observed_calls=sum(map(len,depths.values())),passed=passed,
                    scope='Measured graph-call coverage. Not a hardware kernel count.'))
        marker=folder/'measurement_begin.json'
        if cfg.get('calibration_serial') and marker.exists():
            begin=json.loads(marker.read_text())['time'];spans=[]
            for p in folder.glob('atomic_spans_*.json'):
                value=json.loads(p.read_text())
                if value['role']=='decode':spans += [r for r in value['prompt_spans'] if r['time']>=begin and 0<=r['seed']-cfg['seed']<cfg['count']]
            largest=max((r['loop_count'] for r in spans),default=0);requested=cfg.get('D_prefill_loop_cap') or cfg['depth']
            if largest>requested:aliases.append(dict(name=folder.name,requested_loop_cap=requested,
                observed_largest_loop_group=largest,Prelude_in_cap=cfg.get('D_prefill_cap_includes_prelude',False),
                scope='Retained grouped-throughput calibration. Not evidence of small execution quanta.'))
    write(ROOT/'analysis/campaign_inventory.json',dict(cells=cells,state_qualification=qualification,
        qualified_test_requests=sum(r['count'] for r in qualification if r['passed']),
        all_recorded_qualifications_passed=all(r['passed'] for r in qualification),
        retained_failed_cells=failures,packed_core_coverage=core,
        all_packed_core_coverage_passed=all(r['passed'] for r in core),nominal_quantum_aliases=aliases,
        source_hashes={str(p.relative_to(REPO)):digest(p) for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py','.cc')},
        scope='Qualification counts include repeated prompts and multiple implementation controls. They are not independent serving requests or a generalized model-accuracy benchmark.'))
    print('STATE_REQUESTS',sum(r['count'] for r in qualification),'ALL_PASS',all(r['passed'] for r in qualification),
        'CORE_COVERAGE',all(r['passed'] for r in core),'RETAINED_FAILURES',len(failures),'ALIASES',len(aliases))


if __name__=='__main__':main()
