"""Evaluate the finer exact-work search without claiming a global bound."""
import collections
import json
import math
import time
from .common import ROOT,write,digest,DEADLINE
from .replay import Cost
from .cohort_axis_oracle import beam,rule
from .resampling import ratio_interval


def main():
    protocol=ROOT/'state/fine_cohort_oracle_protocol.json'
    assert protocol.exists(),'Freeze the search protocol before inspecting this search result'
    stats=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    confirmed={(r['stage'],r['layers'],r['output'],r['QPS'],r['count'],r['batch'],r['window'])
        for r in stats['comparisons'] if r['reference']=='Token' and r['candidate']=='Layer' and r['starts']>=3}
    kappa=json.loads((ROOT/'configs/frozen_single_pressure_rule_v1.json').read_text())['rule']['selected_kappa']
    results=[];budget_stops=[]
    for version in (2,3,4):
        source=ROOT/('analysis/frozen_model_validation_and_oracle'+(f'_v{version}' if version>=3 else '')+'.json')
        if not source.exists():continue
        modelpath=ROOT/f'configs/frozen_action_service_costs_v{version}.json'
        models=json.loads(modelpath.read_text())['model']['models']
        lookup={(r['method'],r['layers'],r['output'],r['batch']):Cost(**r['cost_seconds']) for r in models}
        tests=json.loads(source.read_text())['results']
        for case in tests:
            key=case['stage'],case['layers'],case['output'],case['QPS'],case['count'],case['batch'],case['window']
            if key not in confirmed or case['stage']=='source_window_control':continue
            if time.time()>DEADLINE-1200:
                budget_stops.append(dict(version=version,key=key,seed=case['seed'],status='NOT_RUN'))
                continue
            stage,m,o,q,n,b,w=key
            table={a:lookup[a,m,o,b] for a in ('Layer','Token')}
            cfg=json.loads((ROOT/'configs'/(case['names']['Layer']+'.json')).read_text())
            coarse=case['oracle'];expanded=[]
            for a in coarse['oracle_sequence']:expanded.extend([a]*(coarse['block']//b))
            small=beam(cfg['arrival_offsets'],table,m,b,w,o,width=8,seeds=(expanded,))
            large=beam(cfg['arrival_offsets'],table,m,b,w,o,width=32,seeds=(expanded,small['oracle_sequence']))
            policies={kind:rule(cfg['arrival_offsets'],table,m,kappa,b,w,o,kind) for kind in ('one','two')}
            results.append(dict(model_version=version,stage=stage,layers=m,output=o,QPS=q,count=n,
                batch=b,window=w,seed=case['seed'],names=case['names'],coarse=coarse,
                beam8=small,beam32=large,matched_cohort_rules=policies,
                fixed_trace_validation=case['fixed_trace_validation'],
                qualified_for_counterfactual_interpretation=case['qualified_for_counterfactual_interpretation'],
                evidence='OFFLINE_ORACLE',
                scope='The large search includes the smaller-search winner and both fixed axes. One axis per actual-size cohort prevents group fragmentation. Exact aggregate GEMM work closes at the finite trace end. Passing fixed traces does not validate new mixed-axis GPU ownership.'))
            print('FINE_COHORT_ORACLE',version,stage,m,o,q,b,case['seed'],
                large['oracle']['qualified_output_tokens_s'],flush=True)
    grouped=collections.defaultdict(list)
    for r in results:
        grouped[r['model_version'],r['stage'],r['layers'],r['output'],r['QPS'],r['batch']].append(r)
    aggregate=[]
    for key,rows in sorted(grouped.items()):
        reference=[r['beam32']['fixed'][r['beam32']['best_fixed']]['qualified_output_tokens_s'] for r in rows]
        aggregate.append(dict(model_version=key[0],stage=key[1],layers=key[2],output=key[3],QPS=key[4],batch=key[5],
            starts=len(rows),seeds=[r['seed'] for r in rows],
            predicted_goodput_ratios={kind:ratio_interval(reference,
                [r['beam32']['oracle']['qualified_output_tokens_s'] if kind=='oracle' else
                 r['matched_cohort_rules'][kind]['metrics']['qualified_output_tokens_s'] for r in rows])
                 for kind in ('oracle','one','two')},
            all_fixed_validation_passed=all(r['qualified_for_counterfactual_interpretation'] for r in rows),
            beam32_over_beam8=ratio_interval([r['beam8']['oracle']['qualified_output_tokens_s'] for r in rows],
                [r['beam32']['oracle']['qualified_output_tokens_s'] for r in rows]),
            scope='Reference is per-trace best fixed axis inside the same calibrated model. CI resamples starts, not model uncertainty. A finite beam result is a feasible-search value, not a global upper bound.'))
    write(ROOT/'analysis/fine_cohort_axis_oracle.json',dict(time=time.time(),results=results,aggregate=aggregate,
        budget_stops=budget_stops,protocol_SHA256=digest(protocol),evidence='OFFLINE_ORACLE',
        scope='A stricter search-resolution check than whole18/36-request action blocks. No actual online dynamic speedup, physical GPU slack, Native-relative tail guarantee or sustainable capacity is inferred.'))
    print('FINE_COHORT_ORACLE_COMPLETE',len(results),len(budget_stops),flush=True)


if __name__=='__main__':
    main()
