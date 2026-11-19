"""Compare a restricted cross-cut oracle with discovery-frozen fixed actions."""
import collections
import json
import math
import time
from .common import ROOT,write,digest
from .axis_oracle import exhaustive,objective
from .replay import Cost
from .resampling import ratio_interval


def main():
    modelpath=ROOT/'configs/frozen_action_service_costs_v2.json'
    if not modelpath.exists():
        return
    models=json.loads(modelpath.read_text())['model']['models']
    table={r['method']+str(r['layers']):Cost(**r['cost_seconds']) for r in models
        if r['batch']==1 and r['output']==16}
    assert len(table)==8
    frozen=json.loads((ROOT/'configs/frozen_best_fixed_axes_v1.json').read_text())
    chosen={r['QPS']:{a:v['axis']+str(v['layers']) for a,v in r['selection'].items()}
        for r in frozen['selection']['selections']}
    stats=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    starts=[r for r in stats['starts'] if r['stage']=='heldout_confirmation' and
        r['output']==16 and r['actual_batch']==1]
    groups=collections.defaultdict(dict)
    for r in starts:
        key=r['QPS'],r['seed'],r['count'],r['source_window']
        groups[key][r['method']+str(r['layers'])]=r
    output=[]
    for (q,seed,n,window),actual in sorted(groups.items()):
        if not set(table)<=set(actual) or n%18:
            continue
        arrivals=actual['Layer2']['arrival_offsets']
        assert all(r['arrival_offsets']==arrivals for r in actual.values())
        oracle=exhaustive(arrivals,table,1,window,16,18)
        validations={}
        for action in table:
            r=actual[action];pred=oracle['fixed'][action]
            ttft=abs(pred['TTFT_p95_ms']/r['metric_summary']['TTFT_p95_ms']-1)
            raw=abs(pred['output_tokens_s']/r['metric_summary']['output_tokens_s']-1)
            wait=abs(pred['Ready_D_wait_mean_ms']-r['Ready_D_mean_ms'])
            joint=abs(pred['joint_attainment']-r['common_Balanced']['Joint_attainment'])
            good=r['common_Balanced']['Output_token_SLO_goodput']
            gp=(abs(pred['qualified_output_tokens_s']/good-1) if good else None)
            validations[action]=dict(TTFT_relative_error=ttft,raw_tokens_relative_error=raw,
                Ready_D_absolute_error_ms=wait,Joint_attainment_absolute_error=joint,
                qualified_token_relative_error=gp,
                passed=ttft<=.10 and raw<=.05 and wait<=max(10.,.20*r['Ready_D_mean_ms']) and
                    joint<=.05 and (gp<=.10 if good else pred['qualified_output_tokens_s']==0))
        selected={a:oracle['fixed'][v] for a,v in chosen[q].items()}
        output.append(dict(QPS=q,seed=seed,count=n,batch=1,window=window,
            frozen_fixed_actions=chosen[q],predicted_frozen_fixed=selected,
            measured_frozen_fixed={a:actual[v]['metric_summary'] for a,v in chosen[q].items()},
            measured_frozen_fixed_goodput={a:actual[v]['common_Balanced']['Output_token_SLO_goodput']
                for a,v in chosen[q].items()},
            Native_measured=(dict(actual['Native0']['metric_summary'],
                common_Balanced=actual['Native0']['common_Balanced']) if 'Native0' in actual else None),
            oracle=oracle,validation=validations,
            per_trace_best_fixed_action=oracle['best_fixed'],
            all_fixed_actions_validated=all(v['passed'] for v in validations.values()),
            evidence='OFFLINE_ORACLE',
            scope='Eight fixed actions: Layer/Token Last2/4/8/12. Each18-request block preserves exact rows within its chosen cut. The total migrated work can change with cut selection. Uniform credit reservations and cohort state update. Mixed-axis execution is not GPU-qualified. This oracle is not a global upper bound or measured improvement.'))
    grouped=[]
    for q in sorted({r['QPS'] for r in output}):
        cells=[r for r in output if r['QPS']==q]
        grouped.append(dict(QPS=q,starts=len(cells),seeds=[r['seed'] for r in cells],
            predicted_goodput_oracle_over_frozen_fixed=ratio_interval(
                [r['predicted_frozen_fixed']['Overall']['qualified_output_tokens_s'] for r in cells],
                [r['oracle']['oracle']['qualified_output_tokens_s'] for r in cells]),
            predicted_goodput_oracle_over_per_trace_best_fixed=ratio_interval(
                [r['oracle']['fixed'][r['oracle']['best_fixed']]['qualified_output_tokens_s'] for r in cells],
                [r['oracle']['oracle']['qualified_output_tokens_s'] for r in cells]),
            all_starts_all_actions_validated=all(r['all_fixed_actions_validated'] for r in cells),
            scope='Independent arrival starts are the bootstrap unit. Model uncertainty is not included in this interval. A validation failure makes modeled headroom Diagnostic.'))
    write(ROOT/'analysis/cross_cut_restricted_oracle.json',dict(time=time.time(),
        model_SHA256=digest(modelpath),fixed_selection_SHA256=digest(ROOT/'configs/frozen_best_fixed_axes_v1.json'),
        results=output,aggregate=grouped,evidence='OFFLINE_ORACLE',
        scope='The discovery-frozen fixed action is the deployable comparison. Per-trace best fixed remains an oracle diagnostic. Native is a measured external reference. No online axis controller, sustainable capacity or Decode5% tail guarantee is inferred.'))
    print('CROSS_CUT_RESTRICTED_ORACLE',len(output),len(grouped),flush=True)


if __name__=='__main__':
    main()
