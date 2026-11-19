"""Test the final structural feedback model on untouched confirmation seeds."""
import collections
import functools
import json
import statistics
import time
from .common import ROOT,write,digest
from .replay import Cost,summary
from .control_replay import simulate
from .cohort_axis_oracle import beam,rule
from .resampling import ratio_interval
from .measurement_export import quantile


def main():
    path=ROOT/'configs/frozen_control_feedback_replay_inputs_v1.json'
    receipt=ROOT/'state/control_feedback_replay_inputs_frozen.json'
    if not path.exists() or not receipt.exists():
        print('CONTROL_REPLAY_INPUTS_NOT_QUALIFIED',flush=True);return
    assert digest(path)==json.loads(receipt.read_text())['SHA256']
    inputs=json.loads(path.read_text());table={a:Cost(**r) for a,r in inputs['costs'].items()}
    stats=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    tests=[r for r in stats['starts'] if r['stage']=='batch4_knob_confirmation' and r['method'] in table]
    results=[];pairs=collections.defaultdict(dict)
    for run in tests:
        simulator=functools.partial(simulate,output=run['output'],idle_RTT=inputs['idle_RTT_seconds'])
        rows=simulator(run['arrival_offsets'],[table[run['method']]]*run['count'],run['actual_batch'],run['source_window'])
        predicted=summary(rows,run['output']);actual=run['metric_summary'];actual_gp=run['common_Balanced']['Output_token_SLO_goodput']
        errors=dict(TTFT_p95_relative=abs(predicted['TTFT_p95_ms']/actual['TTFT_p95_ms']-1),
            output_tokens_s_relative=abs(predicted['output_tokens_s']/actual['output_tokens_s']-1),
            Ready_D_mean_absolute_ms=abs(predicted['Ready_D_wait_mean_ms']-run['Ready_D_mean_ms']),
            Joint_attainment_absolute=abs(predicted['joint_attainment']-run['common_Balanced']['Joint_attainment']),
            qualified_tokens_relative=abs(predicted['qualified_output_tokens_s']/actual_gp-1) if actual_gp else None)
        timing=errors['TTFT_p95_relative']<=.10 and errors['output_tokens_s_relative']<=.05 and errors['Ready_D_mean_absolute_ms']<=max(10.,.2*run['Ready_D_mean_ms'])
        slo=errors['Joint_attainment_absolute']<=.05 and (errors['qualified_tokens_relative']<=.10 if actual_gp else predicted['qualified_output_tokens_s']==0)
        feedback=json.loads((ROOT/'analysis'/(run['name']+'_reservation_feedback.json')).read_text())
        observed=[r['reservation_RTT_ms'] for r in feedback['requests']]
        signed=[1000*r['reservation_wait']-a for r,a in zip(rows,observed)]
        item=dict(name=run['name'],method=run['method'],QPS=run['QPS'],seed=run['seed'],
            predicted=predicted,actual=dict(actual,Ready_D_mean_ms=run['Ready_D_mean_ms'],
                qualified_tokens_s=actual_gp,Joint_attainment=run['common_Balanced']['Joint_attainment']),
            errors=errors,timing_passed=timing,SLO_passed=slo,
            ACK_error=dict(MAE_ms=statistics.mean(abs(v) for v in signed),
                p95_absolute_ms=quantile([abs(v) for v in signed],.95),
                signed_mean_ms=statistics.mean(signed),underprediction_frequency=sum(v<0 for v in signed)/len(signed)),
            evidence='OFFLINE_REPLAY',
            scope='Structural host-step replay. All costs and the idle RTT were frozen before measured arrivals. No actual future finish enters the prediction. Preparation is approximated as one blocked host step per member. Async admission/front-matter and within-step detail remain uninstrumented. Failed gates make counterfactual gains Diagnostic.')
        results.append(item);pairs[run['QPS'],run['seed']][run['method']]=(run,item)
    searches=[]
    kappa=json.loads((ROOT/'configs/frozen_single_pressure_rule_v1.json').read_text())['rule']['selected_kappa']
    for (q,seed),pair in sorted(pairs.items()):
        if set(pair)!=set(('Layer','Token')):continue
        run=pair['Layer'][0];passed=all(item['timing_passed'] and item['SLO_passed'] for _,item in pair.values())
        simulator=functools.partial(simulate,output=run['output'],idle_RTT=inputs['idle_RTT_seconds'])
        args=run['arrival_offsets'],table,8,4,run['source_window'],run['output']
        # Keep unsuccessful fixed models visible without spending the final
        # publication reserve on an unqualified mixed-policy search.
        if not passed:
            searches.append(dict(QPS=q,seed=seed,status='NOT_RUN_FAILED_FIXED_TRACE_VALIDATION',
                evidence='DIAGNOSTIC'));continue
        small=beam(*args,width=8,simulator=simulator)
        large=beam(*args,width=32,seeds=(small['oracle_sequence'],),simulator=simulator)
        policies={kind:rule(run['arrival_offsets'],table,8,kappa,4,run['source_window'],run['output'],kind,
            simulator=simulator) for kind in ('one','two')}
        searches.append(dict(QPS=q,seed=seed,status='FIXED_TRACE_GATES_PASS',beam8=small,beam32=large,
            rules=policies,evidence='OFFLINE_ORACLE',
            scope='Conditional modeled headroom only. Queues, control replies, credits and cohort state update after each action. Mixed-axis GPU correctness is NOT_RUN. Finite beam is not a global bound.'))
    aggregate=[]
    for q in sorted({r['QPS'] for r in searches}):
        selected=[r for r in searches if r['QPS']==q and 'beam32' in r]
        if selected:
            aggregate.append(dict(QPS=q,validated_starts=len(selected),seeds=[r['seed'] for r in selected],
                oracle_over_per_trace_fixed=ratio_interval(
                    [r['beam32']['fixed'][r['beam32']['best_fixed']]['qualified_output_tokens_s'] for r in selected],
                    [r['beam32']['oracle']['qualified_output_tokens_s'] for r in selected]),
                scope='Conditional subset after fixed-trace validation. Do not treat missing failed starts as successes or report this subset as an unconditional campaign result.'))
    write(ROOT/'analysis/control_boundary_replay_validation.json',dict(time=time.time(),results=results,
        searches=searches,aggregate=aggregate,inputs_SHA256=digest(path),evidence='OFFLINE_REPLAY',
        scope='Final bounded model check. No parameter refit follows a validation failure. All mean-gap models and fixed measured baselines remain separate. No actual online Dynamic gain is inferred.'))
    print('CONTROL_BOUNDARY_REPLAY_VALIDATION',len(results),sum(r['timing_passed'] and r['SLO_passed'] for r in results),flush=True)


if __name__=='__main__':
    main()
