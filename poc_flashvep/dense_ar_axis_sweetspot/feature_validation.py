"""Validate frozen rules on recorded pre-P host state, without a new scheduler."""
import collections
import json
import statistics
import time
from .common import ROOT,write,digest
from .historical_audit import records
from .replay import Cost
from .axis_oracle import demand_scores,cycle_boundary
from .observable_features import pressure,scores


def main(version=None):
    if version is None:
        for v in (2,3,4):
            if (ROOT/f'configs/frozen_action_service_costs_v{v}.json').exists():main(v)
        return
    path=ROOT/f'configs/frozen_action_service_costs_v{version}.json'
    if not path.exists():
        print('RECORDED_FEATURE_VALIDATION_NOT_READY',flush=True)
        return
    receipt=ROOT/f'state/service_model_v{version}_frozen.json'
    if not receipt.exists():return
    assert json.loads(receipt.read_text())['SHA256']==digest(path)
    model=json.loads(path.read_text())['model']
    costs={(r['method'],r['layers'],r['output'],r['batch']):Cost(**r['cost_seconds'])
        for r in model['models']}
    kappa=json.loads((ROOT/'configs/frozen_single_pressure_rule_v1.json').read_text())['rule']['selected_kappa']
    data=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    features=[]
    for run in data['starts']:
        if version>=3 and run['stage']!='batch4_knob_confirmation':continue
        axis,m,o,b=run['method'],run['layers'],run['output'],run['actual_batch']
        if axis not in ('Layer','Token') or any((a,m,o,b) not in costs for a in ('Layer','Token')):
            continue
        table={a:costs[a,m,o,b] for a in ('Layer','Token')}
        rows=records(ROOT/'raw'/run['name']/'online_decisions.jsonl')
        samples=[]
        for index,decision in enumerate(rows):
            observed=pressure(decision,table[axis],table['Layer'],o,b)
            one=scores(observed['pressure_ratio'],table,kappa,b)
            unfinished=1+observed['P_pending']+observed['D_ready_requests']+observed['active_D_request_count']
            two=demand_scores(table,b,unfinished)
            assert decision['observation']['group_size']==1
            samples.append(dict(request_index=index,time=decision['time'],**observed,
                unfinished=unfinished,one_scores=one,
                one_axis=min(one,key=lambda a:(one[a],a)),
                two_axis=min(two['candidates'],key=lambda a:(two['candidates'][a]['score'],a)),
                demand_result=two))
        assert len(samples)==run['count']
        features.append(dict(name=run['name'],stage=run['stage'],method=axis,layers=m,
            output=o,QPS=run['QPS'],batch=b,count=run['count'],window=run['source_window'],seed=run['seed'],
            pressure_ratio_quantiles=dict(min=min(r['pressure_ratio'] for r in samples),
                median=statistics.median(r['pressure_ratio'] for r in samples),
                max=max(r['pressure_ratio'] for r in samples)),
            one_votes=dict(collections.Counter(r['one_axis'] for r in samples)),
            two_votes=dict(collections.Counter(r['two_axis'] for r in samples)),
            boundary=cycle_boundary(table,b),samples=samples,
            scope='Recorded host state before each measured P submission. No future arrival or completion is read. These shadow votes do not change the actual fixed-axis run.'))
    indexed={r['name']:r for r in features}
    evaluations=[]
    stages=('heldout_confirmation','decode_batch_counterfactual','output_residency_counterfactual','new_QPS_confirmation','batch4_knob_confirmation')
    if version>=3:stages=('batch4_knob_confirmation',)
    for comp in data['comparisons']:
        if comp['reference']!='Token' or comp['candidate']!='Layer' or comp['stage'] not in stages or comp['starts']<3:
            continue
        ci=comp['metrics']['TTFT_p95_ms']['ratio_CI95']
        ratio=comp['metrics']['TTFT_p95_ms']['ratio']
        winner='Layer' if ci and ci[1]<1 else 'Token' if ci and ci[0]>1 else 'UNRESOLVED'
        gp=comp['common_Balanced_goodput']
        gp_values={'Layer':gp['candidate_mean'],'Token':gp['reference_mean']}
        gp_CI=gp['difference_CI95']
        gp_winner='Layer' if gp_CI and gp_CI[0]>0 else 'Token' if gp_CI and gp_CI[1]<0 else 'UNRESOLVED'
        selected=[r for r in features if r['stage']==comp['stage'] and r['layers']==comp['layers'] and
            r['output']==comp['output'] and r['QPS']==comp['QPS'] and r['batch']==comp['batch'] and
            r['count']==comp['count'] and r['window']==comp['window'] and r['seed'] in comp['seeds']]
        assert len(selected)==2*comp['starts']
        votes={rule:sum((collections.Counter(r[rule+'_votes']) for r in selected),collections.Counter())
            for rule in ('one','two')}
        predicted={rule:min(('Layer','Token'),key=lambda a:(-votes[rule][a],a)) for rule in votes}
        table={a:costs[a,comp['layers'],comp['output'],comp['batch']] for a in ('Layer','Token')}
        cycle=demand_scores(table,comp['batch'],1000000)['candidates']
        baselines={'AlwaysLayer':'Layer','AlwaysToken':'Token',
            'StaticMetadataCycle':min(cycle,key=lambda a:(cycle[a]['score'],a))}
        evaluations.append(dict(stage=comp['stage'],layers=comp['layers'],output=comp['output'],
            QPS=comp['QPS'],batch=comp['batch'],starts=comp['starts'],seeds=comp['seeds'],
            measured_Layer_over_Token_TTFT_p95=ratio,CI95=ci,confidence_label=winner,
            measured_qualified_tokens_s=gp_values,confidence_goodput_label=gp_winner,
            goodput_Layer_minus_Token_CI95=gp_CI,
            majority_votes={r:dict(v) for r,v in votes.items()},majority_fixed_axis_forecast=predicted,
            correct={r:(a==winner if winner!='UNRESOLVED' else None) for r,a in predicted.items()},
            fixed_axis_TTFT_regret={r:(max(0.,ratio-1) if a=='Layer' else max(0.,1/ratio-1))
                for r,a in predicted.items()},
            fixed_axis_goodput_regret={r:((max(gp_values.values())-gp_values[a])/max(gp_values.values())
                if max(gp_values.values())>0 else None) for r,a in predicted.items()},
            baseline_forecasts=baselines,
            baseline_correct={r:(a==winner if winner!='UNRESOLVED' else None) for r,a in baselines.items()},
            baseline_TTFT_regret={r:(max(0.,ratio-1) if a=='Layer' else max(0.,1/ratio-1))
                for r,a in baselines.items()},
            baseline_goodput_regret={r:((max(gp_values.values())-gp_values[a])/max(gp_values.values())
                if max(gp_values.values())>0 else None) for r,a in baselines.items()},
            scope='One cell-level validation unit, not each request or token. Majority votes are a static-axis classification diagnostic. Regret compares that fixed-axis forecast against measured paired cell means. It is not the performance of switching actions on the recorded fixed-state trace.'))
    filename='recorded_observable_knob_validation'+(f'_v{version}' if version>=3 else '')+'.json'
    write(ROOT/'analysis'/filename,dict(time=time.time(),
        model_version=version,cost_model_SHA256=digest(path),frozen_kappa=kappa,recorded_features=features,
        heldout_cell_evaluations=evaluations,
        evidence='TRACE_ANALYSIS',
        scope='The one-scalar rule uses a composite P/D pressure. The two-scalar rule uses calibrated Decode demand and total unfinished work. Several raw sensors can contribute to a scalar. Recorded states are axis-dependent. State-updating replay is reported separately.'))
    print('RECORDED_FEATURE_RUNS',version,len(features),'HELDOUT_CELLS',len(evaluations),flush=True)


if __name__=='__main__':
    main()
