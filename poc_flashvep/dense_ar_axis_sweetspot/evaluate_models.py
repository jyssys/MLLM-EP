"""Validate frozen costs and score restricted replay headroom on new traces."""
import collections
import json
import math
import statistics
import time
from .common import ROOT,write,digest
from .axis_oracle import predicted,exhaustive,pressure_rule,demand_rule,demand_scores
from .replay import Cost


def main(version=None):
    if version is None:
        for v in (2,3,4):
            if (ROOT/f'configs/frozen_action_service_costs_v{v}.json').exists():main(v)
        return
    model_path=ROOT/f'configs/frozen_action_service_costs_v{version}.json'
    if not model_path.exists():
        print('MODEL_EVALUATION_NOT_READY',flush=True)
        return
    receipt=ROOT/f'state/service_model_v{version}_frozen.json'
    if not receipt.exists():
        print('MODEL_NOT_QUALIFIED_MISSING_FREEZE_RECEIPT',version,flush=True);return
    assert json.loads(receipt.read_text())['SHA256']==digest(model_path)
    source=json.loads(model_path.read_text())
    models={(r['method'],r['layers'],r['output'],r['batch']):r for r in source['model']['models']}
    doc=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    kappa=json.loads((ROOT/'configs/frozen_single_pressure_rule_v1.json').read_text())['rule']['selected_kappa']
    groups=collections.defaultdict(dict)
    stages=('heldout_confirmation','decode_batch_counterfactual','output_residency_counterfactual',
        'new_QPS_confirmation','residency_discovery','batch4_knob_confirmation')
    if version>=3:stages=('batch4_knob_confirmation',)
    for row in doc['starts']:
        if row['method'] not in ('Layer','Token'):
            continue
        if row['stage'] not in stages and not (version==2 and row['stage']=='source_window_control' and row['source_window']==4):
            continue
        key=(row['stage'],row['layers'],row['output'],row['QPS'],row['count'],row['actual_batch'],row['source_window'],row['seed'])
        groups[key][row['method']]=row
    gate=json.loads((ROOT/'state/oracle_SLO_validation_gate_v1.json').read_text())
    results=[]
    for key,pair in sorted(groups.items()):
        stage,m,o,q,n,b,w,seed=key
        if set(pair)!=set(('Layer','Token')) or any((a,m,o,b) not in models for a in pair):
            continue
        table={a:Cost(**models[a,m,o,b]['cost_seconds']) for a in pair}
        arrivals=pair['Layer']['arrival_offsets']
        assert arrivals==pair['Token']['arrival_offsets']
        block=math.lcm(18,b)
        if n%block:
            continue
        oracle=exhaustive(arrivals,table,b,w,o,block)
        one=pressure_rule(arrivals,table,kappa,b,w,o,block)
        two=demand_rule(arrivals,table,b,w,o,block)
        validations={}
        for a,row in pair.items():
            estimate=oracle['fixed'][a]
            actual=row['metric_summary']
            joint=row['common_Balanced']['Joint_attainment']
            goodput=row['common_Balanced']['Output_token_SLO_goodput']
            errors=dict(TTFT_p95_relative=abs(estimate['TTFT_p95_ms']/actual['TTFT_p95_ms']-1),
                output_tokens_s_relative=abs(estimate['output_tokens_s']/actual['output_tokens_s']-1),
                Ready_D_mean_absolute_ms=abs(estimate['Ready_D_wait_mean_ms']-row['Ready_D_mean_ms']),
                Joint_attainment_absolute=abs(estimate['joint_attainment']-joint),
                qualified_tokens_relative=(abs(estimate['qualified_output_tokens_s']/goodput-1) if goodput>0 else None))
            timing=(errors['TTFT_p95_relative']<=.10 and errors['output_tokens_s_relative']<=.05 and
                errors['Ready_D_mean_absolute_ms']<=max(10.,.20*row['Ready_D_mean_ms']))
            SLO=(errors['Joint_attainment_absolute']<=gate['Joint_attainment_absolute_max'] and
                (errors['qualified_tokens_relative']<=gate['qualified_tokens_relative_max'] if goodput>0
                 else estimate['qualified_output_tokens_s']==0))
            validations[a]=dict(name=row['name'],errors=errors,timing_passed=timing,SLO_passed=SLO,
                actual=dict(actual,Joint_attainment=joint,qualified_tokens_s=goodput),predicted=estimate)
        best=oracle['fixed'][oracle['best_fixed']]
        denominator=best['qualified_output_tokens_s']
        headroom=oracle['oracle']['qualified_output_tokens_s']/denominator-1 if denominator>0 else None
        result=dict(stage=stage,layers=m,output=o,QPS=q,count=n,batch=b,window=w,seed=seed,
            research_split=('DISCOVERY_COUNTERFACTUAL_USED_FOR_HYPOTHESIS_REVISION' if stage=='source_window_control' and seed==2026111901 else 'NEW_RULE_EVALUATION_TRACE'),
            names={a:r['name'] for a,r in pair.items()},model_status={a:models[a,m,o,b]['status'] for a in pair},
            fixed_trace_validation=validations,
            qualified_for_counterfactual_interpretation=all(v['timing_passed'] and v['SLO_passed'] for v in validations.values()),
            oracle=oracle,single_pressure_rule=one,demand_backlog_rule=two,
            modeled_SLO_headroom_over_best_same_cut_fixed=headroom,
            evidence='OFFLINE_ORACLE',
            scope='Keep discovery counterfactuals separate from new-rule evaluation. Source-window seed1901 informed the demand-model hypothesis and is not a held-out architecture test. Long Output and Batch4 include explicit extrapolations. Failed validation makes headroom Diagnostic. No measured online dynamic speedup, Decode-tail guarantee or sustainable-capacity claim follows.')
        results.append(result)
    classifications=[]
    for comp in doc['comparisons']:
        if comp['reference']!='Token' or comp['candidate']!='Layer':
            continue
        if comp['stage'] not in stages or comp['starts']<3:
            continue
        m,o,b=comp['layers'],comp['output'],comp['batch']
        if any((a,m,o,b) not in models for a in ('Layer','Token')):
            continue
        table={a:Cost(**models[a,m,o,b]['cost_seconds']) for a in ('Layer','Token')}
        values=demand_scores(table,b,1000000)
        forecast=min(values['candidates'],key=lambda a:values['candidates'][a]['score'])
        CI=comp['metrics']['TTFT_p95_ms']['ratio_CI95']
        winner='Layer' if CI and CI[1]<1 else 'Token' if CI and CI[0]>1 else 'UNRESOLVED'
        r=comp['metrics']['TTFT_p95_ms']['ratio']
        regret=(max(0.,r-1) if forecast=='Layer' else max(0.,1/r-1))
        classifications.append(dict(stage=comp['stage'],layers=m,output=o,QPS=comp['QPS'],batch=b,
            count=comp['count'],window=comp['window'],starts=comp['starts'],seeds=comp['seeds'],
            actual_Layer_over_Token_TTFT_p95=r,CI95=CI,confidence_label=winner,
            calibrated_saturated_cycle_forecast=forecast,forecast_correct=(forecast==winner if winner!='UNRESOLVED' else None),
            measured_TTFT_regret=regret,
            Decode_residency_per_request_seconds=values['Decode_residency_per_request_seconds'],
            limitation='This cost-only forecast assumes congestion. The two-knob replay uses observed total backlog instead. Accuracy excludes unresolved cells.'))
    filename='frozen_model_validation_and_oracle'+(f'_v{version}' if version>=3 else '')+'.json'
    write(ROOT/'analysis'/filename,dict(time=time.time(),
        model_version=version,model_SHA256=digest(model_path),single_pressure_kappa=kappa,SLO_validation_gate=gate,
        results=results,confident_cell_forecasts=classifications,
        future_information_rule='Oracle can read the whole supplied trace. Rule callbacks see only current arrived/ready/active state. Frozen fixed-trace prediction is Offline Replay, not a real online controller.',
        evidence='OFFLINE_REPLAY_AND_ORACLE'))
    print('NEW_MODEL_EVALUATION',version,len(results),sum(r['qualified_for_counterfactual_interpretation'] for r in results),
        'CONFIDENT_CELLS',sum(r['confidence_label']!='UNRESOLVED' for r in classifications),flush=True)


if __name__=='__main__':
    main()
