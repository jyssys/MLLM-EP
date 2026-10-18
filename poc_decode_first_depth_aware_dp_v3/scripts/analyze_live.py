"""Measured DP4 metrics, observer checks, and calibrated cost validation."""
import argparse
import collections
import csv
import statistics
import numpy as np
from scipy.optimize import nnls
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_decode_first_depth_aware_dp_v3.simulator import stage_features, CostModel, simulate_dp

STAGES={'_run_mixed_recurrent_batch':'recurrent','_run_prefill_batch':'prefill','_stage_coda_batch':'coda'}

def measured_case(folder,case):
    ranks=[read(folder/f'rank_{i}'/(case['name']+'.json')) for i in range(4)]
    outputs=[o for rank in ranks for o in rank['outputs']]
    e2e=[1000*(o['finish_s']-o['arrival_s']) for o in outputs]
    ttft=[1000*(o['first_token_s']-o['arrival_s']) for o in outputs]
    itl=[1000*(b-a) for o in outputs for a,b in zip(o['token_host_times_s'],o['token_host_times_s'][1:])]
    end=max(o['finish_s'] for o in outputs)
    first=min(o['arrival_s'] for o in outputs)
    counts=[len(r['outputs']) for r in ranks]
    finish=[max((o['finish_s'] for o in r['outputs']),default=0.) for r in ranks]
    record=dict(case=case['name'],concurrency=case['concurrency'],arrival=case['arrival'],policy=case['policy'],
        requests=len(outputs),E2E_P50_ms=float(np.quantile(e2e,.5)),E2E_P95_ms=float(np.quantile(e2e,.95)),
        TTFT_P95_ms=float(np.quantile(ttft,.95)),ITL_P95_ms=float(np.quantile(itl,.95)),
        makespan_ms=end*1000,requests_per_s=len(outputs)/(end-first),
        output_tokens_per_s=sum(o['generated_tokens'] for o in outputs)/(end-first),
        finish_CV=float(np.std(finish)/np.mean(finish)),rank_request_counts=counts,
        recurrent_batches=sum(r['stats']['recurrent_batches'] for r in ranks),
        mean_recurrent_batch=sum(r['stats']['recurrent_steps'] for r in ranks)/max(1,sum(r['stats']['recurrent_batches'] for r in ranks)),
        physical_label='MEASURED_LIVE_DP4',trace=any(r['trace'] for r in ranks))
    slo=read(ROOT/'artifacts/slo_frozen.json')['E2E_SLO_ms']
    record['relative_goodput_rps']=sum(v<=slo for v in e2e)/(end-first)
    record['E2E_SLO_ms']=slo
    record['tail_after_first_rank_finishes_ms']=(max(finish)-min(finish))*1000
    controller=read(folder/(case['name']+'.json'))
    witnesses=[]
    if case['arrival']=='burst' and all(counts):
        for event in controller['notifications']:
            if event['event']!='queue' or not event.get('t'):continue
            queued=event['waiting']+event['ready']+event['coda']
            if not queued:continue
            idle=[i for i,f in enumerate(finish) if i!=event['rank'] and f<event['t']]
            if idle:
                witnesses.append(dict(idle_ranks=idle,other_rank=event['rank'],time_s=event['t'],
                    eligible_queued=queued))
    record['idle_while_other_runnable_witnesses']=len(witnesses)
    return record,ranks

def simulated_inputs(case,ranks,assignments):
    byid={o['id']:o for rank in ranks for o in rank['outputs']}
    result=[[] for _ in range(4)]
    for order,a in enumerate(assignments):
        o=byid[a['id']]
        result[a['rank']].append(dict(id=o['id'],prompt_tokens=o['prompt_tokens'],
            depths=[e['depth'] for e in o['depths']],arrival_ms=a['arrival_s']*1000,order=order,cap=256))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--trace-run',type=Path,required=True)
    p.add_argument('--clean-run',type=Path,required=True);args=p.parse_args()
    assert read(args.trace_run/'status.json')['success']
    assert read(args.clean_run/'status.json')['success']
    trace_plan=read(ROOT/'artifacts/live_trace_plan.json')
    clean_plan=read(ROOT/'artifacts/live_clean_plan.json')
    training={s:[] for s in ['recurrent','prefill','coda']}
    cost_bins=collections.defaultdict(list)
    all_metrics=[]
    for case in trace_plan['cases']:
        metric,ranks=measured_case(args.trace_run,case)
        all_metrics.append(metric)
        if not case.get('cost_fit',False):continue
        for rank in ranks:
            for s in rank['stages']:
                label=STAGES[s['stage']]
                if not s['items']:continue
                value=max(s['gpu_ms'],s['host_ms'])
                training[label].append((stage_features(label,s['items']),value))
                context=int(statistics.mean(i.get('context',i.get('query',0)) for i in s['items'])//1024)
                cost_bins[(label,len(s['items']),context)].append(value)
    with (ROOT/'artifacts/per_loop_costs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['stage','active_batch','context_K_bin','samples','median_span_ms','P95_span_ms'])
        writer.writeheader()
        for (stage,batch,context),values in sorted(cost_bins.items()):
            writer.writerow(dict(stage=stage,active_batch=batch,context_K_bin=context,samples=len(values),
                median_span_ms=float(np.median(values)),P95_span_ms=float(np.quantile(values,.95))))
    parameters={}
    for stage,values in training.items():
        x=np.asarray([a for a,b in values],float)
        y=np.asarray([b for a,b in values],float)
        # Suppress isolated graph/cold capture spikes from cost calibration, not raw evidence.
        keep=y<=np.quantile(y,.995)
        coef,_=nnls(x[keep],y[keep])
        parameters[stage]=dict(coefficients=coef.tolist(),minimum_ms=float(np.quantile(y[keep],.05)),
            samples=int(keep.sum()),excluded_high_tail=int((~keep).sum()),
            training_MAE_ms=float(np.mean(abs(x[keep]@coef-y[keep]))))
    model=CostModel(parameters)
    # Fit a common per-stage host-loop residual on calibration requests only.
    residuals=[]
    for case in trace_plan['cases']:
        if not case.get('cost_fit',False):continue
        metric,ranks=measured_case(args.trace_run,case)
        meta=read(args.trace_run/(case['name']+'.json'))
        assigned=simulated_inputs(case,ranks,meta['assignments'])
        pred=simulate_dp(assigned,model)
        for i,rank in enumerate(ranks):
            n=len(pred['ranks'][i]['stages'])
            if n:
                last=max(o['finish_s'] for o in rank['outputs'])*1000
                residuals.append((last-pred['ranks'][i]['makespan_ms'])/n)
    residual=max(0.,float(np.median(residuals)))
    for p in parameters.values():p['coefficients'][0]+=residual
    model=CostModel(parameters)
    validation=[]
    pair_outputs={}
    for case in clean_plan['cases']:
        metric,ranks=measured_case(args.clean_run,case)
        all_metrics.append(metric)
        meta=read(args.clean_run/(case['name']+'.json'))
        assigned=simulated_inputs(case,ranks,meta['assignments'])
        pred=simulate_dp(assigned,model)
        errors={k:abs(pred[k]-metric[k])/metric[k] for k in
            ['makespan_ms','E2E_P50_ms','E2E_P95_ms','TTFT_P95_ms','ITL_P95_ms']}
        validation.append(dict(case=case['name'],policy=case['policy'],arrival=case['arrival'],
            actual={k:metric[k] for k in errors},predicted={k:pred[k] for k in errors},relative_errors=errors))
        if case['arrival']=='burst':
            pair_outputs[(case['concurrency'],case['policy'])]={o['id']:o for rank in ranks for o in rank['outputs']}
    parity=[]
    for c in [1,2,4,8,16,32,64]:
        baseline=pair_outputs[c,'round_robin']
        for policy in ['queue','length','strong_cost']:
            compared=pair_outputs[c,policy]
            exact=sum(baseline[i]['output_ids']==compared[i]['output_ids'] for i in baseline)
            depth_exact=sum([e['depth'] for e in baseline[i]['depths']]==[e['depth'] for e in compared[i]['depths']] for i in baseline)
            parity.append(dict(concurrency=c,policy=policy,n=len(baseline),exact_token_requests=exact,
                exact_depth_records=depth_exact,
                note='Actual exit-depth sequences, without local tick IDs. Different batch shapes can change BF16 token outputs.'))
    primary=[v for v in validation if v['policy'] in ['round_robin','queue']]
    median_errors={k:float(np.median([v['relative_errors'][k] for v in primary])) for k in primary[0]['relative_errors']}
    passed=all(v<=.10 for v in median_errors.values())
    write(ROOT/'artifacts/cdb_cost_model.json',dict(parameters=parameters,host_tick_residual_ms=residual,
        fitting='Calibration requests only. Nonnegative least squares on host/CUDA stage spans.',
        fit_category='MEASURED_COST_MODEL_NOT_KERNEL_SERVICE_SUM',
        validation=validation,median_RR_queue_relative_errors=median_errors,
        status='VALIDATED' if passed else 'SIMULATOR_UNVALIDATED',
        caveat='All metrics and systematic arrival error require inspection. A median pass alone does not prove universal validity.'))
    write(ROOT/'artifacts/live_dp4_metrics.json',all_metrics)
    with (ROOT/'artifacts/live_dp4_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_metrics[0]));writer.writeheader();writer.writerows(all_metrics)
    write(ROOT/'artifacts/scheduler_output_parity.json',parity)
    # C64 RR uses identical held-out IDs in the trace and clean lanes.
    tcase=next(c for c in trace_plan['cases'] if c['name']=='trace_burst_C64_rr')
    ccase=next(c for c in clean_plan['cases'] if c['name']=='clean_burst_C64_round_robin')
    _,tranks=measured_case(args.trace_run,tcase)
    _,cranks=measured_case(args.clean_run,ccase)
    tvalues={o['id']:o for r in tranks for o in r['outputs']}
    cvalues={o['id']:o for r in cranks for o in r['outputs']}
    deltas=[100*((tvalues[i]['finish_s']-tvalues[i]['arrival_s'])/(cvalues[i]['finish_s']-cvalues[i]['arrival_s'])-1) for i in tvalues]
    write(ROOT/'artifacts/DP4_observer_check.json',dict(n=len(deltas),
        median_E2E_delta_percent=float(np.median(deltas)),mean_E2E_delta_percent=float(np.mean(deltas)),
        exact_output_requests=sum(tvalues[i]['output_ids']==cvalues[i]['output_ids'] for i in tvalues),
        scope='One paired C64 burst. Requests share GPU/run conditions. This is not 64 independent observer repeats.'))
    print('MODEL_VALIDATION',passed,median_errors)

if __name__=='__main__':main()
