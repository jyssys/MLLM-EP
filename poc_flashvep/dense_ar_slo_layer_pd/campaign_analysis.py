"""Build an auditable result inventory without inventing unmeasured costs."""
import collections
from .common import *
from .analyze import records,percentile,completed_D_layers


def decision_audit(name):
    folder=ROOT/'raw'/name
    values=records(folder/'online_decisions.jsonl')
    cuts=collections.Counter();candidates={};reasons=collections.Counter()
    smaller=[];host=[];epochs={}
    anchor=values[0]['time'] if values else 0.
    for index,row in enumerate(values):
        state=row['observation'];choice=row['choice'];size=state['group_size']
        cuts[choice['layers']]+=size;reasons[choice['reason']]+=size
        if 'policy_decision_host_ms' in row:host.append(row['policy_decision_host_ms'])
        epoch=int((row['time']-anchor)//120)*120
        block=epochs.setdefault(str(epoch),dict(decisions=0,requests=0,cuts=collections.Counter(),
            overlapping_candidate_rejections=collections.Counter()))
        block['decisions']+=1;block['requests']+=size;block['cuts'][choice['layers']]+=size
        bycut={c['layers']:c for c in choice['candidates']}
        for cut,c in bycut.items():
            total=candidates.setdefault(str(cut),dict(evaluated_groups=0,evaluated_requests=0,
                accepted_groups=0,accepted_requests=0,rejected_groups=0,
                overlapping_rejection_groups=collections.Counter(),
                expired_own_TTFT_conflicts=0,expired_pending_TTFT_conflicts=0))
            total['evaluated_groups']+=1;total['evaluated_requests']+=size
            total['accepted_groups']+=bool(c['accepted'])
            total['accepted_requests']+=size*bool(c['accepted'])
            total['rejected_groups']+=not c['accepted']
            total['overlapping_rejection_groups'].update(set(c['rejection_reasons']))
            total['expired_own_TTFT_conflicts']+=len(c.get('expired_own_TTFT_not_protected',[]))
            total['expired_pending_TTFT_conflicts']+=len(c.get('expired_pending_TTFT_not_protected',[]))
            block['overlapping_candidate_rejections'].update(set(c['rejection_reasons']))
        largest=bycut.get(12)
        if choice['layers'] in (2,4,8) and largest and not largest['accepted']:
            actual=bycut[choice['layers']]
            smaller.append(dict(index=index,time=row['time'],requests=size,layers=choice['layers'],
                Last12_rejections=largest['rejection_reasons'],
                selected_D_finish_upper=actual['predicted_D_finish'],
                earliest_decode_deadline=state['earliest_decode_deadline'],
                own_TTFT_deadline=state['own_TTFT_deadline']))
    return dict(name=name,decisions=len(values),selected_request_histogram=dict(cuts),
        selected_reasons=dict(reasons),candidate_gates=candidates,
        selected_smaller_when_Last12_rejected=smaller,epochs=epochs,
        policy_Host_ms=dict(samples=len(host),p50=percentile(host,.5),p95=percentile(host,.95),
            p99=percentile(host,.99),maximum=max(host,default=None)),
        evidence='TRACE_ANALYSIS',scope='Candidate gates overlap. Evaluated and accepted candidates are not actual committed work. Cut histograms here describe proposals. Completed request fields and actual commits determine migrated work. Policy time is a Host enclosure, not GPU execution or independent samples.')


def delivery_tail(name):
    worst=None;all_values=[]
    for row in records(ROOT/'raw'/name/'requests.jsonl'):
        for a,b in zip(row['token_events'],row['token_events'][1:]):
            if b['count']-a['count']!=1 or 'worker_time' not in a or 'worker_time' not in b:continue
            parent=1000*(b['time']-a['time']);worker=1000*(b['worker_time']-a['worker_time'])
            value=dict(sample_id=row['sample_id'],token_from=a['count'],token_to=b['count'],
                parent_ms=parent,worker_ms=worker,receive_lag_change_ms=parent-worker)
            all_values.append(value)
            if worst is None or parent>worst['parent_ms']:worst=value
    return dict(name=name,maximum_interval_identity=worst,
        worker_emission_ITL_p99_ms=percentile([v['worker_ms'] for v in all_values],.99),
        parent_ITL_p99_ms=percentile([v['parent_ms'] for v in all_values],.99),
        evidence='TRACE_ANALYSIS',scope='Parent interval equals worker-emission interval plus receive-lag change. Worker time includes GPU and Host scheduling. This identity does not identify the underlying Host cause or decompose percentile differences.')


def inventory():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists(), 'Run the large ledger scan after GPU measurement closes'
    runs=[];decision_values=[];tails=[];stage_seconds=0.
    for folder in sorted((ROOT/'raw').iterdir()):
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists():continue
        result=json.loads(receipt.read_text());cfg=json.loads((folder/'configuration.json').read_text())
        stage_seconds+=result.get('elapsed_seconds',0.)
        output=ROOT/'analysis'/(folder.name+'_output_gate.json')
        output_pass=json.loads(output.read_text()).get('passed') if output.exists() else None
        rows=records(folder/'requests.jsonl')
        cuts=collections.Counter(completed_D_layers(r,cfg) for r in rows)
        long=ROOT/'analysis'/(folder.name+'_sustainable_v2.json')
        metrics=ROOT/'analysis'/(folder.name+'_metrics.json')
        lifetimes={p.name:json.loads(p.read_text()).get('passed') for p in folder.glob('*lifetime_*.json')}
        run=dict(name=folder.name,arm=cfg['arm'],phase=cfg.get('phase'),count=cfg['count'],
            completed_requests=len(rows),output_cap=cfg['cap'],offered_QPS=cfg.get('arrival_rate'),
            arrival_seed=cfg['arrival_seed'],status=result['status'],elapsed_seconds=result.get('elapsed_seconds'),
            correctness_only=cfg.get('correctness_only',False),clean_timing=cfg.get('clean_timing',True),
            output_gate_passed=output_pass,lifetime_checks=lifetimes,
            actual_completed_cut_histogram=dict(cuts),
            actual_migrated_requests=sum(n for k,n in cuts.items() if k),
            actual_migrated_layer_work=sum(k*n for k,n in cuts.items()),
            source_archive_SHA256=digest(folder/'source.zip') if (folder/'source.zip').exists() else None,
            sustainable_analysis=str(long.relative_to(ROOT)) if long.exists() else None,
            finite_analysis=str(metrics.relative_to(ROOT)) if metrics.exists() else None)
        if long.exists():
            run['long_window']=json.loads(long.read_text());window=run['long_window']
            cohort=collections.Counter(completed_D_layers(r,cfg) for r in rows
                if window['window_begin']<=r['arrival']<window['window_end'])
            run.update(measurement_arrival_cut_histogram=dict(cohort),
                measurement_arrival_request_count=sum(cohort.values()),
                measurement_arrival_migrated_requests=sum(n for k,n in cohort.items() if k),
                measurement_arrival_migrated_layer_work=sum(k*n for k,n in cohort.items()))
        runs.append(run)
        if (folder/'online_decisions.jsonl').exists() and result.get('complete'):
            decision_values.append(decision_audit(folder.name))
        if ('async_continuous' in folder.name or 'incremental_guard' in folder.name) and rows:
            tails.append(delivery_tail(folder.name))
    value=dict(time=time.time(),runs=runs,executed_jobs=len(runs),
        complete_jobs=sum(r['status']=='COMPLETE' for r in runs),
        terminal_failure_jobs=sum(r['status']!='COMPLETE' for r in runs),
        summed_stage_wall_seconds=stage_seconds,
        scope='Summed job wall time includes startup, warm-up, transfer and drain. It is not GPU kernel busy time. Pending jobs without a terminal receipt remain outside this inventory. Raw evidence is preserved.',
        evidence='TRACE_ANALYSIS')
    write(ROOT/'analysis/campaign_result_inventory.json',value)
    write(ROOT/'analysis/campaign_candidate_gate_attribution.json',dict(runs=decision_values,
        scope='Candidate rejection counts overlap. Proposed choices are separate from completed migration.',evidence='TRACE_ANALYSIS'))
    write(ROOT/'analysis/campaign_Host_delivery_tail_identities.json',dict(runs=tails,
        unresolved_cause='NOT_IDENTIFIED',evidence='TRACE_ANALYSIS'))
    print('CAMPAIGN_INVENTORY',len(runs),round(stage_seconds/3600,3),len(decision_values),flush=True)


if __name__=='__main__':inventory()
