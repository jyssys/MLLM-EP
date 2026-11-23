"""Partition observed P paths without allocating shared GPU work to a request."""
import argparse
import bisect
import csv
import json
import time
from collections import Counter,defaultdict
import numpy as np
from .common import ROOT,write,bind,digest
from .kernel_analysis import analyze_activity,merge,duration


def records(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def csv_save(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


class Timeline:
    def __init__(self,values):
        self.values=merge(values);self.ends=[b for a,b in self.values];self.starts=[a for a,b in self.values]
        self.prefix=[0]
        for a,b in self.values:self.prefix.append(self.prefix[-1]+b-a)

    def area(self,left,right):
        if right<=left:return 0
        lo=bisect.bisect_right(self.ends,left)
        hi=bisect.bisect_left(self.starts,right)
        if hi<=lo:return 0
        result=self.prefix[hi]-self.prefix[lo]
        a,b=self.values[lo];result-=max(0,left-a)
        a,b=self.values[hi-1];result-=max(0,b-right)
        return result


def overlap(left,right):
    a,b=merge(left),merge(right);i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if hi>lo:out.append([lo,hi])
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return out


def credits(folder):
    cfg=json.loads((folder/'configuration.json').read_text());limit=cfg.get('continuous_source_credit_limit',0)
    window=json.loads((folder/'window.json').read_text());left,right=window['start'],window['end']
    pending=set();acquired={};spans={};durations=[];by_policy=Counter();area=full=0.;cursor=left;full_spans=[]
    events=records(folder/'credit_events.jsonl')
    if not limit:return dict(limit=0,policy='UNBOUNDED_COORDINATOR_WINDOW',per_request_blocked_ms={})
    for event in sorted(events,key=lambda r:r['time']):
        point=event['time'];assert left<=point<=right
        area+=len(pending)*(point-cursor);full+=(point-cursor)*int(len(pending)>=limit)
        if len(pending)>=limit and point>cursor:full_spans.append([cursor,point])
        assert event['owned_before']==len(pending)
        rid=event['request_id']
        if event['event']=='ACQUIRE_P_DISPATCH':
            assert rid not in pending;pending.add(rid);acquired[rid]=point
        else:
            assert event['event'] in ('RELEASE_FIRST_CLIENT_TOKEN','RELEASE_SAFE_SOURCE_RETIREMENT') and rid in pending
            pending.remove(rid);spans[rid]=[acquired[rid],point];durations.append(1000*(point-acquired[rid]));by_policy[event['event']]+=1
        assert len(pending)<=limit;cursor=point
    area+=len(pending)*(right-cursor);full+=(right-cursor)*int(len(pending)>=limit)
    if len(pending)>=limit and right>cursor:full_spans.append([cursor,right])
    assert not pending and len(spans)==cfg['count']
    blocked=Counter();eligible=Counter();segments=records(folder/'P_wait_segments.jsonl')
    for row in segments:
        target=blocked if row['source_owned']>=limit else eligible
        for index in row['pending']:target['q'+str(index)]+=1000*(row['end']-row['start'])
    offered={r['request_id']:r for r in records(folder/'offered_requests.jsonl')}
    assert set(offered)==set(acquired)
    timeline=Timeline(full_spans);joint={rid:1000*timeline.area(r['arrival'],acquired[rid]) for rid,r in offered.items()}
    return dict(limit=limit,policy='SAFE_RETIREMENT_WITH_CLIENT_FALLBACK' if cfg.get('source_credit_release_on_retirement') else 'FIRST_CLIENT_TOKEN',
        occupancy_time_weighted_mean=area/(right-left),full_fraction=full/(right-left),
        mean_acquire_to_release_ms=float(np.mean(durations)),release_events=dict(by_policy),credit_spans=spans,
        per_request_blocked_ms=dict(blocked) if segments else None,
        per_request_observed_credit_eligible_pending_ms=dict(eligible) if segments else None,
        pending_segment_status='INSTRUMENTED' if segments else 'NOT_INSTRUMENTED_IN_THIS_CONTINUOUS_SOURCE',
        source_window_full_intervals=merge(full_spans),
        per_request_arrival_to_dispatch_full_window_overlap_ms=joint,
        mean_arrival_to_dispatch_full_window_overlap_ms=float(np.mean(list(joint.values()))),
        scope='Coordinator-window occupancy only. This is not actual GPU batch,KV capacity or hidden-slot ownership. '
            'Absent continuous pending-segment records are NOT_INSTRUMENTED,not zero waiting. '
            'Arrival→dispatch/full-window overlap is reconstructed from exact credit events and scheduled arrivals. '
            'It is a joint observed state,not exclusive causal credit wait. Coordinator enqueue can lag scheduled arrival. '
            'It does not establish physical GPU idle or causal speedup.')


def cpc(test):
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    names=test['jobs'];folders=[ROOT/'raw'/name for name in names]
    if not all((p/'dispatcher_result.json').exists() for p in folders):return dict(status='NOT_RUN_OR_INCOMPLETE',passed=False)
    assert all(json.loads((p/'dispatcher_result.json').read_text())['complete'] for p in folders)
    metrics=[summarize(name) for name in names];outputs=[records(p/'requests.jsonl') for p in folders]
    gold=[(r['prompt_id'],r['output_tokens']) for r in sorted(outputs[0],key=lambda r:r['sample_id'])]
    exact=all([(r['prompt_id'],r['output_tokens']) for r in sorted(rows,key=lambda r:r['sample_id'])]==gold for rows in outputs)
    fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms')
    ratios={key:metrics[1][key]/((metrics[0][key]+metrics[2][key])/2) for key in fields}
    passed=exact and all(abs(value-1)<=.05 for value in ratios.values())
    counter=json.loads((folders[1]/'activity_prefill/activity_counters.json').read_text());passed &= counter['passed']
    return dict(status='PASS' if passed else 'DIAGNOSTIC',passed=bool(passed),ratios=ratios,
        exact_request_and_output_work=exact,collector_counters=counter,
        scope='Full-trace clean/profile/clean5% gate on all five latency metrics. Exact per-request output lengths. '
            'All P queries are inside this full trace. This is one triplet,not a serving superiority CI.')


def profile_path(name,kernels):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    event_file=folder/'P_model_forward_events_prefill.json'
    if not event_file.exists():
        from ..dense_ar_longbench400_pd.service_accounting import analyze
        host=analyze(name);roi=kernels['ROI'];offset=roi['left']['cupti_ns']-(roi['left']['host_lower_ns']+roi['left']['host_upper_ns'])/2
        busy=Timeline(kernels['intervals']);output={f'q{v["sample_id"]}':v for v in records(folder/'requests.jsonl')}
        ready={r['request_id']:r['source_ready'] for r in records(folder/'handoffs.jsonl')};rows=[]
        for value in host['rows']:
            rid=value['request_id'];left=output[rid]['arrival']*1e9+offset;right=ready[rid]*1e9+offset
            assert roi['left']['cupti_ns']<=left and right<=roi['right']['cupti_ns']
            active=busy.area(left,right)
            rows.append(dict(value,Arrival_to_P_ready_ms=(right-left)/1e6,
                all_P_kernel_inside_arrival_to_ready_ms=active/1e6,no_observed_P_kernel_ms=(right-left-active)/1e6))
        means=dict(host['means'])
        for key in ('Arrival_to_P_ready_ms','all_P_kernel_inside_arrival_to_ready_ms','no_observed_P_kernel_ms'):
            means[key]=float(np.mean([v[key] for v in rows]))
        return dict(name=name,rows=rows,means=means,credit=credits(folder),batches=[],
            per_request_forward_GPU_attribution='NOT_MEASURED',
            GPU_resource_demand_mean_per_request_ms=kernels['kernel_busy_ns']/1e6/cfg['count'],
            scope='Original source kernel-only capture. No model-call event observer was installed. '
                'Total P kernel union divided by request count is mean resource demand,not a per-request elapsed service path. '
                'Do not infer zero own compute from missing per-request event attribution.')
    events=json.loads(event_file.read_text());anchor=events['origin']
    event_origin=(anchor['host_lower_ns']+anchor['host_upper_ns'])/2
    roi=kernels['ROI'];offset=roi['left']['cupti_ns']-(roi['left']['host_lower_ns']+roi['left']['host_upper_ns'])/2
    drift=(roi['right']['cupti_ns']-(roi['right']['host_lower_ns']+roi['right']['host_upper_ns'])/2)-offset
    max_uncertainty=anchor['uncertainty_ns']+max(a['host_upper_ns']-a['host_lower_ns'] for a in (roi['left'],roi['right']))+abs(drift)
    origin=event_origin+offset;busy=Timeline(kernels['intervals']);own=defaultdict(list);shared=defaultdict(list);batches=[]
    for index,row in enumerate(events['rows']):
        left=origin+row['GPU_start_ms_from_origin']*1e6;right=origin+row['GPU_end_ms_from_origin']*1e6
        assert right>=left
        queries=[q for q in row['queries'] if q['request_id'].startswith('q')]
        group=shared if len(queries)>1 else own
        for q in queries:group[q['request_id']].append([left,right])
        batches.append(dict(batch=index,GPU_start_CUPTI_ns=left,GPU_end_CUPTI_ns=right,**row))
    output={f'q{r["sample_id"]}':r for r in records(folder/'requests.jsonl')}
    ready={r['request_id']:r['source_ready'] for r in records(folder/'handoffs.jsonl')}
    from ..dense_ar_longbench400_pd.service_accounting import analyze
    host=analyze(name);credit=credits(folder);host_rows={r['request_id']:r for r in host['rows']};rows=[]
    for rid,value in output.items():
        left=value['arrival']*1e9+offset;right=ready[rid]*1e9+offset
        assert roi['left']['cupti_ns']<=left and right<=roi['right']['cupti_ns'],('Full-path capture required',rid)
        unique=Timeline(overlap(own[rid],kernels['intervals']));mixed=Timeline(overlap(shared[rid],kernels['intervals']))
        own_ns=unique.area(left,right);mixed_ns=mixed.area(left,right);busy_ns=busy.area(left,right)
        assert own_ns+mixed_ns<=busy_ns+1,(rid,own_ns,mixed_ns,busy_ns)
        total=right-left;extra=busy_ns-own_ns-mixed_ns;no_kernel=total-busy_ns
        row=dict(name=name,request_id=rid,prompt_id=value['prompt_id'],input_tokens=value['input_tokens'],output_tokens=value['output_tokens'],
            **{k:v for k,v in host_rows[rid].items() if k.endswith('_ms')},
            Arrival_to_P_ready_ms=total/1e6,own_single_request_forward_kernel_ms=own_ns/1e6,
            all_P_kernel_inside_arrival_to_ready_ms=busy_ns/1e6,
            shared_forward_kernel_ms=mixed_ns/1e6,other_P_kernel_or_outside_model_scope_ms=extra/1e6,
            no_observed_P_kernel_ms=no_kernel/1e6,
            arrival_to_P_dispatch_overlap_full_source_window_ms=credit['per_request_arrival_to_dispatch_full_window_overlap_ms'].get(rid,0))
        assert abs(row['Arrival_to_P_ready_ms']-sum(row[k] for k in ('own_single_request_forward_kernel_ms',
            'shared_forward_kernel_ms','other_P_kernel_or_outside_model_scope_ms','no_observed_P_kernel_ms')))<1e-4
        for category,spans in kernels['category_intervals'].items():
            row['own_forward_'+category+'_union_ms']=Timeline(overlap(own[rid],spans)).area(left,right)/1e6
        rows.append(row)
    fields=[k for k,v in rows[0].items() if k.endswith('_ms')]
    return dict(name=name,rows=rows,means={k:float(np.mean([r.get(k,0) for r in rows])) for k in fields},
        credit=credit,batches=batches,event_origin_uncertainty_ns=anchor['uncertainty_ns'],
        mapping_uncertainty_bound_ns=max_uncertainty,CUPTI_anchor_drift_ns=drift,
        scope='Exact additive Arrival→P-ready time partition on the mapped GPU timeline,subject to the reported clock uncertainty. '
            'Single-request model-enclosure activity is associated with that forward,but lacks operator-to-request correlation. '
            'Concurrent kernels on other streams can enter the enclosure. Do not call it a correlation-verified exclusive request service. '
            'Shared-batch kernels remain unallocated. '
            'Other P kernels can belong to preceding requests or metadata/sampling outside the model-call scope. '
            'No P kernel is not proof of all-engine idle or a specific host cause. Waiting time is not exclusive service. '
            'The same kernel can delay many requests. Summed request waiting is not total GPU work. '
            'Clock uncertainty uses endpoint anchors and drift. It is not a formal bound on unobserved interior clock jitter.')


def analyze():
    bind();plan=ROOT/'jobs/same_pair_p_critical_cpc_r4.json';tests=json.loads(plan.read_text())['tests'];results=[]
    stock=ROOT/'jobs/stock_original_kernel_cpc_and_state_diagnostics.json'
    if stock.exists():tests+=json.loads(stock.read_text())['tests']
    for test in tests:
        if not all((ROOT/'raw'/n/'dispatcher_result.json').exists() for n in test['jobs']):
            results.append(dict(**test,evidence='NOT_RUN_OR_INCOMPLETE',
                observer_gate=dict(passed=False,status='NOT_RUN_OR_INCOMPLETE')));continue
        gate=cpc(test);name=test['jobs'][1]
        try:
            kernels=analyze_activity(ROOT/'raw'/name/'activity_prefill');path=profile_path(name,kernels)
            # Save large interval lists once. Summary tables retain the source hash.
            detail=ROOT/'analysis'/(name+'_kernel_and_path.json');write(detail,dict(kernels=kernels,path=path,observer_gate=gate))
            csv_save(ROOT/'figures'/(name+'_per_request_path.csv'),path['rows'])
            results.append(dict(**test,observer_gate=gate,evidence='MEASURED_ONLINE_KERNEL_ACTIVITY' if gate['passed'] else 'DIAGNOSTIC',
                mean_path=path['means'],kernel_busy_fraction=kernels['kernel_busy_fraction'],
                GPU_resource_demand_mean_per_request_ms=kernels['kernel_busy_ns']/1e6/
                    json.loads((ROOT/'raw'/name/'configuration.json').read_text())['count'],
                kernel_partition_ns=kernels['exclusive_time_partition_ns'],detail=str(detail.relative_to(ROOT)),detail_SHA256=digest(detail)))
        except Exception as error:results.append(dict(**test,observer_gate=gate,evidence='FAILED',error=repr(error)))
    write(ROOT/'analysis/p_only_fulltrace_cpc_analysis.json',dict(time=time.time(),plan_SHA256=digest(plan),results=results,
        scope='No failed observer capture is primary. Old QPS4.8 ROI attribution never substitutes for this QPS2 full path. '
            'Start means and request waiting have different denominators. All failed triplets remain in the registry.'))
    csv_save(ROOT/'figures/p_only_fulltrace_cpc_summary.csv',results)
    print([(r['label'],r['start'],r['evidence'],r.get('error')) for r in results],flush=True)


def cpu_test():
    timeline=Timeline([[0,3],[2,4],[7,8]])
    for left,right,expected in ((2,7.5,2.5),(3,4,.9999999999999999),(-1,9,5),(4,7,0),(5,6,0),(9,10,0)):
        assert abs(timeline.area(left,right)-expected)<1e-10,(left,right,timeline.area(left,right))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('test','analyze'));args=parser.parse_args()
    if args.action=='test':cpu_test();print('P_TIMELINE_PARTITION_CPU_PASS')
    else:analyze()
