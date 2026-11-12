"""Report censored D inactivity without inventing unobserved activity."""
import csv,json,statistics
from .common import *
from .analyze import metrics,records,percentile
from .physical_analysis import activity,merge,gaps,queue_screen


def main(plan_file='quiet_confirmation_plan.json',output_file='native_quiet_same_window_screen.json'):
    plan=json.loads((ROOT/'state'/plan_file).read_text());names=plan['jobs']
    values=[metrics(n) for n in names]
    if not all(v['complete'] for v in values):print('INCOMPLETE');return
    a,observed,b=values;name=names[1];folder=ROOT/'raw'/name
    keys=('completed_QPS','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
    ratio={k:observed[k]/statistics.mean([a[k],b[k]]) for k in keys}
    aa={k:b[k]/a[k] for k in keys};interference=all(abs(v-1)<=.05 for v in ratio.values())
    window=json.loads((folder/'window.json').read_text());duration=window['last_arrival']-window['start']
    left=window['start']+.2*duration;right=window['start']+.8*duration
    P=activity(name,'prefill',left,right);Q=queue_screen(name,left,right)
    if plan.get('capacity_job'):
        rows=records(ROOT/'raw'/plan['capacity_job']/'cohorts.jsonl')
        reference=max(statistics.mean(r['batch']/r['P_plus_first_sample_s'] for r in rows if r['batch']==b) for b in (1,2,4))
        Q.update(P_capacity_reference_requests_s=reference,P_service_proximity=Q['P_service_requests_s']/reference)
    begin=json.loads((folder/'activity_decode_begin.json').read_text());end=json.loads((folder/'activity_decode_end.json').read_text())
    anchor=begin['anchor'];offset=anchor['host_midpoint']-anchor['cupti_ns']/1e9
    epochs=merge([(max(left,e['start']['host_midpoint']),min(right,e['end']['host_midpoint']))
        for e in end['capture_epochs'] if e['end']['host_midpoint']>left and e['start']['host_midpoint']<right])
    kernels=[]
    with (folder/'activity_decode.csv').open() as f:
        for r in csv.DictReader(f):
            if r['kind']=='kernel':kernels.append((int(r['start_ns'])/1e9+offset,int(r['end_ns'])/1e9+offset))
    idle=[]
    for x,y in epochs:idle.extend(gaps(kernels,x,y))
    lengths=[1000*(y-x) for x,y in idle];total=sum(y-x for x,y in idle)
    counters=end['counters'];valid=end['result']==0 and counters['dropped']==0 and counters['invalid']==0
    # The witness is after all outputs. It verifies collector activity, not slack.
    witness=end['post_serving_witness'];wa=witness['start']['host_midpoint'];wb=witness['end']['host_midpoint']
    witness_kernels=sum(y>wa and x<wb for x,y in kernels);valid=valid and witness_kernels>=2
    detail=[r for p in folder.glob('extension_prefill_*.jsonl') for r in records(p)
        if r['event']=='SCHEDULER_RESULT' and left<=r['time']<=right]
    waiting=[r for x in detail for r in x['waiting_requests']]
    statuses=sorted(set(str(r['status']) for r in waiting))
    queue_audit=bool(waiting) and all(r['computed']==0 and r['prompt_tokens']==8000 for r in waiting)
    free=min((x['KV_free_blocks'] for x in detail),default=0)
    D=dict(evidence='TRACE_ANALYSIS_CENSORED_QUIET_EPOCHS',capture_epoch_count=len(epochs),
        captured_seconds=sum(y-x for x,y in epochs),measurement_window_seconds=right-left,
        measured_kernel_inactive_seconds=total,physical_inactive_lower_bound_percent=100*total/(right-left),
        selected_idle_window_count=len(idle),selected_idle_mean_ms=statistics.mean(lengths) if lengths else None,
        selected_idle_p50_ms=percentile(lengths,.5),selected_idle_p95_ms=percentile(lengths,.95),
        selected_idle_p99_ms=percentile(lengths,.99),selected_idle_max_ms=max(lengths,default=None),
        entire_D_kernel_busy_percent='NOT_MEASURED',entire_D_idle_distribution='NOT_MEASURED',
        thresholds={str(ms):dict(count=sum(t>=ms for t in lengths),total_seconds=sum(t/1000 for t in lengths if t>=ms)) for ms in (1,2,5,10,20,30,50,100)},
        trace_integrity_pass=valid,counters=counters,post_serving_witness_kernels=witness_kernels,
        scope='Only fenced empty-engine epochs are observed. Uncaptured D periods are UNKNOWN. Inactive time is a lower bound, not the full physical idle ratio or a Decode-safe admission budget.')
    qualified=interference and valid and P['trace_integrity_pass']
    P_gate=P['kernel_busy_percent']>=80 and Q['queue_growth'] and Q['P_service_proximity']>=.85 and Q['P_KV_allocation_refusals']==0 and queue_audit and free>=500
    result=dict(names=names,metrics=values,observer_ratios=ratio,A_A_ratios=aa,interference_pass=interference,
        trace_qualified=qualified,P_gate=P_gate,P_activity=P,P_queue=Q,D_quiet_activity=D,
        P_queue_audit=dict(statuses=statuses,waiting_observations=len(waiting),uncomputed_prompt_waiters_only=queue_audit,
            minimum_free_KV_blocks=free,one_prompt_blocks=500),
        Decode_absolute_SLO_pass=observed['Decode_absolute_SLO_pass'],
        Phase_B_additional_work_intervention='NOT_RUN',spatial_compute_slack='NOT_MEASURED')
    write(ROOT/'analysis'/output_file,result)
    with (folder/'qualified_scope_epochs.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['start_monotonic','end_monotonic','kind'])
        writer.writerows((x,y,'D_captured_epoch') for x,y in epochs)
        writer.writerows((x,y,'D_measured_kernel_inactive') for x,y in idle)
    print(json.dumps(dict(interference_pass=interference,trace_qualified=qualified,P_gate=P_gate,
        ratios=ratio,P_busy_percent=P['kernel_busy_percent'],P_queue=Q,D=D),indent=2))


if __name__=='__main__':
    import sys
    main(*sys.argv[1:])
