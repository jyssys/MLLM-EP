"""Describe observed timing accounts without claiming additive p99 causes."""
import csv,json,statistics
from collections import defaultdict
from .common import *
from .analyze import summary,union_length,write as write_result


def weighted_composition(reference,candidate):
    grouped=defaultdict(list)
    for row in reference:grouped[int(row['batch'])].append(float(row['recurrent_enclosure_ms']))
    means={b:statistics.mean(v) for b,v in grouped.items()}
    supported=[r for r in candidate if int(r['batch']) in means]
    return dict(candidate_intervals=len(candidate),common_support_intervals=len(supported),
        reference_core_mean_ms=statistics.mean(float(r['recurrent_enclosure_ms']) for r in reference),
        reference_cost_at_candidate_batch_mix_ms=statistics.mean(means[int(r['batch'])] for r in supported) if supported else None,
        observed_candidate_core_common_support_mean_ms=statistics.mean(float(r['recurrent_enclosure_ms']) for r in supported) if supported else None,
        scope='Descriptive common-support reweighting of actual batch sizes. This does not randomize batch composition or identify a causal fraction of p99.')


def kernel_summary(name,events):
    folder=ROOT/'raw'/name;p=folder/'kernel_tail_capture.json'
    if not p.exists():return None
    capture=json.loads(p.read_text())
    if capture.get('status')!=0 or capture.get('dropped') or capture.get('invalid'):
        return dict(name=name,status='NOT_MEASURED',capture=capture)
    a,b=capture['start'],capture['stop']
    def host(ns):
        return a['host_midpoint']+(ns-a['cupti_ns'])/(b['cupti_ns']-a['cupti_ns'])*(b['host_midpoint']-a['host_midpoint'])
    left,right=a['host_midpoint'],b['host_midpoint'];kernels=[]
    with (folder/'kernel_tail.csv').open() as f:
        for row in csv.DictReader(f):
            start,end=host(int(row['start_ns'])),host(int(row['end_ns']))
            if start<right and end>left:
                kernels.append(dict(start=max(left,start),end=min(right,end),stream=int(row['stream']),node=int(row['graph_node'])))
    if not kernels:return dict(name=name,status='NO_KERNEL_COVERAGE',capture=capture)
    busy=union_length((k['start'],k['end']) for k in kernels)
    streams=capture.get('streams',{})
    decode={v['cupti_stream'] for key,v in streams.items() if key in ('core','decode_capture') and v.get('status')==0}
    prompt={v['cupti_stream'] for key,v in streams.items() if key in ('prompt','prompt_capture') and v.get('status')==0}
    classified=[]
    for event in events:
        if event['prompt'] or event['operation']!='recurrent':continue
        lo,hi=event['begin']['time'],event['end']['time']
        if not (lo>=a['host_upper'] and hi<=b['host_lower']):continue
        selected=[k for k in kernels if k['stream'] in decode and k['start']<hi and k['end']>lo]
        if not selected or decode&prompt:continue
        union_ms=1000*union_length((max(lo,k['start']),min(hi,k['end'])) for k in selected)
        classified.append(dict(batch=len(event['requests']),enclosure_ms=event['device_enclosure_ms'],
            observed_decode_stream_kernel_union_ms=union_ms,
            complement_within_enclosure_ms=1000*(hi-lo)-union_ms,kernels=len(selected)))
    result=dict(name=name,status='OBSERVED_INSTRUMENTED_KERNEL_TAIL',window_s=right-left,
        kernel_union_s=busy,kernel_union_percent=100*busy/(right-left),
        kernel_idle_s=right-left-busy,kernels=len(kernels),
        observed_streams=sorted({k['stream'] for k in kernels}),stream_mapping=streams,
        contained_decode_operations=classified,capture=capture,
        scope='Actual kernel intervals only in this captured tail. Not the full cohort. Operation assignment requires nonoverlapping stream-role mappings. Kernels outside a successfully mapped Decode stream remain unclassified. Instrumentation qualification is separate.')
    write(ROOT/'analysis'/(name+'_kernel_intervals.json'),kernels)
    return result


def main():
    rows=json.loads((ROOT/'analysis/new_diagnostics.json').read_text());by={r['name']:r for r in rows}
    replacements=json.loads((ROOT/'state/trace_versions.json').read_text())['replacements']
    screens=[]
    for arm in ('zero','full','tiled','loop31'):
        clean=by[f'tail_{arm}_clean_n64']['metrics']
        observed=replacements.get(f'tail_{arm}_events_n64',f'tail_{arm}_events_n64')
        x=by[observed]['metrics'];values={}
        for metric,stat in (('TTFT_ms','p95'),('TPOT_ms','p95'),('ITL_ms','p99'),('ITL_ms','maximum')):
            values[metric+'/'+stat]=x[metric][stat]/clean[metric][stat]
        screens.append(dict(arm=arm,event_run=observed,ratios=values,
            five_percent_screen=all(.95<=v<=1.05 for v in values.values()),
            scope='One matched pair. An A/A sensitivity control is separate. A passed point screen is not a causal interference confidence interval.'))
    zero=replacements['tail_zero_events_n64']
    with (ROOT/'analysis'/(zero+'_cycles.csv')).open() as f:reference=list(csv.DictReader(f))
    accounting=[];composition=[]
    for arm in ('full','tiled','loop31'):
        name=replacements.get(f'tail_{arm}_events_n64',f'tail_{arm}_events_n64')
        with (ROOT/'analysis'/(name+'_cycles.csv')).open() as f:candidate=list(csv.DictReader(f))
        keys=('ITL_ms','model_enclosure_ms','between_enclosure_ms','output_observation_delta_ms')
        delta={k:statistics.mean(float(r[k]) for r in candidate)-statistics.mean(float(r[k]) for r in reference) for k in keys}
        accounting.append(dict(arm=arm,delta_means_ms=delta,
            arithmetic_residual_ms=delta['ITL_ms']-sum(delta[k] for k in keys[1:]),
            scope='An additive mean observation account. Not causal percentages and not a p99 decomposition. Device enclosures contain execution waits and possible host submission gaps.'))
        composition.append(dict(arm=arm,**weighted_composition(reference,candidate)))
    kernels=[]
    for name in ('tail_zero_kernel_tail_n64','tail_full_kernel_tail_n64','tail_zero_kernel_v2_n64','tail_full_kernel_v2_n64'):
        p=ROOT/'analysis'/(name+'_event_timeline.json')
        if p.exists():kernels.append(kernel_summary(name,json.loads(p.read_text())['rows']))
    aa=by['tail_zero_repeat_clean_n64']['metrics'];baseline=by['tail_zero_clean_n64']['metrics']
    kernel_screens=[]
    for name in ('tail_zero_kernel_v2_n64','tail_full_kernel_v2_n64'):
        if name not in by or by[name]['status']!='COMPLETE':continue
        arm='zero' if '_zero_' in name else 'full'
        baseline_clean=by[f'tail_{arm}_clean_n64']['metrics'];value=by[name]['metrics']
        ratios={k+'/'+s:value[k][s]/baseline_clean[k][s] for k,s in
            (('TTFT_ms','p95'),('TPOT_ms','p95'),('ITL_ms','p99'),('ITL_ms','maximum'))}
        kernel_screens.append(dict(name=name,ratios_to_clean=ratios,
            five_percent_screen=all(.95<=v<=1.05 for v in ratios.values())))
    write(ROOT/'analysis/attribution.json',dict(instrumentation=screens,kernel_instrumentation=kernel_screens,
        zero_AA={k:{s:aa[k][s]/baseline[k][s] for s in ('p95','p99','maximum')} for k in ('TTFT_ms','TPOT_ms','ITL_ms')},
        mean_observation_account=accounting,common_support_batch_reweighting=composition,kernel_tails=kernels,
        limitations='None of the accounts is an additive causal attribution of the original N128 p99. Specific SM, HBM, cache, and launch contributions remain NOT_IDENTIFIED without additional evidence.'))
    print(json.dumps(dict(screens=screens,mean_account=accounting,composition=composition,
        kernels=[dict(name=x['name'],status=x['status'],assigned=len(x.get('contained_decode_operations',[]))) for x in kernels if x])))


if __name__=='__main__':main()
