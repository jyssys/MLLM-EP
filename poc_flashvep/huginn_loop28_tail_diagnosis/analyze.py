"""Keep accounting identities separate from causal attribution."""
import argparse,csv,math,statistics
from collections import defaultdict,Counter
from .common import *


def percentile(values,p):
    if not values:return None
    values=sorted(values);x=(len(values)-1)*p;a=int(x);b=math.ceil(x)
    return values[a]+(values[b]-values[a])*(x-a)


def summary(values):
    values=list(values)
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
        p50=percentile(values,.5),p95=percentile(values,.95),p99=percentile(values,.99),
        maximum=max(values,default=None))


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def requests(folder):
    return [x for x in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if not x.get('warmup')]


def latency(folder):
    rows=requests(folder)
    itl=[1000*(b['time']-a['time'])/b['new_tokens'] for x in rows for a,b in zip(x['token_events'],x['token_events'][1:])]
    return dict(count=len(rows),TTFT_ms=summary(x['ttft_ms'] for x in rows),
        TPOT_ms=summary(x['tpot_ms'] for x in rows),ITL_ms=summary(itl),
        E2E_ms=summary(1000*(x['finish']-x['arrival']) for x in rows))


def old_analysis():
    initialize();out=[]
    names=['lifetime3_deferred2_zero_seed2026110940','priority4_normal_zero_seed2026110940',
        'priority4_normal_loop31_seed2026110940','quantum6_loop31_512_seed2026110940',
        'match18_loop28_full_seed2026110940','match18_loop28_tile256_seed2026110940']
    for seed in (2026110951,2026110952,2026110953):
        for arm in ('zero','loop','loopfour'):
            names.append(f'primaryconfirmv2_{arm}_seed{seed}'+('_isolationrepeat' if arm=='loopfour' and seed==2026110952 else ''))
    for name in names:
        folder=OLD_ROOT/'raw'/name
        cfg=json.loads((OLD_ROOT/'configs'/(name+'.json')).read_text())
        window=json.loads((folder/'window.json').read_text());lo=window['start']+.2*(window['last_arrival']-window['start']);hi=window['start']+.9*(window['last_arrival']-window['start'])
        histogram=Counter();graph={}
        for p in folder.glob('atomic_spans_*.json'):
            v=json.loads(p.read_text())
            if v['role']!='decode':continue
            for x in v['time_histogram']:
                if lo<=x['time'] and x['time']+.5<=hi and x['operation']=='recurrent' and x['logical_rows']==x['requests']:
                    histogram[x['requests']]+=x['submissions']
        for p in folder.glob('deep_graph_cache_*.json'):
            v=json.loads(p.read_text());graph[v['role']]=v['counts']['measured']
        req=requests(folder);strata=defaultdict(list)
        for x in req:
            for a,b in zip(x['token_events'],x['token_events'][1:]):
                live=sum(y['first_token']<=a['time']<y['finish'] for y in req)
                strata[live].append(1000*(b['time']-a['time'])/b['new_tokens'])
        out.append(dict(name=name,metrics=latency(folder),actual_decode_batch_histogram=dict(histogram),
            live_request_ITL={str(k):summary(v) for k,v in strata.items()},graph=graph,
            static_budget_active=cfg.get('intervention_method')!='native',
            configuration_sha256=digest(OLD_ROOT/'configs'/(name+'.json')),
            requests_sha256=digest(folder/'requests.jsonl')))
    write(ROOT/'analysis/existing_evidence.json',dict(rows=out,
        scope='Read-only original evidence. Live requests are not physical batch sizes. Half-second batch histograms do not identify individual steps.'))
    print('EXISTING_EVIDENCE_READ_ONLY',len(out))


def union_length(intervals):
    intervals=sorted((a,b) for a,b in intervals if b>a)
    result=[]
    for a,b in intervals:
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
        else:result.append((a,b))
    return sum(b-a for a,b in result)


def event_offsets(row,clock):
    if 'device_begin_offset_ms' in row:return row['device_begin_offset_ms'],row['device_end_offset_ms']
    start=clock['start']['midpoint'];scale=clock['elapsed_device_ms']/(clock['end']['midpoint']-start)
    return (row['begin']['time']-start)*scale,(row['end']['time']-start)*scale


def accounting(itl_ms,cycle_ms,model_ms):
    return dict(model_enclosure_ms=model_ms,between_enclosure_ms=cycle_ms-model_ms,
        output_observation_delta_ms=itl_ms-cycle_ms)


def inspect(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
        return dict(name=name,status='NOT_COMPLETE')
    req=requests(folder);expected={cfg['seed']+i for i in range(cfg['count'])}
    reference={x['seed']:x for x in requests(OLD_ROOT/'raw/lifetime3_deferred2_zero_seed2026110940')}
    exact=all(x['input_ids']==reference[x['seed']]['input_ids'] and x['output_ids']==reference[x['seed']]['output_ids'] for x in req)
    exits=[json.loads(p.read_text()) for p in folder.glob('native_worker_exit_*.json')]
    normal=any(v['passed'] and {x['role'] for x in v['records'] if x['exitcode']==0 and not x['forced_cleanup']}=={'prefill','decode'} for v in exits)
    result=dict(name=name,status='COMPLETE',metrics=latency(folder),exact_outputs=exact,normal_worker_exits=normal)
    traces=list(folder.glob('decode_diagnosis_*.json'))
    if not traces:return result
    trace=json.loads(traces[0].read_text());events=trace['rows'];prompt=[];by_request=defaultdict(list);operations=[]
    for index,row in enumerate(events):
        row['event_index']=index
        row['device_left_ms'],row['device_right_ms']=event_offsets(row,trace['clock'])
        if row['prompt']:prompt.append(row)
        else:
            for r in row['requests']:
                if r['seed'] in expected:by_request[r['seed'],r['generated'],row['operation']].append(row)
    for row in events:
        if row['prompt'] or row['operation']!='recurrent':continue
        intervals=[(max(row['device_left_ms'],p['device_left_ms']),min(row['device_right_ms'],p['device_right_ms'])) for p in prompt]
        overlap=union_length(intervals)
        operations.append(dict(event_index=row['event_index'],batch=len(row['requests']),
            device_enclosure_ms=row['device_enclosure_ms'],prompt_enclosure_overlap_ms=overlap,
            host_submit_ms=1000*(row['host_submit_end']-row['host_submit_begin']),
            host_thread_CPU_ms=row['host_submit_thread_CPU_ms'],stream=row['stream']))
    cycles=[];failures=[]
    for request in req:
        seed=request['seed']
        for g,(previous,current) in enumerate(zip(request['token_events'],request['token_events'][1:]),1):
            keys=[(seed,g-1,'coda'),(seed,g,'prelude'),(seed,g,'recurrent'),(seed,g,'coda')]
            if any(len(by_request[k])!=1 for k in keys):
                failures.append(dict(seed=seed,generated=g,coverage=[len(by_request[k]) for k in keys]));continue
            old,pre,core,coda=[by_request[k][0] for k in keys]
            itl=1000*(current['time']-previous['time']);cycle=coda['device_right_ms']-old['device_right_ms']
            model=sum(x['device_enclosure_ms'] for x in (pre,core,coda))
            parts=accounting(itl,cycle,model)
            assert abs(sum(parts.values())-itl)<1e-6
            cycles.append(dict(seed=seed,generated=g,ITL_ms=itl,batch=len(core['requests']),
                recurrent_enclosure_ms=core['device_enclosure_ms'],**parts,
                previous_output=previous['time'],next_output=current['time'],
                frozen_evaluation_deadline=previous['time']+.1928212538175284,
                deadline_scope='Post-hoc output-delivery threshold. Not an online admission deadline.'))
    costs=defaultdict(list)
    for row in operations:costs[f'B{row["batch"]}/'+('overlap' if row['prompt_enclosure_overlap_ms']>.01 else 'no_overlap')].append(row['device_enclosure_ms'])
    result.update(operation_cost_by_batch_and_enclosure_overlap={k:summary(v) for k,v in costs.items()},
        cycles=len(cycles),cycle_coverage_failures=failures,
        interval_accounting_ms={key:summary(x[key] for x in cycles) for key in
            ('ITL_ms','model_enclosure_ms','between_enclosure_ms','output_observation_delta_ms','recurrent_enclosure_ms')},
        prompt_quantum_ms=summary(x['device_enclosure_ms'] for x in prompt),
        phase_streams={op:sorted({x['stream'] for x in events if x['operation']==op}) for op in {x['operation'] for x in events}},
        clock=trace['clock'],scope='Operation CUDA Event enclosures include waits and submission gaps. Overlap is enclosure overlap, not proven concurrent kernels. Mean accounting is additive. Percentiles and causal classes are not additive.')
    csv_write(ROOT/'analysis'/(name+'_cycles.csv'),cycles)
    csv_write(ROOT/'analysis'/(name+'_operations.csv'),operations)
    write(ROOT/'analysis'/(name+'_event_timeline.json'),trace)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old',action='store_true');args=parser.parse_args()
    if args.old:return old_analysis()
    initialize();plan=json.loads((ROOT/'state/diagnostic_plan.json').read_text())
    extra=ROOT/'state/extra_diagnostic_plan.json'
    jobs=plan['jobs']+(json.loads(extra.read_text())['jobs'] if extra.exists() else [])
    more=ROOT/'state/kernel_v2_plan.json'
    if more.exists():jobs+=json.loads(more.read_text())['jobs']
    rows=[inspect(n) for n in jobs if n!='tail_serial_state8']
    write(ROOT/'analysis/new_diagnostics.json',rows)
    print(json.dumps([dict(name=x['name'],status=x['status'],cycles=x.get('cycles'),exact=x.get('exact_outputs')) for x in rows]))


if __name__=='__main__':main()
