"""Read actual CUDA intervals. Never infer SM utilization from busy time."""
import json
import re
import sqlite3
import statistics
import subprocess
from .common import ROOT,write,digest


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end<=start:
            continue
        if result and start<=result[-1][1]:
            result[-1][1]=max(result[-1][1],end)
        else:
            result.append([start,end])
    return result


def category(name):
    if re.search(r'flash|fmha|attention',name,re.I):
        return 'Attention'
    if re.search(r'gemm|nvjet|cutlass|matmul|cublas',name,re.I):
        return 'GEMM'
    if re.search(r'norm',name,re.I):
        return 'Norm'
    if re.search(r'activation|silu',name,re.I):
        return 'Activation'
    return 'Other'


def extract(name,role):
    folder=ROOT/'raw'/name
    report=folder/('nsys_'+role+'.nsys-rep')
    database=folder/('nsys_'+role+'.sqlite')
    assert report.exists(),'No actual Nsight report: '+str(report)
    if not database.exists():
        command=['/usr/local/bin/nsys','export','--type=sqlite',
            '--output='+str(database),str(report)]
        result=subprocess.run(command,capture_output=True,text=True)
        write(folder/('nsys_export_'+role+'.json'),dict(command=command,
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        assert result.returncode==0,result.stderr[-2000:]
    connection=sqlite3.connect(database)
    connection.row_factory=sqlite3.Row
    tables={r[0] for r in connection.execute('SELECT name FROM sqlite_master WHERE type="table"')}
    assert 'CUPTI_ACTIVITY_KIND_KERNEL' in tables and 'NVTX_EVENTS' in tables,tables
    strings={r['id']:r['value'] for r in connection.execute('SELECT * FROM StringIds')}
    markers=[];labels=[]
    for event in connection.execute('SELECT * FROM NVTX_EVENTS'):
        row=dict(event)
        text=row.get('text') or strings.get(row.get('textId'),'')
        match=re.search(r'ELASTIC_MONO_START_([0-9.]+)',text)
        if match:
            markers.append(float(match[1])-row['start']/1e9)
        if text.startswith('MATCHED_MODEL_SUBMISSION|'):
            labels.append(dict(start_ns=row['start'],end_ns=row.get('end'),label=text))
    assert markers,'No host/GPU-trace clock anchor'
    offset=statistics.median(markers)
    clock=json.loads((folder/f'nsys_clock_{role}.json').read_text())
    assert clock['start'] is not None and clock['end'] is not None,clock
    capture=[clock['start']['after'],clock['end']['before']]
    kernels=[]
    for event in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        row=dict(event)
        key=row.get('demangledName',row.get('shortName'))
        title=strings.get(key,str(key))
        start,end=row['start']/1e9+offset,row['end']/1e9+offset
        if end>=capture[0] and start<=capture[1]:
            kernels.append(dict(start=start,end=end,name=title,category=category(title),
                stream=row.get('streamId'),device=row.get('deviceId'),
                correlation_id=row.get('correlationId')))
    connection.close()
    assert kernels,'No CUDA kernels in bounded capture'
    return dict(role=role,capture_monotonic=capture,clock_offset_seconds=offset,
        anchor_spread_ms=1000*(max(markers)-min(markers)),
        start_host_anchor_bracket_ms=1000*(clock['start']['after']-clock['start']['before']),
        report_SHA256=digest(report),kernel_intervals=kernels,
        labels=[dict(start=r['start_ns']/1e9+offset,
            end=r['end_ns']/1e9+offset if r['end_ns'] else None,label=r['label']) for r in labels])


def analyze(name,observer_passed=False):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    roles=('replica0','replica1') if cfg['arm']=='DP2' else ('prefill','decode')
    outputs={}
    errors={}
    for role in roles:
        try:
            outputs[role]=extract(name,role)
        except BaseException as error:
            errors[role]=repr(error)
    batches={}
    for role in roles:
        observed=[]
        for path in folder.glob('extension_'+role+'_*.jsonl'):
            for line in path.read_text().splitlines():
                event=json.loads(line)
                if event.get('event')!='MODEL_INPUT_READY':continue
                ids=event['request_ids']
                if not ids or not all(rid.split('-')[0].startswith('q') for rid in ids):continue
                n=len(ids)
                flags=event['prefilling'][:n]
                observed.append(dict(time=event['time'],request_ids=ids,
                    query_lengths=event['query_lengths'][:n],computed_tokens=event['computed_tokens'][:n],
                    prefill_requests=sum(flags),decode_requests=n-sum(flags),query_tokens=event['tokens']))
        batches[role]=dict(rows=observed,actual_observed_steps=len(observed),
            actual_prefill_Decode_mixed_steps=sum(r['prefill_requests']>0 and r['decode_requests']>0 for r in observed),
            scope='Actual prepared input arrays from this diagnostic run. Native D can flag the first cached new-token query as prefilling. That query is Full36 decode work, not prompt replay. Counts are not inferred from NVML or kernel names. Observer qualification is separate.')
    result=dict(name=name,roles=outputs,errors=errors,actual_input_batches=batches,
        observer_gate=observer_passed,
        evidence='TRACE_ANALYSIS' if observer_passed and not errors else 'DIAGNOSTIC',
        SM_counters='NOT_MEASURED',HBM_counters='NOT_MEASURED',
        scope='Actual same-run CUDA kernel intervals. Busy means any kernel, not compute saturation. Kernel-name categories are descriptive. Concurrent kernel durations must not be added as an exclusive critical path. Failed observer gates prohibit clean serving busy/idle claims.')
    if len(outputs)==2:
        window=json.loads((folder/'window.json').read_text())
        lo=max(window['start'],max(r['capture_monotonic'][0] for r in outputs.values()))
        hi=min(window['end'],min(r['capture_monotonic'][1] for r in outputs.values()))
        assert hi>lo
        result['same_window_monotonic']=[lo,hi]
        for role,data in outputs.items():
            spans=union([(max(lo,r['start']),min(hi,r['end'])) for r in data['kernel_intervals']])
            busy=sum(b-a for a,b in spans)
            gaps=[];cursor=lo
            for start,end in spans:
                if start>cursor:gaps.append([cursor,start])
                cursor=end
            if cursor<hi:gaps.append([cursor,hi])
            data.update(common_window_busy_fraction=busy/(hi-lo),
                common_window_kernel_union=spans,actual_kernel_gaps=gaps,
                maximum_gap_ms=max([1000*(b-a) for a,b in gaps],default=0))
            grouped={}
            for event in data['kernel_intervals']:
                start,end=max(lo,event['start']),min(hi,event['end'])
                if end<=start:continue
                item=grouped.setdefault(event['category'],dict(count=0,summed_duration_ms=0.,intervals=[]))
                item['count']+=1;item['summed_duration_ms']+=1000*(end-start)
                item['intervals'].append((start,end))
            data['captured_category_times']={key:dict(kernel_count=item['count'],
                sum_of_kernel_durations_ms=item['summed_duration_ms'],
                category_union_ms=1000*sum(b-a for a,b in union(item['intervals'])),
                evidence=result['evidence'],
                scope='Only the common captured window. Concurrent kernel durations are not exclusive wall time. This is an instrumented-start observation, not clean serving cost.')
                for key,item in grouped.items()}
    write(ROOT/'analysis'/(name+'_kernel_intervals.json'),result)
    return result


if __name__=='__main__':
    gates=ROOT/'analysis/online_profiler_observer_gates.json'
    if gates.exists():
        for row in json.loads(gates.read_text()):
            analyze(row['names'][1],row['passed'])
