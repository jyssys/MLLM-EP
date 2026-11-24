"""Read serial P/D kernel controls without assigning online request costs."""
import collections
import bisect
import csv
import json
import math
import time
import numpy as np
from .common import ROOT, REPO, write, digest, preserve
from .runtime import bind
from .reproduction import records
from .failure_analysis import save_csv
from ..dense_ar_p_critical_path.critical_analysis import Timeline
from ..dense_ar_p_critical_path.kernel_analysis import analyze_activity
from ..dense_ar_output_aware_v2.statistics import paired_difference, ratio


def partition(groups, left, right):
    """Partition a clipped enclosure once, including cross-category overlap."""
    left,right=int(round(left)),int(round(right))
    edges=collections.defaultdict(lambda:collections.defaultdict(int))
    for kind, spans in groups.items():
        for a,b in spans:
            a,b=max(a,left),min(b,right)
            if a>=b:continue
            edges[a][kind]+=1
            edges[b][kind]-=1
    active=collections.defaultdict(int)
    result=collections.defaultdict(float)
    cursor=left
    for stamp, changes in sorted(edges.items()):
        kinds=sorted(kind for kind,count in active.items() if count>0)
        key=kinds[0] if len(kinds)==1 else 'CROSS_CATEGORY_OVERLAP' if kinds else 'NO_OBSERVED_KERNEL'
        result[key]+=stamp-cursor
        for kind,count in changes.items():active[kind]+=count
        cursor=stamp
    result['NO_OBSERVED_KERNEL']+=right-cursor
    assert sum(result.values())==right-left
    return dict(result)


def clock_offset(anchor):
    return int(anchor['cupti_ns'])-(int(anchor['host_lower_ns'])+int(anchor['host_upper_ns']))//2


def mapped_host_ns(seconds,offset):
    # Add the epoch offset only after converting the small host clock to int.
    return int(round(seconds*1_000_000_000))+int(offset)


def starts_in_enclosure(starts,left,right):
    """Count CUDA kernel execution records, not host launch API calls."""
    return bisect.bisect_left(starts,right)-bisect.bisect_left(starts,left)


def analyze():
    bind();preserve()
    from .profiles import PLAN, cpc
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from ..dense_ar_output_aware_v2.work_contract import executed_prompt_work
    plan=ROOT/'jobs'/PLAN
    complete=ROOT/'state/serial_native_relative_profile_complete_v2.json'
    assert complete.exists(),'Wait for the frozen profiling schedule to finish'
    receipt=json.loads(complete.read_text())
    executed={(row['label'],row['start']):row for row in receipt['results']}
    results=[];requests=[];operators=[];sources={}
    for test in json.loads(plan.read_text())['tests']:
        status=executed.get((test['label'],test['start']),dict(status='NOT_RUN'))
        item=dict(test,status=status['status'],evidence='NOT_RUN')
        if status['status']!='COMPLETE':
            item.update(error=status.get('error'),evidence='FAILED' if status['status']=='FAILED' else 'NOT_RUN')
            results.append(item);continue
        gate=cpc(test)
        assert gate==status['CPC'],'The CPC result changed during independent analysis'
        qualified=gate['passed']
        item.update(CPC=gate,evidence='ISOLATED_KERNEL_MEASUREMENT' if qualified else 'DIAGNOSTIC')
        if not all(value.get('passed') for value in gate['counters'].values()):
            item['reason']='COLLECTOR_COUNTER_GATE_FAILED. Do not parse incomplete activity as measured cost.'
            results.append(item);continue
        folder=ROOT/'raw'/test['jobs'][1]
        cfg=json.loads((folder/'configuration.json').read_text())
        outputs=records(folder/'requests.jsonl')
        handoffs={r['request_id']:r for r in records(folder/'handoffs.jsonl')}
        submits={(r['request_id'],r['role']):r['time'] for r in records(folder/'request_submissions.jsonl')}
        observed=[]
        for role in ('prefill','decode'):
            activity=analyze_activity(folder/('activity_'+role))
            roi=activity['ROI']
            offsets=[clock_offset(roi[key]) for key in ('left','right')]
            uncertainty=max(roi[key]['host_upper_ns']-roi[key]['host_lower_ns'] for key in ('left','right'))+abs(offsets[1]-offsets[0])+2
            busy=Timeline(activity['intervals'])
            categories={key:Timeline(spans) for key,spans in activity['category_intervals'].items()}
            with (folder/('activity_'+role)/'kernels.csv').open(newline='') as stream:
                kernel_starts=sorted(int(record['start_ns']) for record in csv.DictReader(stream)
                    if record['kind']=='kernel')
            for row in outputs:
                rid='q'+str(row['sample_id'])
                events=row['token_events']
                phases={'P_arrival_to_ready':(row['arrival'],handoffs[rid]['source_ready'])} if role=='prefill' else {
                    'D_submit_to_first_worker':(submits[rid,'decode'],events[0]['worker_time']),
                    'D_subsequent_decode':(events[0]['worker_time'],events[-1]['worker_time'])}
                for phase,(left,right) in phases.items():
                    a,b=mapped_host_ns(left,offsets[0]),mapped_host_ns(right,offsets[0])
                    assert roi['left']['cupti_ns']<=a<=b<=roi['right']['cupti_ns']
                    kernel=busy.area(a,b)/1e6
                    exclusive=partition(activity['category_intervals'],a,b)
                    assert abs(sum(value for key,value in exclusive.items() if key!='NO_OBSERVED_KERNEL')/1e6-kernel)<1e-3
                    value=dict(label=test['label'],start=test['start'],name=test['jobs'][1],request_id=rid,
                        prompt_id=row['prompt_id'],input_tokens=row['input_tokens'],output_tokens=row['output_tokens'],
                        role=role,phase=phase,host_enclosure_ms=1000*(right-left),GPU_kernel_union_ms=kernel,
                        CUDA_kernel_records_starting_in_enclosure=starts_in_enclosure(kernel_starts,a,b),
                        no_observed_kernel_ms=1000*(right-left)-kernel,
                        category_union_ms={key:timeline.area(a,b)/1e6 for key,timeline in categories.items()},
                        exclusive_partition_ms={key:value/1e6 for key,value in exclusive.items()},
                        clock_mapping_uncertainty_ns=uncertainty+max(math.ulp(left),math.ulp(right))*1e9,
                        evidence=item['evidence'])
                    observed.append(value)
                    operators.extend(dict(label=test['label'],start=test['start'],name=test['jobs'][1],
                        request_id=rid,role=role,phase=phase,category=key,ms=value/1e6,evidence=item['evidence'])
                        for key,value in exclusive.items())
            summary=dict(role=role,kernel_count=activity['kernel_count'],counters=activity['counters'],
                clock_mapping_uncertainty_ns=uncertainty,
                ROI_exclusive_partition_ns=activity['exclusive_time_partition_ns'],
                top_kernels=sorted(activity['per_kernel'].items(),key=lambda pair:-pair[1]['summed_ns'])[:30])
            item.setdefault('role_activity',[]).append(summary)
        phases=[]
        for phase in sorted({row['phase'] for row in observed}):
            selected=[row for row in observed if row['phase']==phase]
            kinds=sorted({key for row in selected for key in row['exclusive_partition_ms']})
            phases.append(dict(phase=phase,count=len(selected),
                **{field:float(np.mean([row[field] for row in selected])) for field in
                    ('host_enclosure_ms','GPU_kernel_union_ms','no_observed_kernel_ms',
                    'CUDA_kernel_records_starting_in_enclosure')},
                exclusive_partition_mean_ms={key:float(np.mean([row['exclusive_partition_ms'].get(key,0.) for row in selected])) for key in kinds}))
        item.update(phase_means=phases,metrics=summarize(test['jobs'][1]),
            actual_executed_work=executed_prompt_work(cfg,folder,json.loads((ROOT/'data'/cfg['dataset']).read_text())),
            GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'],seed=cfg['seed'],
            source_SHA256=json.loads((folder/'dispatcher_result.json').read_text())['source_SHA256'])
        requests.extend(observed);results.append(item)
        for filename in ('configuration.json','dispatcher_result.json','requests.jsonl','handoffs.jsonl',
                'activity_prefill/actual_ROI.json','activity_decode/actual_ROI.json'):
            path=folder/filename;sources[str(path.relative_to(REPO))]=digest(path)
    comparisons=[]
    qualified={(row['label'],row['start']):row for row in results if row.get('CPC',{}).get('passed')}
    for reference in ('NativeStrong','StockToken'):
        for label in ('StockToken','Layer','Hybrid'):
            if label==reference:continue
            starts=sorted(index for (name,index) in qualified if name==label and (reference,index) in qualified)
            if not starts:continue
            values=[]
            for index in starts:
                pair=[qualified[key,index] for key in (label,reference)]
                phases=[{row['phase']:row for row in item['phase_means']} for item in pair]
                fields={phase+'_'+field:[p[phase][field] for p in phases]
                    for phase in phases[0] for field in ('host_enclosure_ms','GPU_kernel_union_ms','no_observed_kernel_ms')}
                fields['P_plus_D_first_kernel_ms']=[sum(p[phase]['GPU_kernel_union_ms'] for phase in
                    ('P_arrival_to_ready','D_submit_to_first_worker')) for p in phases]
                values.append(fields)
            comparisons.append(dict(label=label,reference=reference,qualified_paired_starts=len(starts),start_indices=starts,
                fields={key:dict(delta_ms=paired_difference([v[key][0] for v in values],[v[key][1] for v in values]),
                    ratio=ratio([v[key][0] for v in values],[v[key][1] for v in values])) for key in values[0]}))
    value=dict(time=time.time(),passed=True,plan_SHA256=digest(plan),complete_receipt_SHA256=digest(complete),
        results=results,comparisons=comparisons,source_files_SHA256=sources,
        scope='Serial32 original serving APIs. One request is alive. Actual Batch1/1/1 and exactly16 generated tokens. '
            'CPC failures remain diagnostic. Qualified starts fewer than3 have no start-level confidence interval. '
            'D-first includes continuation and first-output preparation. P+D-first kernel sum is not TTFT. '
            'No-observed-kernel time does not identify physical idle,DMA,host wait or stream wait. '
            'Kernel names identify broad categories,not Q/K/V projection subtypes. Cross-category overlap stays separate. '
            'Kernel record counts use CUDA execution starts inside each half-open host enclosure. They are not host launch API counts or launch overhead. '
            'Clock mapping uses integer nanoseconds. Its uncertainty estimate uses endpoint anchors and stored host-clock precision. '
            'These isolated costs do not assign exclusive GPU time to Main400 Natural dynamic-batch requests.')
    write(ROOT/'analysis/serial_native_relative_kernel_analysis_v2.json',value)
    for name,rows in (('serial_native_relative_per_start',results),('serial_native_relative_request_phases',requests),
            ('serial_native_relative_operator_partition',operators),('serial_native_relative_comparisons',comparisons)):
        save_csv(ROOT/'figures'/(name+'.csv'),rows)
    preserve();return value


if __name__=='__main__':
    value=analyze()
    print('SERIAL_NATIVE_RELATIVE_ANALYSIS',[(row['label'],row['start'],row['evidence']) for row in value['results']],flush=True)
