"""Attribute trace-lane producer/readiness costs without inventing GPU CWT.

These clocks are host lifecycle timestamps and local CUDA *durations*.  A
DeepEP prepare interval can include waiting for a peer to produce its input;
it is not an isolated network service-time measurement.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json
from poc_native_lp_ep4.scripts.evidence import estimate


def records(run, pattern):
    return [json.loads(line) for path in run.glob(pattern) for line in path.open()]


def lifecycle_rows(run):
    manifest=read_json(run/'manifest.json')
    lifecycle=records(run,'lifecycle*.jsonl')
    native=records(run,'native*.jsonl')
    entries=records(run,'model_entry*.jsonl')
    outputs=read_json(run/'worker.dp0.json')['waves']
    by=defaultdict(list)
    for e in lifecycle+native+entries:
        if e.get('wave') is not None:by[e['wave']].append(e)
    result=[]
    for wave in outputs:
        if wave['repeat']<0 or wave['arm']=='DECODE_ONLY':continue
        es=by[wave['wave']]
        def first(kind):
            vs=[e['host_monotonic_ns'] for e in es if e.get('diagnostic')==kind]
            return min(vs) if vs else None
        received=first('scheduler_request_received')
        admitted=first('scheduler_first_admission')
        workers=first('worker_first_execution')
        dp0=[e['entry_ns'] for e in es if e.get('dp_rank')==0 and 'entry_ns' in e]
        lm=min(dp0) if dp0 else None
        vision=[e['cuda_ms'] for e in es if e.get('diagnostic')=='vision_encoder'
                and e.get('is_injection',False)]
        pipeline=[e for e in es if e.get('diagnostic')=='pre_lm_input_pipeline'
                  and e.get('is_injection',False) and e.get('native_stage',0)==0]
        later=[e for e in es if e.get('diagnostic')=='pre_lm_input_pipeline'
               and e.get('is_injection',False) and e.get('native_stage',0)>0]
        begin=wave['injection_begin_ns']
        row={k:wave[k] for k in ('wave','arm','modality','repeat')}
        row.update(run=str(run),lane='DIAGNOSTIC_TRACE_NOT_CLEAN_HEADLINE',
            instrumented=wave.get('instrumented',manifest['trace']),
            clock='HOST_MONOTONIC_LIFECYCLE_AND_LOCAL_CUDA_DURATIONS_NOT_CWT',
            arrival_to_received_ms=None if received is None else (received-begin)/1e6,
            received_to_admission_ms=None if received is None or admitted is None else (admitted-received)/1e6,
            admission_to_first_worker_ms=None if admitted is None or workers is None else (workers-admitted)/1e6,
            arrival_to_lm_entry_ms=None if lm is None else (lm-begin)/1e6,
            injection_vision_encoder_cuda_ms=max(vision) if vision else None,
            injection_vision_encoder_invocations=len(vision),
            initial_input_pipeline_host_ms=max((e['host_ms'] for e in pipeline),default=None),
            initial_input_pipeline_cuda_ms=max((e['cuda_ms'] for e in pipeline),default=None),
            later_input_pipeline_host_mean_ms=sum(e['host_ms'] for e in later)/len(later) if later else None,
            later_input_pipeline_cuda_mean_ms=sum(e['cuda_ms'] for e in later)/len(later) if later else None,
            later_input_pipeline_host_max_ms=max((e['host_ms'] for e in later),default=None),
            vision_cache_hits=sum(e.get('diagnostic')=='vision_cache_hit' for e in es))
        result.append(row)
    return result


def summarize_lifecycle(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['instrumented']:groups[(row['arm'],row['modality'])].append(row)
    fields=('arrival_to_received_ms','received_to_admission_ms','admission_to_first_worker_ms',
        'arrival_to_lm_entry_ms','injection_vision_encoder_cuda_ms',
        'initial_input_pipeline_host_ms','initial_input_pipeline_cuda_ms',
        'later_input_pipeline_host_mean_ms','later_input_pipeline_cuda_mean_ms','later_input_pipeline_host_max_ms')
    return [{'arm':arm,'modality':modality,'n':len(items),
             **{key:estimate([r[key] for r in items if r[key] is not None]) for key in fields}}
            for (arm,modality),items in groups.items()]


def dispatch_shape(rows):
    """Layer-0 versus other-layer diagnostic, not a wait-time decomposition."""
    groups=defaultdict(list)
    for r in rows:
        if r['prefill_active']:groups[(r['arm'],r['modality'],r['repeat'])].append(r)
    waves=[]
    for (arm,modality,repeat),items in groups.items():
        first=[r['dispatch_ms'] for r in items if r['layer']==0]
        other=[r['dispatch_ms'] for r in items if r['layer']!=0]
        waves.append({'arm':arm,'modality':modality,'repeat':repeat,
            'layer0_dispatch_max_ms':max(first) if first else None,
            'other_layer_dispatch_mean_ms':sum(other)/len(other) if other else None,
            'scope':'CRITICAL_LOCAL_ELAPSED_PROXY_INCLUDES_PRODUCER_WAIT; NOT_NETWORK_SERVICE'})
    return waves


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=lifecycle_rows(a.run)
    atomic_json(a.output,{'lifecycle_waves':rows,'lifecycle_summary':summarize_lifecycle(rows),
        'dispatch_shape':dispatch_shape(read_json(a.components)['trace_layers']),
        'causal_limit':'Durations alone do not identify network service versus peer readiness. No GPU CWT is emitted.'})


if __name__=='__main__':main()
