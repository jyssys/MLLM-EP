"""Isolated real-prefix HTTP and CUDA-event enclosures, not suffix or speedup."""
import asyncio
import csv
import json
import random
import time

import numpy as np

from .common import ROOT,digest,write,experiment_deadline
from .llm_probe_control import rpc


def prefix_events(folder,arm,quantums):
    event_rows=[];controls=[]
    geometry={r['index']:r['requests'] for r in json.loads((folder/'geometry.json').read_text())['rows']}
    with (folder/'forward_enclosures.csv').open() as stream:
        for value in csv.DictReader(stream):
            index=int(value['index']);requests=geometry[index]
            count=int(value['scheduled_requests']);tokens=int(value['scheduled_tokens'])
            assert len(requests)==count and sum(requests.values())==tokens
            if count==0:
                assert tokens==0 and int(value['encoder_items'])==0
                controls.append(dict(index=index,arm=arm,GPU_enclosure_ms=float(value['GPU_enclosure_ms']),
                    scope='Empty-schedule engine/control work. Preserved, not assigned to prefill.'))
                continue
            assert count==1 and tokens in quantums
            event_rows.append(dict(index=index,arm=arm,quantum=tokens,runtime_request_id=next(iter(requests)),
                GPU_enclosure_ms=float(value['GPU_enclosure_ms']),
                evidence='CUDA-event enclosure, not exclusive kernel cost'))
    assert len(geometry)==len(event_rows)+len(controls)
    return event_rows,controls


def recalculate_saved_cost(name,arm,recovered=False):
    plan=ROOT/'configs'/f'{name}.json';settings=json.loads(plan.read_text())
    comparisons=[]
    with (ROOT/'analysis'/f'{name}_qualification.csv').open() as stream:
        for row in csv.DictReader(stream):
            assert row['observer_gate_passed'] in ['True','False']
            comparisons.append(dict(arm=row['arm'],quantum=int(row['quantum']),HTTP_percentile=int(row['HTTP_percentile']),
                **{k:float(row[k]) for k in ['clean_before_ms','profile_ms','clean_after_ms',
                    'profile_relative_difference','clean_relative_spread']},
                observer_gate_passed=row['observer_gate_passed']=='True'))
    event_rows,controls=prefix_events(ROOT/'raw/gpu_enclosures'/name/'pd',arm,settings['quantum_tokens'])
    assert len(event_rows)==45 and len(comparisons)==10
    for quantum in settings['quantum_tokens']:
        assert sum(r['quantum']==quantum for r in event_rows)==9
    target=ROOT/'analysis'/f'{name}_cost.json';assert not target.exists()
    write(target,dict(time=time.time(),plan_SHA256=digest(plan),
        evidence='DIAGNOSTIC' if not all(r['observer_gate_passed'] for r in comparisons) else 'ISOLATED_GPU_MEASUREMENT',
        observer_gate_passed=all(r['observer_gate_passed'] for r in comparisons),comparisons=comparisons,
        actual_batch=1,independent_starts=1,events=event_rows,empty_schedule_control_events=controls,
        empty_schedule_control_total_enclosure_ms=sum(r['GPU_enclosure_ms'] for r in controls),
        recovered_from_closed_raw_without_GPU_rerun=recovered,
        task_completion=False,suffix_continuation=False,online_interference=False,
        scope='Original same-input prefix execution and original online E. No imported prefix KV, task quality, '
              'sustainable throughput, safe quantum or serving speedup claim.'))
    return all(r['observer_gate_passed'] for r in comparisons)


def measure(tag,arm,row,url,port,encoder_port):
    from .prefix_quantum_correctness_control import request
    name=f'{tag}_{arm}_CPC';quantums=[128,256,512,1024,2048];records=[];collector={}
    plan=ROOT/'configs'/f'{name}.json';assert not plan.exists()
    write(plan,dict(time=time.time(),evidence='ISOLATED_GPU_MEASUREMENT if observer gate passes, otherwise DIAGNOSTIC',
        request_id=row['request_id'],prompt_SHA256=row['prompt_SHA256'],full_input_tokens=row['total_input_tokens'],
        quantum_tokens=quantums,samples_per_quantum_per_stage=9,order_seed=2026112501,
        helper_KV_GiB=8,encoder_co_resident=arm=='E',actual_batch=1,B=8192,S=1024,
        observer_gate=.05,observer='Two CUDA events around original execute. No tensor CPU copies.',
        task_completion=False,suffix_KV_import=False,online_intervention=False,
        scope='HTTP includes original online E, embedding handoff, helper prefix, discarded sample and delivery. '
              'CUDA-event enclosure is not GPU-exclusive kernel cost. Prefix costs do not substitute for suffix costs.'))
    for stage in ['warmup','clean_before','profile','clean_after']:
        schedule=quantums if stage=='warmup' else quantums*9
        random.Random(2026112501).shuffle(schedule)
        opened=[]
        try:
            if stage=='profile':
                for label,worker_port in [('encoder',encoder_port),('pd',port)]:
                    collector[label]=dict(begin=rpc(worker_port,'gpu_enclosures/begin',
                        dict(name=name,state_interval_ms=50)))
                    opened.append((label,worker_port))
            for index,quantum in enumerate(schedule):
                assert time.time()<experiment_deadline()
                run=f'{name}_{stage}_q{quantum}_r{index}'
                value=asyncio.run(request(row,run,url,quantum,False))
                records.append(dict(value,arm=arm,stage=stage,evidence='DIAGNOSTIC until qualification'))
        finally:
            for label,worker_port in reversed(opened):
                collector[label]['end']=rpc(worker_port,'gpu_enclosures/end',{})
            if opened:write(ROOT/'state'/f'{name}_collector.json',collector)
        print('PREFIX_COST_STAGE_COMPLETE',arm,stage,len(schedule),flush=True)
    comparisons=[]
    for quantum in quantums:
        stages={stage:[r['HTTP_wall_ms'] for r in records if r['stage']==stage and r['quantum']==quantum]
            for stage in ['clean_before','profile','clean_after']}
        for percentile in [50,95]:
            a,b,p=(float(np.percentile(stages[s],percentile)) for s in ['clean_before','clean_after','profile'])
            center=(a+b)/2;delta=p/center-1;spread=abs(b-a)/center
            comparisons.append(dict(arm=arm,quantum=quantum,HTTP_percentile=percentile,
                clean_before_ms=a,profile_ms=p,clean_after_ms=b,
                profile_relative_difference=delta,clean_relative_spread=spread,
                observer_gate_passed=abs(delta)<=.05 and spread<=.05))
    with (ROOT/'analysis'/f'{name}_requests.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (ROOT/'analysis'/f'{name}_qualification.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    passed=recalculate_saved_cost(name,arm)
    print('PREFIX_COST_QUALIFICATION',arm,passed,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--recover',required=True)
    parser.add_argument('--arm',choices=['E','D'],required=True);options=parser.parse_args()
    assert options.recover.isidentifier() and options.recover.startswith('prefix_QUANTA_C26')
    print('RECOVERED_WITHOUT_GPU_RERUN',options.recover,recalculate_saved_cost(options.recover,options.arm,True))
