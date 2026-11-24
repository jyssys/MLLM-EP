"""Isolated real-prefix HTTP and CUDA-event enclosures, not suffix or speedup."""
import asyncio
import csv
import json
import random
import time

import numpy as np

from .common import ROOT,digest,write,experiment_deadline
from .llm_probe_control import rpc


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
    event_rows=[]
    folder=ROOT/'raw/gpu_enclosures'/name/'pd'
    with (folder/'forward_enclosures.csv').open() as stream:
        for value in csv.DictReader(stream):
            assert int(value['scheduled_requests'])==1 and int(value['scheduled_tokens']) in quantums
            event_rows.append(dict(arm=arm,quantum=int(value['scheduled_tokens']),
                GPU_enclosure_ms=float(value['GPU_enclosure_ms']),
                evidence='CUDA-event enclosure, not exclusive kernel cost'))
    assert len(event_rows)==45
    write(ROOT/'analysis'/f'{name}_cost.json',dict(time=time.time(),plan_SHA256=digest(plan),
        evidence='DIAGNOSTIC' if not all(r['observer_gate_passed'] for r in comparisons) else 'ISOLATED_GPU_MEASUREMENT',
        observer_gate_passed=all(r['observer_gate_passed'] for r in comparisons),comparisons=comparisons,
        actual_batch=1,independent_starts=1,events=event_rows,
        task_completion=False,suffix_continuation=False,online_interference=False,
        scope='Original same-input prefix execution and original online E. No imported prefix KV, task quality, '
              'sustainable throughput, safe quantum or serving speedup claim.'))
    print('PREFIX_COST_QUALIFICATION',arm,all(r['observer_gate_passed'] for r in comparisons),flush=True)
