"""Separate observer lane. No service-time sum is called E2E latency."""
import gc
import time
from .common import write,guard
from .experiments import new_engine,inputs,generate
from ..ouro_budget_joint_cg.service_observer import ServiceObserver

def run_job(model,job,target):
    results=[]
    for P,N in job['shapes']:
        for D in (1,2,4):
            e=new_engine(model,N,D,job);ids=inputs(P,N)
            generate(e,ids,cap=8)
            control_before=generate(e,ids,cap=16)
            observer=ServiceObserver(e,dict(P=P,N=N,D=D,gpu=guard(),lane='HEAVY_OBSERVER'))
            original=e.scheduler.schedule;cpu=[]
            def schedule(**kw):
                start=time.perf_counter_ns();batch=original(**kw)
                cpu.append(dict(start_ns=start,end_ns=time.perf_counter_ns(),phase=None if batch is None else str(batch.phase)))
                return batch
            e.scheduler.schedule=schedule
            data=generate(e,ids,cap=16)
            events=observer.finish()
            item=dict(P=P,N=N,D=D,gpu=guard(),events=events,scheduler=cpu,data=data,
                meaning='Ready-stream service includes launch gaps. CPU/GPU intervals may overlap. Not clean E2E attribution.')
            e.scheduler.schedule=original;observer.close()
            del schedule,original,observer
            control_after=generate(e,ids,cap=16)
            def token_map(value):return {r['id']:r['tokens'] for r in value['requests']}
            item.update(control_before=control_before,control_after=control_after,
                observer_token_parity=token_map(control_before)==token_map(data)==token_map(control_after),
                observer_wall_ratio=(data['end']-data['start'])/((control_before['end']-control_before['start']+control_after['end']-control_after['start'])/2),
                wall_ratio_scope='Matched cap and prompts. Not pure instrumentation overhead if execution membership/output differs.')
            write(target/f'profile_P{P}_N{N}_D{D}.json',item);results.append(dict(P=P,N=N,D=D,events=len(events)))
            e.close();del e;gc.collect();__import__('torch').cuda.empty_cache()
    return dict(results=results)
