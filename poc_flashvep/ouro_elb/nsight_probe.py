"""CUPTI capture starts at the first real decode scheduling boundary."""
import time
import torch
from .common import write
from .experiments import new_engine,inputs,generate

def run_job(model,job,target):
    from vllm_rlt import SamplingParams
    e=new_engine(model,1,job['D'],job);ids=inputs(256,1)
    generate(e,ids,cap=8)
    original=e.scheduler.schedule;started=[False];markers=[]
    def schedule(**kwargs):
        if not started[0] and any(r.generated_token_ids for r in e.scheduler.requests.values()):
            torch.cuda.cudart().cudaProfilerStart()
            started[0]=True;markers.append(dict(host_s=time.perf_counter(),kind='CAPTURE_START_AFTER_FIRST_HOST_COMMIT'))
        return original(**kwargs)
    e.scheduler.schedule=schedule
    e.add_request('trace',ids[0],SamplingParams(max_tokens=16,max_loops=4,min_loops=4,exit_threshold=1.,temperature=0.,ignore_eos=True))
    outputs=[]
    while e.has_unfinished_requests():
        for out in e.step():outputs.append(dict(tokens=list(out.token_ids),depths=list(out.exit_depths),host_s=time.perf_counter()))
    e.model_runner.synchronize();torch.cuda.cudart().cudaProfilerStop()
    e.scheduler.schedule=original
    item=dict(D=job['D'],P=256,N=1,markers=markers,outputs=outputs,
        evidence='NSIGHT_HEAVY_TRACE_NOT_CLEAN_LATENCY',meaning='Capture begins after first output host commit, before frontend delivery. Initial queued GPU work may precede the first captured launch.')
    write(target/'nsight_outputs.json',item)
    e.close()
    return dict(captured=started[0],outputs=len(outputs))
