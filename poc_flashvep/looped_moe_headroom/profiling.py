"""Heavy trace and operational contention control. No headline profiler latency."""
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from .common import ROOT,write
from .worker import timed

def heavy(a,target,job):
    from .study import rows
    text=(rows(job)[0]['text']+'\n')*32
    ids=a.encode(text,512)
    for _ in range(5):
        a.forward(ids)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profile:
        for i in range(2):
            with torch.profiler.record_function(f'FULL_NATIVE_FORWARD_{i}'):
                a.forward(ids)
                torch.cuda.synchronize()
    profile.export_chrome_trace(str(target/'native_trace.json.gz'))
    records=[]
    for event in profile.key_averages():
        records.append(dict(name=event.key,count=event.count,self_cpu_us=event.self_cpu_time_total,
            self_device_us=getattr(event,'self_device_time_total',None),
            evidence='PROFILED_ACCUMULATED_SERVICE_TIME_NOT_CLEAN_E2E_SHARE'))
    write(target/'operator_service.json',dict(model=a.name,forwards=2,input_length=512,operators=records))
    with gzip.open(target/'native_trace.json.gz','rt') as f:
        trace=json.load(f)
    events=trace['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e]
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    union=0
    if intervals:
        start,end=intervals[0]
        for s,e in intervals[1:]:
            if s>end:
                union+=end-start
                start,end=s,e
            else:
                end=max(end,e)
        union+=end-start
    summary=dict(model=a.name,kernel_events=len(kernels),gpu_kernel_union_us=union,
        observed_first_to_last_kernel_us=intervals[-1][1]-intervals[0][0] if intervals else None,
        semantics='Heavy trace domain only. Union removes kernel overlap. Gaps include host, copies, and synchronization, not certified idle.',
        evidence='PROFILED_GPU_TIMELINE_NOT_CLEAN_TIMING')
    write(target/'trace_summary.json',summary)
    text_reference='The capital of France is'
    native_ids=a.encode(text_reference)
    wrapped=a.generate(native_ids,8,stop_eos=False)
    if a.name.startswith('ifm_'):
        from xllm.paper_part1.native_inference import generate_native
        official=generate_native(a.model,a.tokenizer,[text_reference],max_prompt_len=128,max_gen_len=8,use_sampling=False)[0]
    else:
        official=a.model.generate(input_ids=native_ids,max_new_tokens=8,do_sample=False,use_cache=False,
            eos_token_id=None,forced_eos_token_id=None)[0,native_ids.shape[-1]:].tolist()
    assert official==wrapped['ids'],'Official generation API differs from the adapter'
    write(target/'official_generation_api_parity.json',dict(model=a.name,prompt=text_reference,
        official_ids=official,adapter_ids=wrapped['ids'],pass_exact=True,cap=8,eos_disabled=True))
    from .timing import generation_components
    small=ids[:,:128]
    a.generate(small,16,stop_eos=False)
    clean=a.generate(small,16,stop_eos=False)
    write(target/'generation_clean_control.json',clean)
    generation_components(a,small,target,16)
    observed=json.loads((target/'generation_expert_events.json').read_text())
    assert observed['generated_ids']==clean['ids'],'Generation instrumentation changed tokens'
    if a.name.startswith('ifm_'):
        from .backend_diagnostic import run
        write(target/'backend_routing_diagnostic.json',run(a,ids[:,:32]))
    return dict(status='PASS' if kernels else 'PARTIAL_CPU_ONLY_NO_CUDA_EVENTS',**summary)

def contention(a,target,job):
    from .study import rows
    ids=a.encode((rows(job)[0]['text']+'\n')*32,512)
    for _ in range(5):
        a.forward(ids)
    torch.cuda.synchronize()
    barrier=ROOT/'state'/job['barrier']
    barrier.mkdir(parents=True,exist_ok=True)
    write(barrier/f"ready_{job['id']}.json",dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],time=time.time()))
    until=time.monotonic()+300
    while len(list(barrier.glob('ready_*.json')))<job['participants']:
        assert time.monotonic()<until,'Contention barrier timeout'
        time.sleep(.1)
    lock=(barrier/'release.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (barrier/'release.json').exists():
            write(barrier/'release.json',dict(time=time.time()+2))
        release=json.loads((barrier/'release.json').read_text())['time']
    while time.time()<release:
        time.sleep(.01)
    values=timed(lambda:a.forward(ids),20,5)
    hardware=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=index,uuid,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used','--format=csv'],text=True)
    write(target/'contention.json',dict(model=a.name,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],participants=job['participants'],
        wall_ms=values,hardware=hardware,host_loadavg=os.getloadavg(),evidence='LIVE_INDEPENDENT_SINGLE_GPU_CONTENTION_CONTROL'))
    return dict(status='PASS',model=a.name,participants=job['participants'])
