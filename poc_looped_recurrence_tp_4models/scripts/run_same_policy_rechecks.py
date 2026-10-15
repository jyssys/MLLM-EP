"""Repeat two calibration contrasts in new sessions with primary engine kwargs."""
import argparse
import dataclasses
import gzip
import os
import signal
import subprocess
import sys
import time
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read


def collect(name,output):
    reference=ROOT/'results/raw'/name
    state=read(reference/'status.json');model=state['model'];R=state['r'];TP=state['tp'];family=state['family']
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,GPUS[:TP]))
    os.environ.update(POC4_MODEL=model,POC4_FAMILY=family,POC4_R=str(R))
    kwargs=read(reference/'engine_contract.json')
    assert kwargs['worker_extension_cls']=='poc_looped_recurrence_tp_4models.worker_probe.FourModelProbe'
    assert kwargs['disable_log_stats'] is False and kwargs['enforce_eager'] is False
    from vllm import LLM,SamplingParams
    llm=LLM(**kwargs);output=Path(output)
    write(output/'engine_contract.json',kwargs);write(output/'ownership.json',llm.collective_rpc('inspect_four',timeout=120))
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:rows=[r for r in json.load(stream) if r['model']==model]
    def watchdog(signum,frame):raise TimeoutError('Recheck request exceeded 300 seconds')
    signal.signal(signal.SIGALRM,watchdog)
    def generate(row):
        llm.collective_rpc('inspect_memory',timeout=120,args=(True,))
        llm.collective_rpc('teacher',timeout=120,args=(row['teacher_ids'],))
        signal.alarm(300);start=time.perf_counter_ns()
        try:value=llm.generate([{'prompt_token_ids':row['prompt_ids']['1024']}],
            SamplingParams(temperature=0,max_tokens=64,ignore_eos=True),use_tqdm=False)[0]
        finally:signal.alarm(0)
        wall=(time.perf_counter_ns()-start)/1e6
        assert list(value.outputs[0].token_ids)==row['teacher_ids']
        llm.collective_rpc('inspect_memory',timeout=120)
        return dict(wall_ms=wall,metrics=dataclasses.asdict(value.metrics),output_ids=list(value.outputs[0].token_ids))
    for _ in range(3):generate(rows[0])
    records=[]
    for row in rows:
        for repeat in range(3):records.append(dict(request_id=row['request_id'],repeat=repeat,**generate(row)))
    write(output/'results.json',dict(config=name,model=model,R=R,TP=TP,family=family,results=records,
        scope='Independent model session. Exact primary engine kwargs and sampler. Same eight calibration requests. Not fresh held-out.'))
    write(output/'status.json',dict(success=True,config=name));os._exit(0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--collect');parser.add_argument('--output');args=parser.parse_args()
    if args.collect:collect(args.collect,args.output);return
    deadline=time.monotonic()+3600
    while True:
        progress=ROOT/'results/diagnostic_progress_v4.json'
        try:ready=progress.exists() and len(read(progress))==10 and all(r.get('exit_code')==0 for r in read(progress))
        except json.JSONDecodeError:ready=False
        if ready:break
        if time.monotonic()>deadline:raise TimeoutError('Do not overlap rechecks with unfinished diagnostics')
        time.sleep(20)
    records=[]
    for R,TP in [(4,1),(2,4)]:
        # Reverse the family order relative to the primary and retained-graph lanes.
        for family in ['tied','native']:
            name=f'ouro_{family}_R{R}_TP{TP}_v1';output=ROOT/'results/rechecks'/name
            assert not output.exists(),'Preserve any earlier attempt'
            command=[sys.executable,'-m','poc_looped_recurrence_tp_4models.scripts.gpu_session',
                '--gpus',','.join(map(str,GPUS[:TP])),'--timeout','1200','--output','results/rechecks/'+name,
                '--',PYTHON,'-m','poc_looped_recurrence_tp_4models.scripts.run_same_policy_rechecks',
                '--collect',name,'--output',str(output)]
            result=subprocess.run(command,cwd=PROJECT,check=False);records.append(dict(config=name,exit_code=result.returncode))
            write(ROOT/'results/recheck_progress.json',records);print(name,result.returncode,flush=True)
            if result.returncode:break
        if records[-1]['exit_code']:break


if __name__=='__main__':main()
