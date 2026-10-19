"""Persistent independent replicas. Requests never move after initial placement."""
import argparse
import os
import selectors
import subprocess
import time
from poc_depth_cdb_factorial.contract import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--trace',action='store_true')
    p.add_argument('--validate',action='store_true')
    args=p.parse_args()
    gpus=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    assert gpus in [[4],GPUS],gpus
    world=len(gpus)
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    workload=read(ROOT/'artifacts/workload_frozen.json')
    byid={r['id']:r for rows in [workload['speed'],workload['quality']] for r in rows}
    plan=read(args.plan)
    selector=selectors.DefaultSelector()
    processes=[]
    logs=[]
    buffers={r:b'' for r in range(world)}
    def send(rank,message):
        processes[rank].stdin.write(json.dumps(message)+'\n')
        processes[rank].stdin.flush()
    def receive(timeout=1):
        events=[]
        for key,_ in selector.select(timeout):
            data=os.read(key.fileobj.fileno(),65536)
            if not data:raise RuntimeError(f'Worker {key.data} stopped: {processes[key.data].poll()}')
            buffers[key.data]+=data
            while b'\n' in buffers[key.data]:
                line,buffers[key.data]=buffers[key.data].split(b'\n',1)
                if line.startswith(b'DP_EVENT '):events.append((key.data,json.loads(line[9:])))
        return events
    try:
        for rank,gpu in enumerate(gpus):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            log=open(out/f'worker_{rank}.log','x')
            logs.append(log)
            command=[PYTHON,'-m','poc_depth_cdb_factorial.scripts.worker','--output',str(out/f'rank_{rank}'),
                '--batch',str(plan.get('local_batch_cap',16))]
            if args.trace:command.append('--trace')
            if args.validate:command.append('--validate')
            worker=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=log,text=True,bufsize=1)
            processes.append(worker)
            selector.register(worker.stdout,selectors.EVENT_READ,rank)
        ready=set()
        while len(ready)<world:
            for rank,event in receive():
                if event['event']=='ready':ready.add(rank)
        write(out/'replicas.json',dict(mapping=dict(enumerate(gpus)),DP=world,TP=1,EP=1,PP=1,
            logical_device_per_process=0,model=MODEL_ID,q=.5,kv='last_exited'))
        summaries=[]
        for case in plan['cases']:
            assert case['arm']!='A0' or gpus==[4] or case['kind']=='smoke'
            # A0 comparisons otherwise run alone. All ranks receive identical control framing.
            origin=time.perf_counter()+.5
            for rank in range(world):send(rank,dict(op='case',case=case['name'],arm=case['arm'],
                quality=case['kind']=='quality',origin_s=origin))
            case_ready=set()
            while len(case_ready)<world:
                for rank,event in receive(.01):
                    if event['event']=='case_ready':case_ready.add(rank)
            while time.perf_counter()<origin:time.sleep(.001)
            first_submit=time.perf_counter()
            assignments=[]
            for rid in case['ids']:
                rank=case['placements'][rid]
                assert 0<=rank<world
                now=time.perf_counter()-origin
                send(rank,dict(op='request',row=byid[rid],cap=case['cap'],arrival_s=0.))
                assignments.append(dict(id=rid,rank=rank,assigned_s=now,arrival_s=0.))
            for rank in range(world):send(rank,dict(op='close'))
            completed=set()
            done=set()
            notifications=[]
            while len(done)<world:
                for rank,event in receive(.02):
                    notifications.append(dict(rank=rank,**event))
                    if event['event']=='done':completed.add(event['id'])
                    if event['event']=='case_done':done.add(rank)
            assert completed==set(case['ids'])
            record=dict(case=case,assignments=assignments,notifications=notifications,
                first_submitted_s=first_submit-origin,controller_wall_ms=(time.perf_counter()-first_submit)*1000,
                category='MEASURED_LIVE',trace=args.trace,validation=args.validate)
            write(out/(case['name']+'.json'),record)
            summaries.append(dict(case=case['name'],requests=len(completed),controller_wall_ms=record['controller_wall_ms']))
            write(out/'progress.json',summaries)
            print('CASE_COMPLETE',summaries[-1],flush=True)
        for rank in range(world):send(rank,dict(op='shutdown'))
        for process in processes:assert process.wait(timeout=60)==0
        write(out/'status.json',dict(success=True,cases=len(summaries),live_online=True,RL=False,migration=False))
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()

if __name__=='__main__':main()
