"""Four independent native CDB replicas with admission-only DP placement."""
import argparse
import json
import os
import random
import selectors
import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    rows=read(ROOT/'artifacts/discovery_frozen.json')['rows']
    byid={r['id']:r for r in rows}
    plan=read(args.plan)
    selector=selectors.DefaultSelector()
    processes=[]
    logs=[]
    buffers={i:b'' for i in range(4)}
    def send(rank,message):
        processes[rank].stdin.write(json.dumps(message)+'\n')
        processes[rank].stdin.flush()
    def receive(timeout=1):
        events=[]
        for key,_ in selector.select(timeout):
            data=os.read(key.fileobj.fileno(),65536)
            if not data:
                raise RuntimeError(f'Worker {key.data} stopped: {processes[key.data].poll()}')
            buffers[key.data]+=data
            while b'\n' in buffers[key.data]:
                line,buffers[key.data]=buffers[key.data].split(b'\n',1)
                if line.startswith(b'DP_EVENT '):events.append((key.data,json.loads(line[9:])))
        return events
    try:
        for rank,gpu in enumerate([4,5,6,7]):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            log=open(out/f'worker_{rank}.log','x')
            logs.append(log)
            command=[PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.live_worker',
                '--output',str(out/f'rank_{rank}'),'--batch',str(plan.get('local_batch_cap',16))]
            if args.trace:command.append('--trace')
            worker=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            processes.append(worker)
            selector.register(worker.stdout,selectors.EVENT_READ,rank)
        ready=set()
        while len(ready)<4:
            for rank,event in receive():
                if event['event']=='ready':ready.add(rank)
        write(out/'replicas.json',dict(ready=True,mapping={str(i):g for i,g in enumerate([4,5,6,7])},
            DP=4,TP=1,PP=1,gate=.5,kv='last_exited',live_online=True))
        summaries=[]
        for case in plan['cases']:
            origin=time.perf_counter()+.5
            for rank in range(4):send(rank,dict(op='case',case=case['name'],origin_s=origin))
            case_ready=set()
            while len(case_ready)<4:
                for rank,event in receive(.01):
                    if event['event']=='case_ready':case_ready.add(rank)
            requests=[byid[i] for i in case['ids']]
            count=len(requests)
            outstanding=[0]*4
            expected_work=[0.]*4
            assigned={}
            completed=set()
            assignments=[]
            notifications=[]
            launched=0
            rng=random.Random(case.get('seed',20261018))
            if case['arrival']=='poisson':
                offsets=[]
                value=0.
                for _ in requests:
                    offsets.append(value)
                    value+=rng.expovariate(case['rate_rps'])
            else:offsets=[0.]*count
            closed=False
            while len(completed)<count:
                now=time.perf_counter()-origin
                while launched<count and now>=offsets[launched] and (
                    case['arrival']!='closed_loop' or len(assigned)-len(completed)<case['concurrency']):
                    row=requests[launched]
                    policy=case['policy']
                    features=plan['causal_features'][row['id']]
                    estimate=features['expected_work_ms']
                    if policy=='round_robin':rank=launched%4
                    elif policy=='single_rank':rank=0
                    elif policy=='preassigned':rank=case['placements'][row['id']]
                    elif policy=='queue':rank=min(range(4),key=lambda i:(outstanding[i],i))
                    elif policy=='length':
                        rank=min(range(4),key=lambda i:(expected_work[i],outstanding[i],i))
                        estimate=features['expected_output_tokens']+features['prompt_tokens']*.05
                    elif policy=='strong_cost':
                        rank=min(range(4),key=lambda i:(expected_work[i]/plan['rank_rates'][i],outstanding[i],i))
                    else:raise ValueError(policy)
                    arrival=now if case['arrival']=='closed_loop' else offsets[launched]
                    # No future output length or exit trace appears in this record.
                    assignments.append(dict(id=row['id'],rank=rank,arrival_s=arrival,
                        assigned_s=now,estimate=estimate,observed_outstanding=list(outstanding)))
                    assigned[row['id']]=(rank,estimate)
                    outstanding[rank]+=1
                    expected_work[rank]+=estimate
                    send(rank,dict(op='request',row=row,cap=case.get('cap',256),arrival_s=arrival))
                    launched+=1
                if launched==count and not closed:
                    for rank in range(4):send(rank,dict(op='close'))
                    closed=True
                for rank,event in receive(.001):
                    notifications.append(dict(rank=rank,**event))
                    if event['event']=='done':
                        rid=event['id']
                        assert rid not in completed
                        completed.add(rid)
                        owner,estimate=assigned[rid]
                        assert owner==rank
                        outstanding[rank]-=1
                        expected_work[rank]-=estimate
            done=set()
            # Some rank case_done messages can arrive before the final request notification.
            done.update(n['rank'] for n in notifications if n['event']=='case_done')
            while len(done)<4:
                for rank,event in receive(.01):
                    notifications.append(dict(rank=rank,**event))
                    if event['event']=='case_done':done.add(rank)
            record=dict(case=case,assignments=assignments,notifications=notifications,
                wall_ms=(time.perf_counter()-origin)*1000,category='MEASURED_LIVE_DP4')
            write(out/(case['name']+'.json'),record)
            summaries.append(dict(case=case['name'],requests=count,wall_ms=record['wall_ms']))
            write(out/'progress.json',summaries)
            print('LIVE_CASE',summaries[-1],flush=True)
        for rank in range(4):send(rank,dict(op='shutdown'))
        for worker in processes:assert worker.wait(timeout=60)==0
        write(out/'status.json',dict(success=True,cases=len(summaries),RL=False,migration=False))
    finally:
        for worker in processes:
            if worker.poll() is None:worker.terminate()
        for worker in processes:
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
        for log in logs:log.close()

if __name__=='__main__':main()
