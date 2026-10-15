"""Benchmark independent official engines. No serving scheduler or placement method."""
import argparse
import dataclasses
import gzip
import multiprocessing as mp
import os
import time
from poc_looped_multigpu_kv_placement.contract import *


def actor(index,tp,commands,ready,results,start_gate,output):
    assigned=GPUS[index*tp:(index+1)*tp]
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,assigned))
    from vllm import LLM,SamplingParams
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:rows=[r for r in json.load(stream) if r['model']=='ouro']
    engine=LLM(model=str(model_path('ouro')),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=tp,
        seed=41015,max_model_len=32768,max_num_seqs=8,max_num_batched_tokens=32768,
        enable_prefix_caching=False,enable_chunked_prefill=False,gpu_memory_utilization=.85,
        disable_log_stats=False,async_scheduling=False,distributed_executor_backend='mp',
        worker_extension_cls='poc_looped_multigpu_kv_placement.worker_probe.KVWorkerProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    write(output/f'actor{index}.ownership.json',engine.collective_rpc('inspect_ownership',timeout=120))
    params=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True)
    ready.put(dict(actor=index,kind='loaded',mask=os.environ['CUDA_VISIBLE_DEVICES']))
    while True:
        command=commands.get();kind=command['kind']
        if kind=='stop':break
        if kind=='warmup':
            for _ in range(3):engine.generate([{'prompt_token_ids':rows[index]['prefix_token_ids'][str(command['prefix'])]}],params,use_tqdm=False)
            engine.collective_rpc('inspect_memory',timeout=120,args=(True,))
            ready.put(dict(actor=index,kind='warmup'));continue
        if kind=='memory':
            write(output/f'actor{index}.P{command["prefix"]}.memory.json',engine.collective_rpc('inspect_memory',timeout=120))
            ready.put(dict(actor=index,kind='memory'));continue
        assert kind=='batch'
        chosen=[rows[i] for i in command['indices']]
        prompts=[{'prompt_token_ids':row['prefix_token_ids'][str(command['prefix'])]} for row in chosen]
        ready.put(dict(actor=index,kind='batch'));start_gate.wait()
        start=time.perf_counter_ns();value=engine.generate(prompts,params,use_tqdm=False);end=time.perf_counter_ns()
        records=[dict(request_id=row['request_id'],output_ids=list(result.outputs[0].token_ids),
            metrics=dataclasses.asdict(result.metrics)) for row,result in zip(chosen,value)]
        assert all(len(row['output_ids'])==64 for row in records)
        results.put(dict(actor=index,start_ns=start,end_ns=end,requests=records))
    os._exit(0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tp',type=int,choices=[1,2],required=True)
    args=parser.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==MASK
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    context=mp.get_context('spawn');dp=4//args.tp
    queues=[context.Queue() for _ in range(dp)];ready=context.Queue();results=context.Queue();gate=context.Event()
    workers=[context.Process(target=actor,args=(i,args.tp,queues[i],ready,results,gate,output)) for i in range(dp)]
    for worker in workers:worker.start()
    def wait_ready(kind):
        messages=[ready.get(timeout=600) for _ in range(dp)]
        assert {row['actor'] for row in messages}==set(range(dp)) and all(row['kind']==kind for row in messages)
    wait_ready('loaded')
    for prefix in [1024,32704]:
        for queue in queues:queue.put(dict(kind='warmup',prefix=prefix))
        wait_ready('warmup')
        for concurrency in [4,8]:
            for offset in range(0,8,concurrency):
                indices=list(range(offset,offset+concurrency))
                for repeat in range(3):
                    for i,queue in enumerate(queues):queue.put(dict(kind='batch',prefix=prefix,indices=indices[i::dp]))
                    wait_ready('batch');start=time.perf_counter_ns();gate.set()
                    observed=[results.get(timeout=300) for _ in range(dp)]
                    end=time.perf_counter_ns();gate.clear()
                    write(output/f'P{prefix}.B{concurrency}.batch{offset}.repeat{repeat}.json',dict(
                        tp=args.tp,dp=dp,physical_GPUS=GPUS,prefix=prefix,concurrency=concurrency,repeat=repeat,
                        controller_wall_ms=(end-start)/1e6,aggregate_TPS=concurrency*64/((end-start)/1e9),actors=observed,
                        scope='Actual finite-cohort four-GPU independent native engines. Includes control/result IPC. Not a B1 speedup.'))
                    print(prefix,concurrency,offset,repeat,round((end-start)/1e6,3),flush=True)
        for queue in queues:queue.put(dict(kind='memory',prefix=prefix))
        wait_ready('memory')
    write(output/'status.json',dict(success=True,tp=args.tp,dp=dp))
    for queue in queues:queue.put(dict(kind='stop'))
    for worker in workers:worker.join(timeout=30)
    os._exit(0)


if __name__=='__main__':main()
