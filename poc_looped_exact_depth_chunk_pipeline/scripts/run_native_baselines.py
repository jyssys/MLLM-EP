"""Fresh TP4 and native chunk controls, or physical 48-layer stage costs."""
import argparse
import dataclasses
import os
import signal
import statistics
import time
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--r',type=int,required=True);parser.add_argument('--chunk',type=int,default=0)
    args=parser.parse_args();assert args.tp in [1,2,4] and args.r in [1,4]
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,GPUS[:args.tp]))
    os.environ.update(POC4_MODEL='ouro',POC4_FAMILY='native',POC4_R=str(args.r))
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    from vllm import LLM,SamplingParams
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=args.tp,
        hf_overrides={'total_ut_steps':args.r,'early_exit_threshold':1.},max_model_len=32768,max_num_seqs=8,
        max_num_batched_tokens=args.chunk or 32768,gpu_memory_utilization=.85,seed=41015,
        enable_prefix_caching=False,enable_chunked_prefill=bool(args.chunk),enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=False,disable_log_stats=False,distributed_executor_backend='mp',
        worker_extension_cls='poc_looped_exact_depth_chunk_pipeline.stage_probe.StageProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    write(output/'engine_contract.json',kwargs);llm=LLM(**kwargs)
    owned=llm.collective_rpc('inspect_four',timeout=120)
    assert len(owned)==args.tp and all(r['cache_slots']==48*args.r for r in owned)
    assert all(len({c['data_ptr'] for c in r['cache']})==48*args.r for r in owned),'Logical KV slots must not alias'
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in owned}==set(UUIDS[:args.tp])
    write(output/'ownership.json',owned)
    if args.tp==4 and args.r==4 and args.chunk==0:
        probes=[r for r in read(ROOT/'results/frozen_quality_questions.json') if r['dataset']=='aime24'][:2]
        for index,row in enumerate(probes):
            llm.collective_rpc('teacher',timeout=120,args=(None,))
            llm.collective_rpc('start_parity_capture',timeout=120)
            value=llm.generate([{'prompt_token_ids':row['prompt_ids']}],SamplingParams(temperature=0,max_tokens=4,ignore_eos=False),use_tqdm=False)[0]
            evidence=llm.collective_rpc('finish_parity_capture',timeout=120,args=(f'parity_reference_{index}',len(row['prompt_ids'])))
            write(output/f'parity_reference_{index}.json',dict(problem_id=row['id'],prompt_ids=row['prompt_ids'],
                output_ids=list(value.outputs[0].token_ids),evidence=evidence))
    rows=read(ROOT/'results/frozen_hardware.json')['requests'];count=64 if args.r==4 else 4
    def watchdog(signum,frame):raise TimeoutError('Hardware request exceeded 300 seconds')
    signal.signal(signal.SIGALRM,watchdog)
    def generate(row,P,N):
        llm.collective_rpc('teacher',timeout=120,args=(row['teacher_ids'][:N],))
        signal.alarm(300);start=time.perf_counter_ns()
        try:value=llm.generate([{'prompt_token_ids':row['prefix_token_ids'][str(P)]}],
            SamplingParams(temperature=0,max_tokens=N,ignore_eos=True),use_tqdm=False)[0]
        finally:signal.alarm(0)
        assert list(value.outputs[0].token_ids)==row['teacher_ids'][:N]
        return dict(wall_ms=(time.perf_counter_ns()-start)/1e6,metrics=dataclasses.asdict(value.metrics),
            output_ids=list(value.outputs[0].token_ids),scope='HARDWARE_ONLY. Forced output is not accuracy.')
    for P in [1024,8192,16384,32704]:
        selected=rows if P!=32704 else [r for r in rows if r['request_id'] in ['coding_00','math_00','document_00']]
        # Chunks >= P are equivalent unchunked geometry at this prompt length.
        if args.chunk and args.chunk>P and args.r==1:continue
        for _ in range(3):generate(selected[0],P,count)
        for row in selected:
            records=[]
            for repeat in range(3):
                llm.collective_rpc('inspect_memory',timeout=120,args=(True,))
                records.append(dict(repeat=repeat,**generate(row,P,count)))
            memory=llm.collective_rpc('inspect_memory',timeout=120)
            write(output/f'P{P}.{row["request_id"]}.clean.json',dict(P=P,request_id=row['request_id'],
                R=args.r,TP=args.tp,chunk=args.chunk,records=records,memory=memory))
        # Three metadata-selected requests give actual per-chunk local intervals.
        for row in [r for r in selected if r['request_id'].endswith('_00')]:
            controls=[generate(row,P,4)['wall_ms'] for _ in range(3)]
            llm.collective_rpc('stage_events_begin',timeout=120);observed=generate(row,P,4)
            trace=llm.collective_rpc('stage_events_finish',timeout=120)
            write(output/f'P{P}.{row["request_id"]}.stages.json',dict(P=P,request_id=row['request_id'],
                R=args.r,TP=args.tp,chunk=args.chunk,trace=trace,clean_wall_ms=controls,observed=observed,
                observer_percent=(observed['wall_ms']/statistics.mean(controls)-1)*100,
                limitation='R1 stage inputs represent first-depth activation statistics. Later-depth shape costs are a calibrated assumption.'))
        if args.tp==4 and args.r==4 and args.chunk==0 and P==16384:
            controls=[generate(selected[0],P,4)['wall_ms'] for _ in range(3)]
            llm.collective_rpc('runtime_counts_start',timeout=120)
            llm.collective_rpc('start_kernel_profile',timeout=120)
            profiled=generate(selected[0],P,4)
            traces=llm.collective_rpc('finish_kernel_profile',timeout=120)
            counts=llm.collective_rpc('runtime_counts_finish',timeout=120)
            write(output/'native_kernel_profile.json',dict(clean_ms=controls,profiled=profiled,traces=traces,counts=counts,
                observer_percent=(profiled['wall_ms']/statistics.mean(controls)-1)*100,
                scope='Separate native R4 execution-count diagnostic. Kernel service is not an additive E2E fraction.'))
        print('COMPLETE',args.tp,args.r,args.chunk,P,flush=True)
    write(output/'status.json',dict(success=True,**vars(args)))
    os._exit(0)


if __name__=='__main__':main()
