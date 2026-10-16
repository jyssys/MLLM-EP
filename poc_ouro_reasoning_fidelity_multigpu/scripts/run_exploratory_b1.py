"""Small natural-output continuity audit. Not the broad Stage P campaign."""
import argparse
import dataclasses
import os
import signal
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    args=parser.parse_args()
    expected=','.join(map(str,GPUS[:args.tp]))
    assert os.environ['CUDA_VISIBLE_DEVICES']==expected
    assert (ROOT/'results/quality_decision.json').exists(), 'Review quality before this fallback'
    decision=read(ROOT/'results/quality_decision.json')
    assert decision['broad_systems_authorized'] is False
    assert decision['small_exploratory_audit_authorized'] is True
    from vllm import LLM,SamplingParams
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=args.tp,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=32768,
        max_num_seqs=1,max_num_batched_tokens=32768,gpu_memory_utilization=.85,seed=41015,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=False,disable_log_stats=False,
        distributed_executor_backend='mp',
        worker_extension_cls='poc_looped_multigpu_kv_placement.worker_probe.KVWorkerProbe',
        compilation_config={'cudagraph_capture_sizes':[1]})
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    owners=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(owners)==args.tp and all(r['cache_slots']==192 for r in owners)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in owners}==set(UUIDS[:args.tp])
    write(output/'ownership.json',owners)
    rows=read(ROOT/'results/fidelity_completion_inputs.json')
    assert len(rows)==2
    def alarm(signum,frame):
        raise TimeoutError('Exploratory B1 request exceeded its 900-second watchdog')
    signal.signal(signal.SIGALRM,alarm)
    def generate(row,cap):
        params=SamplingParams(temperature=0.,top_p=1.,top_k=-1,seed=41015,
                              max_tokens=cap,ignore_eos=False,stop=[],repetition_penalty=1.)
        signal.alarm(900)
        start=time.perf_counter_ns()
        try:
            result=llm.generate([dict(prompt_token_ids=row['prompt_ids'])],params,use_tqdm=False)[0]
        finally:
            signal.alarm(0)
        wall=(time.perf_counter_ns()-start)/1e6
        assert list(result.prompt_token_ids)==row['prompt_ids']
        value=result.outputs[0]
        metrics=dataclasses.asdict(result.metrics)
        count=len(value.token_ids)
        return dict(id=row['id'],TP=args.tp,cap=cap,forced=False,temperature=0.,top_p=1.,
            prompt_ids=row['prompt_ids'],generated_ids=list(value.token_ids),text=value.text,
            prompt_tokens=len(row['prompt_ids']),generated_tokens=count,
            finish_reason=value.finish_reason,stop_reason=value.stop_reason,
            clean_call_wall_ms=wall,metrics=metrics,
            backend_queue_ms=1000*(metrics['scheduled_ts']-metrics['queued_ts']),
            backend_TTFT_ms=1000*(metrics['first_token_ts']-metrics['queued_ts']),
            backend_E2E_ms=1000*(metrics['last_token_ts']-metrics['queued_ts']),
            average_cached_TPOT_ms=None if count<2 else 1000*(metrics['last_token_ts']-metrics['first_token_ts'])/(count-1),
            output_TPS=count/(wall/1000),
            filled_KV_structural_bytes_per_rank=(len(row['prompt_ids'])+count-1)*1572864/args.tp,
            filled_KV_status='STRUCTURAL_PAYLOAD_ESTIMATE. Not an observed allocator occupancy.',
            scope='Exploratory B1 continuity, two independent MATH problems. Not author-style sampling accuracy.')
    for repeat in range(3):
        generate(rows[0],32)
    write(output/'memory_after_warmup.json',llm.collective_rpc('inspect_memory',args=(True,)))
    records=[]
    for repeat in range(3):
        for row in rows:
            record=generate(row,8192)
            record['repeat']=repeat
            records.append(record)
            write(output/'natural_requests.json',records)
    write(output/'memory_after_measurement.json',llm.collective_rpc('inspect_memory'))
    write(output/'status.json',dict(success=True,TP=args.tp,
        independent_problems=2,technical_repeats=3,heavy_observer=False,
        scope='Small fallback audit. Broad TP/DP profiling and method claims remain unauthorized.'))
    os._exit(0)


if __name__=='__main__':
    main()
