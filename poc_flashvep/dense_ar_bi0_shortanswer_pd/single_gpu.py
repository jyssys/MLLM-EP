"""Use the installed standard vLLM API with no model-math adapters."""
import json
import os
import statistics
import sys
import time
from .common import ROOT,MODEL,MODEL_REVISION,write


def main():
    cfg=json.load(open(sys.argv[1]));folder=ROOT/'raw'/cfg['name']
    assert os.environ['VLLM_BATCH_INVARIANT']==str(int(cfg['batch_invariant']))
    activity=None
    if cfg.get('kernel_activity'):
        from .activity import Activity
        activity=Activity(folder)
    import torch
    torch.set_num_threads(1)
    from vllm import LLM,SamplingParams
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.model_executor.layers.layernorm import RMSNorm
    replay=CudaGraphManager.run_fullgraph;counter=[0]
    def replayed(self,*a,**kw):
        counter[0]+=1;return replay(self,*a,**kw)
    CudaGraphManager.run_fullgraph=replayed
    options=dict(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',
        tensor_parallel_size=1,max_model_len=40960,max_num_batched_tokens=8192,
        max_num_seqs=1024,enable_prefix_caching=False,kv_cache_memory_bytes=42*1024**3,
        seed=2026111901,disable_log_stats=True,async_scheduling=cfg.get('async_scheduling',True),
        compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',
            cudagraph_capture_sizes=[1,2,4,8,16,32]))
    if cfg.get('stock_execution_defaults'):
        options.pop('compilation_config');options.pop('async_scheduling')
    engine=None;results=[]
    try:
        write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
        llm=LLM(**options);engine=llm.llm_engine;actual=engine.vllm_config
        write(folder/'actual_runtime.json',dict(model_revision=MODEL_REVISION,options=options,
            actual_scheduler_B=actual.scheduler_config.max_num_batched_tokens,
            actual_scheduler_S=actual.scheduler_config.max_num_seqs,
            actual_async_scheduling=actual.scheduler_config.async_scheduling,
            actual_cudagraph_mode=str(actual.compilation_config.cudagraph_mode),
            actual_compile_mode=str(actual.compilation_config.mode),
            model_forward_module=Qwen3ForCausalLM.forward.__module__,
            RMSNorm_forward_module=RMSNorm.forward_cuda.__module__,
            batch_invariant=cfg['batch_invariant'],
            boundary_sync=cfg.get('boundary_sync',False),
            scope='Standard installed vLLM API. No repository model, normalization, KV or sampling adapters. A read-only graph replay counter is installed. One GPU, not a two-GPU capacity baseline.'))
        for length in cfg.get('lengths',[4096,8192,16384]):
            pool=json.loads((ROOT/'data'/f'diagnostic_L{length}.json').read_text())
            for index,row in enumerate(pool[:8]):
                rid=f'length{length}_{index}';events=[];steps=[];first=None;final=None
                params=SamplingParams(max_tokens=16,temperature=0,ignore_eos=True,seed=2026111901)
                before_replays=counter[0]
                torch.cuda.synchronize();started=time.monotonic()
                engine.add_request(rid,dict(prompt_token_ids=row['input_ids']['qwen8']),params)
                previous_end=None;decode_left=None;gpu_enclosures=[]
                while engine.has_unfinished_requests():
                    if cfg.get('CUDA_enclosure'):
                        gpu_begin=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True);gpu_begin.record()
                    begin=time.monotonic();outputs=engine.step();end=time.monotonic()
                    if cfg.get('CUDA_enclosure'):
                        gpu_end.record();gpu_enclosures.append((gpu_begin,gpu_end))
                    steps.append(dict(start=begin,end=end,wall_ms=1000*(end-begin),
                        prior_host_gap_ms=1000*(begin-previous_end) if previous_end else None))
                    previous_end=end
                    for output in outputs:
                        ids=list(output.outputs[0].token_ids)
                        if ids and first is None:
                            if cfg.get('boundary_sync'):torch.cuda.synchronize()
                            first=time.monotonic()
                            if activity:decode_left=activity.stamp()
                        if ids:events.append(dict(time=time.monotonic(),count=len(ids)))
                        if output.finished:final=ids
                torch.cuda.synchronize();finished=time.monotonic()
                decode_right=activity.stamp() if activity else None
                assert final is not None and len(final)==16 and len(events)==16
                assert [r['count'] for r in events]==list(range(1,17))
                intervals=[1000*(b['time']-a['time']) for a,b in zip(events,events[1:])]
                result=dict(input_tokens=length,prompt_id=row['prompt_id'],warmup=index<2,
                    output_ids=final,TTFT_ms=1000*(events[0]['time']-started),
                    first_to_last_token_ms=1000*(events[-1]['time']-events[0]['time']),
                    legacy_prepared_decode_end_minus_first_ms=1000*(finished-first),
                    mean_TBT_ms=statistics.mean(intervals),intervals_ms=intervals,
                    token_events=events,engine_steps=steps,actual_engine_step_calls=len(steps),
                    graph_replays=counter[0]-before_replays,
                    decode_kernel_ROI=dict(left=decode_left,right=decode_right) if activity else None,
                    CUDA_step_enclosures_ms=[a.elapsed_time(b) for a,b in gpu_enclosures],
                    enclosure_scope='Stream-event enclosure includes waits and submission gaps. Not exclusive GPU kernel time.')
                results.append(result)
                print('SINGLE_GPU_REQUEST',length,index,result['TTFT_ms'],result['mean_TBT_ms'],result['graph_replays'],flush=True)
        write(folder/'single_gpu_results.json',dict(rows=results,model_revision=MODEL_REVISION,
            batch_invariant=cfg['batch_invariant'],measured_requests_per_length=6,warmup_per_length=2,
            boundary_sync=cfg.get('boundary_sync',False),
            evidence='MEASURED_DIAGNOSTIC_REFERENCE',
            scope='One independent server start. Sequential concurrency1. Different lengths share this start. Six requests are not six independent starts. Kernel-exclusive time is not measured by these wall clocks. Fixed output16 is diagnostic, not the real-workload main benchmark.'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if activity:activity.close()
        if engine is not None:engine.engine_core.shutdown()


if __name__=='__main__':main()
