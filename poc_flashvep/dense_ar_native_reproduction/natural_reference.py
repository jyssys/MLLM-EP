"""Generate the frozen real workload once through the standard vLLM API."""
import json
import os
import sys
import time
from .common import ROOT,MODEL,MODEL_REVISION,write,digest
from .engine_options import options


def main():
    cfg=json.load(open(sys.argv[1]));folder=ROOT/'raw'/cfg['name']
    assert digest(ROOT/'data'/cfg['dataset'])==cfg['dataset_SHA256']
    import torch
    torch.set_num_threads(1)
    from vllm import LLM,SamplingParams
    engine=None;results=[]
    try:
        write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
        opts=options(cfg);llm=LLM(**opts);engine=llm.llm_engine
        runner=engine.model_executor.driver_worker.worker.model_runner
        actual=engine.vllm_config
        write(folder/'actual_runtime.json',dict(options=opts,
            actual_scheduler_B=actual.scheduler_config.max_num_batched_tokens,
            actual_scheduler_S=actual.scheduler_config.max_num_seqs,
            actual_chunked_prefill=actual.scheduler_config.enable_chunked_prefill,
            actual_cudagraph_mode=str(actual.compilation_config.cudagraph_mode),
            actual_compile_mode=str(actual.compilation_config.mode),
            GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
            KV_blocks=runner.kv_cache_config.num_blocks,KV_dtype=str(runner.kv_cache_dtype),
            KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),
            KV_layout=os.environ.get('VLLM_KV_CACHE_LAYOUT'),batch_invariant=cfg['batch_invariant'],
            model_revision=MODEL_REVISION))
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
        eos=json.loads((MODEL/'generation_config.json').read_text())['eos_token_id']
        def generate(rid,row,cap,ignore_eos):
            events=[];first=None;result=None;calls=0;started=time.monotonic()
            engine.add_request(rid,dict(prompt_token_ids=row['input_ids']['qwen8']),
                SamplingParams(max_tokens=cap,temperature=0,top_p=1,top_k=-1,
                    ignore_eos=ignore_eos,stop_token_ids=[] if ignore_eos else eos,seed=cfg.get('seed',2026111901)))
            while engine.has_unfinished_requests():
                outputs=engine.step();calls+=1
                for output in outputs:
                    value=output.outputs[0];ids=list(value.token_ids)
                    if ids:
                        now=time.monotonic();first=first or now
                        events.append(dict(time=now,count=len(ids)))
                    if output.finished:result=dict(output_ids=ids,finish_reason=value.finish_reason,stop_reason=value.stop_reason)
            assert result is not None and len(result['output_ids'])>0
            result.update(prompt_id=row['prompt_id'],task=row['task'],input_tokens=row['input_tokens'],
                output_tokens=len(result['output_ids']),token_events=events,engine_step_calls=calls,
                started=started,finished=time.monotonic(),TTFT_ms=1000*(first-started),
                cap_hit=result['finish_reason']=='length')
            return result
        longest=max(data,key=lambda r:r['input_tokens'])
        warm=generate('context_memory_warmup',longest,16,True)
        assert warm['output_tokens']==16
        write(folder/'context_memory_gate.json',dict(passed=True,input_tokens=longest['input_tokens'],
            diagnostic_output_tokens=warm['output_tokens'],model_context_limit=40960,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            scope='Standard-engine maximum-input memory control. Not a Native KV-transfer state gate. No Main performance claim.'))
        for index,row in enumerate(data):
            result=generate(f'reference_{index}',row,cfg['output_cap'],False)
            results.append(result)
            with (folder/'natural_outputs.jsonl').open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            print('NATURAL_REFERENCE',index,row['task'],row['input_tokens'],result['output_tokens'],result['finish_reason'],flush=True)
        assert len(results)==200
        write(folder/'natural_reference.json',dict(rows=results,dataset_SHA256=cfg['dataset_SHA256'],
            model_revision=MODEL_REVISION,output_cap=cfg['output_cap'],EOS_token_ids=eos,
            batch_invariant=cfg['batch_invariant'],evidence='STANDARD_VLLM_NATURAL_OUTPUT_REFERENCE',
            scope='Sequential concurrency1 standard vLLM. Natural EOS with one common cap. Output token IDs, including emitted stop tokens, define the output contract. This is not the open-loop Main QPS benchmark.'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if engine is not None:engine.engine_core.shutdown()


if __name__=='__main__':main()
