"""Measure natural output lengths with the original single-GPU vLLM model."""
import argparse
import importlib.metadata
import json
import os
import time
from .common import ROOT,MODEL,MODEL_REVISION,write,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);parser.add_argument('--version',required=True);args=parser.parse_args()
    assert args.version.isalnum(),args.version
    shard=args.shard;run_folder=ROOT/'raw'/f'CandidateNaturalReference_{args.version}';folder=run_folder/f'GPU{shard}'
    folder.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data'/f'candidate_reference_shard{shard}.json';data=json.loads(path.read_text())
    assert os.environ['VLLM_BATCH_INVARIANT']=='0'
    import torch
    torch.set_num_threads(1)
    from vllm import LLM,SamplingParams
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    assert Qwen3ForCausalLM.forward.__module__=='vllm.model_executor.models.qwen3'
    options=dict(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=40960,max_num_batched_tokens=8192,max_num_seqs=1024,
        kv_cache_memory_bytes=42*1024**3,enable_prefix_caching=False,enable_chunked_prefill=True,
        async_scheduling=True,seed=2026112101,disable_log_stats=True,
        attention_config=dict(backend='FLASH_ATTN'),block_size=128,
        compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[1,2,4,8,16,32]))
    engine=None;results=folder/'natural_reference.jsonl'
    try:
        write(folder/'status.json',dict(status='STARTING',time=time.time(),pid=os.getpid(),shard=shard))
        llm=LLM(**options);engine=llm.llm_engine
        write(folder/'actual_runtime.json',dict(time=time.time(),options=options,model_revision=MODEL_REVISION,
            source_dataset_SHA256=digest(path),physical_GPU=shard,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
            device_name=torch.cuda.get_device_name(0),scheduler_B=engine.vllm_config.scheduler_config.max_num_batched_tokens,
            scheduler_S=engine.vllm_config.scheduler_config.max_num_seqs,
            model_forward_module=Qwen3ForCausalLM.forward.__module__,
            versions={name:importlib.metadata.version(name) for name in ('vllm','torch','transformers','nixl')},
            scope='Original installed single-GPU vLLM model. Sequential concurrency1. No model-math,NIXL or partial-forward adapter. Timings are diagnostic,not serving capacity.'))
        completed={json.loads(line)['prompt_id'] for line in results.read_text().splitlines()} if results.exists() else set()
        params=SamplingParams(max_tokens=2048,temperature=0,ignore_eos=False,seed=2026112101)
        for row in data:
            if row['prompt_id'] in completed:continue
            if (run_folder/'STOP_AFTER_REQUEST.json').exists():break
            began=time.monotonic()
            output=llm.generate([dict(prompt_token_ids=row['input_ids']['qwen8'])],params,use_tqdm=False)[0]
            value=output.outputs[0];ids=list(value.token_ids)
            record=dict(prompt_id=row['prompt_id'],sample_id=row['sample_id'],task=row['task'],
                source_id=row['source_id'],input_tokens=row['input_tokens'],input_ids_SHA256=row['input_ids_SHA256'],
                context_SHA256=row['context_SHA256'],input_bucket=row['input_bucket'],output_ids=ids,
                output_tokens=len(ids),text=value.text,finish_reason=value.finish_reason,stop_reason=value.stop_reason,
                cap_hit=value.finish_reason=='length',natural_EOS=value.finish_reason=='stop',
                wall_seconds=time.monotonic()-began,physical_GPU=shard,BI=0,
                evidence='NATURAL_OUTPUT_REFERENCE',scope='Natural EOS,commoncap2048. Output length is measured,not a reference-answer token count. Timing is diagnostic only.')
            assert ids and len(ids)<=2048
            with results.open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            completed.add(row['prompt_id'])
            write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid(),completed=len(completed),expected=len(data)))
            print('NATURAL_REFERENCE',shard,len(completed),len(data),row['task'],len(ids),flush=True)
        assert len(completed)==len(data),'Reference stopped at a safe per-request checkpoint'
        write(folder/'status.json',dict(status='COMPLETE',time=time.time(),completed=len(completed),expected=len(data)))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error)));raise
    finally:
        if engine is not None:engine.engine_core.shutdown()


if __name__=='__main__':main()
