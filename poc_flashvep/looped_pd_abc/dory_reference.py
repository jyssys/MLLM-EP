"""Execute the official, unchanged Dory PR model using vLLM TP1."""
import argparse
import time
import os
from .common import ROOT,MODELS,MODEL_PINS,write,append

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',default='dory_official_smoke');p.add_argument('--requests',type=int,default=4)
 p.add_argument('--cap',type=int,default=16);p.add_argument('--batch-size',type=int,default=4)
 p.add_argument('--kv-mode',choices=('per_loop','last_loop'),default='per_loop');args=p.parse_args();dest=ROOT/'raw'/args.name
 write(dest/'status.json',dict(status='RUNNING',pid=os.getpid()))
 try:
  from .dory_plugin import install
  install()
  from vllm import LLM,SamplingParams
  llm=LLM(model=str(MODELS['dory']),dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
   max_model_len=4096,max_num_seqs=4,gpu_memory_utilization=.55,trust_remote_code=False,
   hf_overrides=dict(n_recurrent_loops=2,recurrent_kv_cache_mode=args.kv_mode),disable_log_stats=True)
  from .worker import prompts
  inputs=prompts('dory',args.requests,0)
  params=SamplingParams(temperature=0,max_tokens=args.cap,seed=20261029)
  llm.generate([dict(prompt_token_ids=inputs[0])],params,use_tqdm=False)
  start=time.monotonic();outputs=[]
  for batch_start in range(0,len(inputs),args.batch_size):
   outputs.extend(llm.generate([dict(prompt_token_ids=x) for x in inputs[batch_start:batch_start+args.batch_size]],params,use_tqdm=False))
  elapsed=time.monotonic()-start
  for i,o in enumerate(outputs):
   append(dest/'requests.jsonl',dict(sample_id=i,input_ids=inputs[i],output_ids=o.outputs[0].token_ids,
    text=o.outputs[0].text,finish_reason=o.outputs[0].finish_reason,model_revision=MODEL_PINS['dory'][1],
    depth=2,kv_layout=args.kv_mode,batch_size=args.batch_size,cohort_seconds=elapsed,evidence='MEASURED_OFFICIAL_MODEL_VLLM026_COMPATIBILITY_PLUGIN'))
  write(dest/'status.json',dict(status='COMPLETE',cohort_seconds=elapsed))
 except BaseException as e:
  write(dest/'status.json',dict(status='FAILED',error=repr(e)));raise

if __name__=='__main__':main()
