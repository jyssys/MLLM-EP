"""Run unchanged checkpoint-owned HF code. Failures remain first-class evidence."""
import argparse
import os
import time
import torch
from .common import ROOT,MODELS,MODEL_PINS,write,append
from .worker import prompts

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=('ouro','huginn'),required=True)
 p.add_argument('--name',required=True);p.add_argument('--requests',type=int,default=4)
 p.add_argument('--cap',type=int,default=8);p.add_argument('--length',type=int,default=128);a=p.parse_args()
 # This reference reads the same frozen prompt IDs as the native pilot when available.
 dest=ROOT/'raw'/a.name;write(dest/'status.json',dict(status='RUNNING',pid=os.getpid()))
 try:
  from transformers import AutoModelForCausalLM
  torch.set_num_threads(1)
  model=AutoModelForCausalLM.from_pretrained(MODELS[a.model],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0').eval()
  source=ROOT/'raw'/f'{a.model}_parity2_native_P0_C1_s0'/'requests.jsonl'
  if not source.exists():source=ROOT/'raw'/f'{a.model}_parity_native_P0_C1_s0'/'requests.jsonl'
  if a.length==0 and source.exists():
   import json
   collected={r['sample_id']:r['input_ids'] for r in map(json.loads,source.read_text().splitlines()) if not r['warmup']}
   input_rows=[collected[i] for i in range(a.requests)]
  else:input_rows=prompts(a.model,a.requests,a.length)
  for i,ids in enumerate(input_rows):
   seed=20261029+i;torch.manual_seed(seed);torch.cuda.manual_seed(seed)
   kwargs=dict(max_new_tokens=a.cap,do_sample=False,use_cache=True)
   if a.model=='huginn':kwargs['num_steps']=32
   inputs=torch.tensor([ids],device='cuda:0')
   begin=time.monotonic()
   with torch.inference_mode():output=model.generate(inputs,attention_mask=torch.ones_like(inputs),**kwargs)
   torch.cuda.synchronize()
   generated=output[0,len(ids):].tolist()
   append(dest/'requests.jsonl',dict(sample_id=i,input_ids=ids,output_ids=generated,seed=seed,
    model_revision=MODEL_PINS[a.model][1],depth=MODEL_PINS[a.model][2],e2e_ms=1000*(time.monotonic()-begin),
    evidence='MEASURED_CHECKPOINT_OWNED_HF_REFERENCE',transformers_version=__import__('transformers').__version__,rng_scope='HF global generator. Huginn consumes its original discarded normal draw.'))
  write(dest/'status.json',dict(status='COMPLETE'))
 except BaseException as e:
  write(dest/'status.json',dict(status='FAILED',error=repr(e)));raise

if __name__=='__main__':main()
