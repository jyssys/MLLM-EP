"""Audit BF16 q=1 output selection. All four physical recurrent passes still run."""
import argparse
import json
import os
import time
import torch
from .common import ROOT,MODELS,MODEL_PINS,UUIDS,append,write,digest

def q1_posthoc_selection(gates):
 remaining=torch.ones_like(gates[0].squeeze(-1));pdf=[]
 for index,gate in enumerate(gates):
  hazard=torch.sigmoid(gate.squeeze(-1))
  pdf.append(hazard*remaining if index<len(gates)-1 else remaining)
  if index<len(gates)-1:remaining=remaining*(1-hazard)
 cdf=torch.stack(pdf,dim=2).cumsum(dim=2);mask=cdf>=1.
 choice=torch.argmax(mask.float(),dim=2);choice[~mask.any(dim=2)]=len(gates)-1
 return choice,cdf

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--name',default='ouro_reference_gate_audit');args=parser.parse_args()
 folder=ROOT/'raw'/args.name;write(folder/'status.json',dict(status='RUNNING',pid=os.getpid()))
 write(folder/'configuration.json',dict(arguments=vars(args),model_pin=MODEL_PINS['ouro'],source_sha256=digest(__file__),
  sampling='Greedy, max_new_tokens32, EOS enabled',default_threshold=1.,explicit_last_step=3,
  scope='Reference diagnostic only. Not clean serving performance.'))
 try:
  torch.set_num_threads(1)
  physical=int(os.environ['CUDA_VISIBLE_DEVICES'])
  assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[physical].removeprefix('GPU-')
  if os.environ.get('ABC_DIAGNOSTIC_BURN_OTHERS')=='1':
   from . import resources
   resources.base.restore([g for g in range(8) if g!=physical])
  from transformers import AutoModelForCausalLM
  model=AutoModelForCausalLM.from_pretrained(MODELS['ouro'],torch_dtype=torch.bfloat16,
   trust_remote_code=True,local_files_only=True).to('cuda:0').eval()
  native=[json.loads(l) for l in (ROOT/'raw/ouro_parity_native_P0_C1_s0/requests.jsonl').read_text().splitlines()]
  inputs={r['sample_id']:r['input_ids'] for r in native if not r['warmup']}
  old={r['sample_id']:r for r in map(json.loads,(ROOT/'raw/ouro_HF_4576_natural100/requests.jsonl').read_text().splitlines())}
  divergent=json.loads((ROOT/'state/ouro_reference_divergences.json').read_text())
  selected=sorted({r['sample_id'] for r in divergent}|{0,1,2,3})
  traces=[]
  def hook(module,arguments,result):
   gates=result[2]
   if len(gates)!=4:raise RuntimeError('The official model did not return four recurrent gate values')
   choice,cdf=q1_posthoc_selection(gates)
   traces.append(dict(posthoc_q1_depth_on_current_path=int(choice[0,-1])+1,
    final_position_cdf=cdf[0,-1].tolist(),gate_dtype=str(gates[0].dtype),
    evaluated_loop_count=len(gates),meaning='Post-hoc output selection, not compute skipping'))
  handle=model.model.register_forward_hook(hook);forced={};diagnostics=[]
  def generate(index,last):
   traces.clear();seed=20261029+index;torch.manual_seed(seed);torch.cuda.manual_seed(seed)
   ids=inputs[index];tokens=torch.tensor([ids],device='cuda:0');kw=dict(max_new_tokens=32,do_sample=False,use_cache=True,
    use_weighted_exit=False,exit_threshold=1.0)
   if last:kw['exit_at_step']=3
   begin=time.monotonic()
   with torch.inference_mode():out=model.generate(tokens,attention_mask=torch.ones_like(tokens),**kw)
   return dict(sample_id=index,input_ids=ids,output_ids=out[0,len(ids):].tolist(),seed=seed,
    model_revision=MODEL_PINS['ouro'][1],depth=4,explicit_last_step=last,exit_at_step=3 if last else None,
    gate_trace=list(traces),e2e_ms=1000*(time.monotonic()-begin),
    evidence='MEASURED_OFFICIAL_GATE_DIAGNOSTIC_NOT_SERVING_PERFORMANCE',transformers_version=__import__('transformers').__version__)
  for index in selected:
   for last in (False,True):
    row=generate(index,last);row['matches_previous_default']=row['output_ids']==old[index]['output_ids']
    append(folder/'diagnostics.jsonl',row);diagnostics.append(row)
    if last:forced[index]=row
  early=sum(any(t['posthoc_q1_depth_on_current_path']<4 for t in r['gate_trace']) for r in diagnostics if not r['explicit_last_step'])
  full=ROOT/'raw/ouro_HF_4576_explicit_last_R4_100'
  if early:
   write(full/'status.json',dict(status='RUNNING',pid=os.getpid()))
   for index in range(100):
    row=forced.get(index) or generate(index,True)
    append(full/'requests.jsonl',row)
   write(full/'status.json',dict(status='COMPLETE',scope='Explicit zero-based exit_at_step=3. Original q=1 reference remains unchanged.'))
  handle.remove()
  write(folder/'status.json',dict(status='COMPLETE',diagnostic_prompts=len(selected),
   default_paths_with_posthoc_depth_below_four=early,
   full_explicit_R4_reference='COMPLETE' if early else 'NOT_RUN_NO_EARLY_SELECTION_OBSERVED_IN_DIAGNOSTIC',
   limitation='Diagnostic prompts include previous mismatches. Their rates are not representative accuracy or gate-frequency estimates.'))
 except BaseException as error:
  write(folder/'status.json',dict(status='FAILED',error=repr(error)));raise

if __name__=='__main__':main()
