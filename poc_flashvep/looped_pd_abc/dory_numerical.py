"""Isolate full-prompt versus split-coda BF16 geometry using one fixed backend."""
import json
import os
import torch
from .common import ROOT,write,append
from .state_probe import Native,generate
from .worker import prompts
from . import resources

def main():
 dest=ROOT/'raw/dory_coda_geometry';resources.base.stop([0]);write(dest/'status.json',dict(status='RUNNING',pid=os.getpid()))
 try:
  torch.set_num_threads(1);model=Native('dory','cuda:0')
  reference={r['sample_id']:r for r in map(json.loads,(ROOT/'raw/dory_official_B1_100/requests.jsonl').read_text().splitlines())}
  for i,ids in enumerate(prompts('dory',4,0)):
   a,x,ca=generate((model,model),ids,32,None,20261029+i,coda_mode='full')
   b,y,cb=generate((model,model),ids,32,None,20261029+i,coda_mode='split')
   p=x.softmax(-1);q=y.softmax(-1)
   append(dest/'samples.jsonl',dict(sample_id=i,full_output=a['output_ids'],split_output=b['output_ids'],
    original_output=reference[i]['output_ids'],full_original_exact=a['output_ids']==reference[i]['output_ids'],
    split_original_exact=b['output_ids']==reference[i]['output_ids'],full_split_exact=a['output_ids']==b['output_ids'],
    first_logit_relative_l2=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x)),
    first_logit_max_abs=float((x-y).abs().max()),first_logit_KL=float((p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum()),
    top1_full=int(x.argmax()),top1_split=int(y.argmax()),evidence='MEASURED_NUMERICAL_ABLATION_NOT_SERVING_SPEEDUP'))
  write(dest/'status.json',dict(status='COMPLETE'))
 except BaseException as e:write(dest/'status.json',dict(status='FAILED',error=repr(e)));raise
 finally:resources.base.restore([0])

if __name__=='__main__':main()
