"""Record source, checkpoint, hardware, and dependency provenance without secrets."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RLT,REPO,MODEL_PINS,MODELS,write,digest,setup

def command(args,cwd=None):
 p=subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 return dict(command=args,returncode=p.returncode,output=p.stdout)

def main():
 setup()
 hardware={name:command(args) for name,args in {
  'gpu_ids':['nvidia-smi','-L'],'topology':['nvidia-smi','topo','-m'],
  'gpu_state':['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version,clocks.sm,power.limit,temperature.gpu','--format=csv'],
  'cpu':['lscpu'],'memory':['free','-h']}.items()}
 write(ROOT/'configs/hardware.json',hardware)
 pins={}
 for name,(repo,revision,R) in MODEL_PINS.items():
  path=MODELS[name];config=json.loads((path/'config.json').read_text())
  from safetensors import safe_open
  files=sorted(path.glob('*.safetensors'));tensors=0;parameters=0;dtypes={}
  for file in files:
   with safe_open(file,framework='pt',device='cpu') as f:
    for key in f.keys():
     s=f.get_slice(key);count=1
     for n in s.get_shape():count*=n
     tensors+=1;parameters+=count;dtypes[s.get_dtype()]=dtypes.get(s.get_dtype(),0)+count
  pins[name]=dict(repo=repo,revision=revision,loops=R,config=config,config_sha256=digest(path/'config.json'),
   checkpoint_tensor_count=tensors,checkpoint_element_count=parameters,dtypes=dtypes,
   tokenizer_files={p.name:digest(p) for p in path.glob('*') if p.name.startswith(('tokenizer','special_tokens','chat_template'))})
 write(ROOT/'configs/models.json',pins)
 packages={}
 for package in ('torch','transformers','triton','flash-attn-4','nixl','nixl-cu12','vllm','numpy','ninja'):
  try:packages[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:packages[package]='NOT_INSTALLED'
 write(ROOT/'configs/environment.json',dict(packages=packages,time=time.time(),python=__import__('sys').version,
  runtime_revision=command(['git','rev-parse','HEAD'],RLT),branch=command(['git','branch','--show-current'],REPO)))
 patch=subprocess.check_output(['git','diff','--binary'],cwd=RLT)
 (ROOT/'sources/rlt_changes.patch').write_bytes(patch)
 write(ROOT/'configs/patch_manifest.json',dict(patch_sha256=digest(ROOT/'sources/rlt_changes.patch'),
  source_files={str(p.relative_to(REPO)):digest(p) for p in (REPO/'poc_flashvep/looped_pd_abc').glob('*.py')}))
 print('AUDIT_SAVED',len(pins),packages)

if __name__=='__main__':main()
