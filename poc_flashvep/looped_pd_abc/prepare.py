"""Download only the pinned official Dory BF16 checkpoint. Never use a proxy."""
import json
import time
from .common import ROOT,MODEL_PINS,setup,write

def main():
 setup()
 name,revision,depth=MODEL_PINS['dory']
 write(ROOT/'state/dory_download.json',dict(status='RUNNING',model=name,revision=revision,time=time.time()))
 try:
  from huggingface_hub import snapshot_download
  path=snapshot_download(name,revision=revision,allow_patterns=['*.json','*.safetensors','*.py','*.jinja','README.md'],max_workers=4)
 except BaseException as error:
  write(ROOT/'state/dory_download.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
 write(ROOT/'state/dory_download.json',dict(status='COMPLETE',path=path,revision=revision,time=time.time()))
 print('PINNED_DORY_READY',path,flush=True)

if __name__=='__main__':main()
