"""Pins and scopes for the new campaign. No predecessor constant is modified."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
ASSETS=Path('/home/esjung/external/ouro-deep-discovery-assets')
SHARED_DEPS=Path('/home/esjung/external/decode-first-depth-dp-assets')
REFERENCE_PYTHON='/home/esjung/.venvs/llada2-mini-reference/bin/python'
NATIVE_PYTHON='/home/esjung/.venvs/cp-composition-vllm026/bin/python'
PYTHON='/home/esjung/.venvs/ouro-deep-discovery/bin/python'
CDB=Path('/home/esjung/external/decode-first-cdb-upstream')
CDB_REVISION='abdfae0adbdd12be8c6123ccb34eee98a38afb39'
GPUS=[4,5,6,7]
UUIDS=['6076e2f2-5b63-3761-5586-56ceb7df8139','a1a1cfcf-93a1-3544-9a5e-e58144b68730',
       'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','4cc26b88-19fc-1988-f9e0-17858aa7a99b']
MODELS={
 '1.4B':dict(id='ByteDance/Ouro-1.4B',revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1',layers=24),
 '2.6B':dict(id='ByteDance/Ouro-2.6B',revision='1ed04250da1a9936042725d302e81c8fa2ab5abd',layers=48),
}
for value in MODELS.values():
    value['path']=Path('/home/esjung/.cache/huggingface/hub')/('models--'+value['id'].replace('/','--'))/'snapshots'/value['revision']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def read(path):return json.loads(Path(path).read_text())

def validate_mask():
    mask=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    assert mask and set(mask)<=set(GPUS) and len(mask)==len(set(mask))
    return mask
