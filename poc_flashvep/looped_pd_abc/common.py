"""Pinned inputs, append-only evidence, and physical GPU identities."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'poc_flashvep/reports/looped_pd_abc'
RLT=Path('/home/esjung/external/looped-pd-abc-rlt')
PYTHON='/home/esjung/.venvs/looped-pd-abc/bin/python'
RLT_REVISION='6b164d3844d35ab8e753df1864f0c87dc180401a'
MODEL_PINS={
 'ouro':('ByteDance/Ouro-1.4B','574fa66cb8bf5abdc979642d01cf2b79b16bfab1',4),
 'huginn':('tomg-group-umd/huginn-0125','bb6621b65e90b6a4b9b29ef88dc83866d450470c',32),
 'dory':('nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16','b5437f97df4b998a505f4cac5b23791cd3e9df94',2)}
_APPEND_COST={}
MODELS={key:Path('/home/esjung/.cache/huggingface/hub')/('models--'+name.replace('/','--'))/'snapshots'/revision
 for key,(name,revision,depth) in MODEL_PINS.items()}
UUIDS=['GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d','GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb',
 'GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e','GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8',
 'GPU-6076e2f2-5b63-3761-5586-56ceb7df8139','GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730',
 'GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b']

def setup():
 for sub in ('raw','logs','configs','sources','figures','state','archives'):
  (ROOT/sub).mkdir(parents=True,exist_ok=True)

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
  json.dump(value,f,indent=2,default=str);f.write('\n');temporary=f.name
 os.replace(temporary,path)

def append(path,value):
 begin=time.perf_counter()
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:
  f.write(json.dumps(value,default=str)+'\n');f.flush();os.fsync(f.fileno())
 elapsed=time.perf_counter()-begin
 stats=_APPEND_COST.setdefault(str(path),dict(count=0,total_seconds=0.,max_seconds=0.))
 stats['count']+=1;stats['total_seconds']+=elapsed;stats['max_seconds']=max(stats['max_seconds'],elapsed)

def append_cost():return {path:dict(values) for path,values in _APPEND_COST.items()}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def environment(gpus):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
  PYTHONPATH=f'{RLT}:{REPO}',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env.get('PATH','')
 env['ABC_HUGINN_REFERENCE']='1'
 return env

def record_command(name,command,gpus,pid):
 append(ROOT/'commands.jsonl',dict(name=name,command=command,gpus=gpus,pid=pid,time=time.time()))
