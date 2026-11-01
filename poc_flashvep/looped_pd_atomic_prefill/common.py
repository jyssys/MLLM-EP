"""Pins, isolated evidence paths, and durable experiment records."""
import json
import os
import time
from pathlib import Path
from ..looped_pd_abc.common import REPO,RLT,PYTHON,UUIDS,write,append,digest,environment as inherited_environment

ROOT=REPO/'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'
SOURCE=Path(__file__).parent
PRIOR=REPO/'poc_flashvep/reports/looped_pd_bottleneck_crossover'
SPEC=REPO/'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover_poc_spec.md'
BRANCH='flashvep/looped-pd-atomic-prefill-qps-crossover'
BASE='f9bdad32855e4af620cace1111b366c366039c40'
START_EPOCH=1793505788.
MODEL_PINS={
    'ouro':('ByteDance/Ouro-1.4B','574fa66cb8bf5abdc979642d01cf2b79b16bfab1',4),
    'huginn':('tomg-group-umd/huginn-0125','bb6621b65e90b6a4b9b29ef88dc83866d450470c',32),
    'qwen8':('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218',1)}
MODEL_PATHS={m:Path('/home/esjung/.cache/huggingface/hub')/('models--'+p[0].replace('/','--'))/'snapshots'/p[1] for m,p in MODEL_PINS.items()}
CONTEXT_LIMITS={'ouro':65536,'huginn':4096,'qwen8':40960}
GRID={'qwen8':[(8000,200),(8000,100),(3900,100),(3900,64),(3900,32)],
    'ouro':[(4096,100),(4096,64),(8192,200),(8192,100),(8192,64),(8192,32),(16384,100)],
    'huginn':[(3900,100),(3900,64),(3900,32),(3500,100),(3500,64),(3500,32)]}
REPORTS=['looped_pd_atomic_prefill_runtime_audit.md','looped_pd_atomic_prefill_correctness.md',
    'looped_pd_atomic_prefill_capacity.md','looped_pd_atomic_prefill_qps_search.md',
    'looped_pd_atomic_prefill_qps_crossover_final_summary.md','looped_pd_atomic_prefill_reproduce.md']
OWNED=['poc_flashvep/looped_pd_atomic_prefill/','poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover/',str(SPEC.relative_to(REPO))]
OWNED+=['poc_flashvep/reports/'+n for n in REPORTS]


def setup():
    for n in ('raw','data','configs','logs','state','figures'):(ROOT/n).mkdir(parents=True,exist_ok=True)


def config():return json.loads(Path(os.environ['ARC_CONFIG']).read_text())


def environment(gpus):
    env=inherited_environment(gpus)
    env['PYTHONPATH']=str(SOURCE/'bootstrap')+os.pathsep+env['PYTHONPATH']
    env['ATOMIC_PD_WORKER']='1'
    env.pop('BC_WORKER',None)
    return env


def redirect():
    from ..looped_pd_bottleneck_crossover import common as previous
    previous.ROOT=ROOT;previous.setup=setup;previous.config=config
    previous.redirect()


def save_config(value):
    setup();path=ROOT/'configs'/(value['name']+'.json')
    if path.exists():
        assert json.loads(path.read_text())==value,'Do not overwrite an existing configuration'
    else:write(path,value)
    return path.name


def base(model,name,length,cap,**extra):
    c=dict(name=name,model=model,model_revision=MODEL_PINS[model][1],arm='PD_G1',cut=MODEL_PINS[model][2],
        depth=MODEL_PINS[model][2],length=length,cap=cap,context_limit=CONTEXT_LIMITS[model],
        state_capacity_tokens=length,capacity=8,slots=8,P_slots=4,D_slots=8,
        chunk=length,chunk_label='full',token_budget=max(length,(32768//length)*length),graphs=True,wavefront=False,
        asynchronous=model=='ouro',kv_blocks=22528 if model=='ouro' else 26624,
        seed=2026110107,arrival_seed=20261101,gpus=[0,1] if model=='qwen8' else [2,3] if model=='ouro' else [4,5],
        handoff_mode='native_G1',worker='rlt_worker' if model!='qwen8' else 'ar_worker',
        phase='correctness',axis='loop',dynamic=False,workload=f'L{length}_O{cap}',count=8,timeout=3600,estimated_seconds=300,
        dataset=f'test_{model}_L{length}.json',request_caps=[cap]*8,arrival_offsets=[0.]*8,
        warmup_count=2,warmup_cap=2,atomic_prompt=True,verify_transfer=False,state_probe=False)
    c.update(extra)
    assert length+cap<=c['context_limit']
    return c
