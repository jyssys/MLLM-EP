"""Prepare immutable Native qualification cells before any migration."""
import json,random
from .common import *


def save(c):
    initialize();p=ROOT/'configs'/(c['name']+'.json')
    if p.exists():assert json.loads(p.read_text())==c
    else:write(p,c)
    return c['name']


def base(name,count=8,length=8000,cap=32):
    return dict(name=name,model='qwen8',model_revision=MODEL_REVISION,arm='PD_G1',length=length,cap=cap,
        depth=1,cut=1,context_limit=40960,P_slots=4,D_slots=16,chunk=length,token_budget=32000,
        ar_batch_invariant=True,ar_enforce_eager=False,ar_cudagraph_mode='NONE',ar_async_scheduling=False,
        ar_execution_family='BI_COMPILED_NO_GRAPH',gpu_memory_utilization=.80,seed=2026110507,
        arrival_seed=2026111201,gpus=[0,1],count=count,dataset='qualification_qwen8_L8000.json',
        request_caps=[cap]*count,arrival_offsets=[0.]*count,warmup_count=2,
        atomic_prompt=True,timeout=300,startup_timeout=360,phase='NATIVE_CORRECTNESS',
        prefix_caching=False,ignore_eos=True,clean_timing=False,
        qualification_scope='Exact generated token sequence. Full valid KV and Hidden require separate State capture before Layer continuation claims.')


def main():
    initialize();source=ATOMIC/'data/extension_qualification32_qwen8_L8000.json'
    values=json.loads(source.read_text());assert len(values)==32 and len({r['prompt_id'] for r in values})==32
    target=ROOT/'data/qualification_qwen8_L8000.json'
    if not target.exists():write(target,values)
    else:assert json.loads(target.read_text())==values
    write(ROOT/'data/qualification_manifest.json',dict(source=str(source),source_SHA256=digest(source),
        target_SHA256=digest(target),count=32,distinct_prompt_ids=32,length=8000,
        tokenizer_revision=MODEL_REVISION,template='Preserved controlled tokenized text prefixes. No new chat template or filler tokens.',prefix_caching=False))
    slo=json.loads((ATOMIC/'configs/frozen_SLO_qwen8_bi_compiled_none_extension.json').read_text())
    write(ROOT/'configs/frozen_SLO_anchor.json',dict(slo,scope='Preserved qualified BI Native 8000-token execution-family thresholds. New Output16/32 still uses the same limits. No candidate result changes them.'))
    jobs=[]
    for suffix in ('reference','repeat'):
        c=base('native_qwen8_L8000_O32_'+suffix+'8_s0');jobs.append(save(c))
    c=base('native_qwen8_L8000_O32_serial8_s0');c.update(calibration_serial=True,phase='NATIVE_SERIAL_CALIBRATION',clean_timing=True)
    jobs.append(save(c))
    write(ROOT/'state/native_initial_plan.json',dict(jobs=jobs,
        scope='Fresh Native N8 starts first. A pass does not qualify Layer migration, original HF fidelity, or sustainable capacity.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()
