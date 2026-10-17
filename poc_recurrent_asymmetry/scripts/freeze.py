"""Freeze contract and hardware-shaped prompts before profiling."""
import subprocess
import hashlib
import datetime
from poc_recurrent_asymmetry.contract import *


def main():
    target=ROOT/'reports/spec.md'
    assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(SPEC.read_bytes())
    old=read(PROJECT/'poc_async_recurrent_pp/outputs/manifests/data.json')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    rows=[]
    for row in old['calibration']:
        question=tokenizer.apply_chat_template([dict(role='user',content=row['question']+old['prompt_suffix'])],
            tokenize=True,add_generation_prompt=True)
        base=tokenizer.encode('Read the following reference context. '+row['question']+'\n',add_special_tokens=False)
        prompts={}
        for length in [1024,4096,8192,16384]:
            needed=length-len(question)
            assert needed>0
            prompts[str(length)]=(base*((needed+len(base)-1)//len(base)))[:needed]+question
            assert len(prompts[str(length)])==length
        rows.append(dict(id=row['id'],prompts=prompts))
    write(ROOT/'metrics/frozen_workload.json',dict(requests=rows,source='Eight prior disjoint GSM8K train calibration questions.',
        scope='Shape-controlled repeated context plus intact chat question. Not natural long-context benchmark quality.',
        contexts=[1024,4096,8192,16384],batches=[1,2,4,8],chunks=[128,256,512,1024],
        repeats=5,decode_steps=32,R=4,precision='BF16',counterfactuals='No prediction or stale state. Exact depth KV.',
        quality_ids=[r['id'] for dataset in ['gsm8k','math500'] for r in [x for x in old['pilot'] if x['dataset']==dataset][:2]]))
    original=Path('/home/esjung/MLLM-EP-github')
    dirty={name:hashlib.sha256(subprocess.check_output(args,cwd=original)).hexdigest()
           for name,args in [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]}
    write(ROOT/'metrics/frozen_contract.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spec_source=str(SPEC),spec_sha256=sha(SPEC),model_revision=REVISION,
        model_source_sha256=sha(MODEL/'modeling_ouro.py'),weight_sha256=sha(MODEL/'model.safetensors'),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip(),
        parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        original_tracked_diff_sha256=dirty,allowed_physical_GPUs=GPUS,
        workflow='Exact baseline, broad profiling, legal DAG, physical communication, separately labeled oracles.',
        primary_lane='Unmodified pinned HF exact reference. Native optimized vLLM is separately validated before any comparison.',
        traffic_rule='Resident capacity and structural accesses are not actual HBM traffic.',
        prototype='OUT_OF_SCOPE',approximate_predictors='CLOSED. No execution in this PoC.'))
    print('FROZEN',len(rows),'shape-control requests',sha(SPEC),flush=True)


if __name__=='__main__':
    main()
