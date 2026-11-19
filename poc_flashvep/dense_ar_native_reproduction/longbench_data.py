"""Freeze raw LongBench selection before context handling or serving results."""
import collections
import hashlib
import json
import random
import statistics
import struct
import time
import urllib.request
import zipfile
from pathlib import Path
from .common import ROOT,MODEL,MODEL_REVISION,write,digest,initialize,PREVIOUS

DATA_REVISION='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
ZIP_SHA256='cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'
TEMPLATE_REVISION='2e00731f8d0bff23dc4325161044d0ed8af94c1e'
TASKS=('hotpotqa','musique','narrativeqa','gov_report','qmsum')
SAMPLING_SEED=2026111999
ORDER_SEED=2026112000
OUTPUT_CAP=2048
CONTEXT_LIMIT=40960
INPUT_LIMIT=CONTEXT_LIMIT-OUTPUT_CAP


def quantile(values,p):
    v=sorted(values);x=(len(v)-1)*p;i=int(x);j=min(i+1,len(v)-1)
    return v[i]+(v[j]-v[i])*(x-i)


def stats(values):
    return dict(count=len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values),
        **{'p'+str(p):quantile(values,p/100) for p in (10,25,50,75,90,95,99)})


def token_hash(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()


def main():
    initialize();selection_path=ROOT/'data/longbench200_sampling_frozen.json'
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots')/DATA_REVISION/'data.zip'
    assert digest(archive)==ZIP_SHA256
    template_path=ROOT/'references/longbench_v1_dataset2prompt.json'
    if not template_path.exists():
        url=f'https://raw.githubusercontent.com/THUDM/LongBench/{TEMPLATE_REVISION}/LongBench/config/dataset2prompt.json'
        with urllib.request.urlopen(url,timeout=60) as response:templates=json.loads(response.read())
        write(template_path,templates)
    templates=json.loads(template_path.read_text())
    selected=[];pools={}
    with zipfile.ZipFile(archive) as bundle:
        for number,task in enumerate(TASKS):
            members=[n for n in bundle.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl']
            assert len(members)==1,(task,members)
            raw=[json.loads(line) for line in bundle.read(members[0]).decode().splitlines()]
            indices=random.Random(SAMPLING_SEED+number).sample(range(len(raw)),40)
            pools[task]=dict(source_rows=len(raw),member=members[0],indices=indices)
            for index in indices:
                row=raw[index];assert row['dataset']==task
                selected.append(dict(task=task,source_index=index,source_id=row['_id'],source_record=row))
    random.Random(ORDER_SEED).shuffle(selected)
    frozen=dict(dataset='zai-org/LongBench',revision=DATA_REVISION,archive_SHA256=ZIP_SHA256,
        sampling_seed=SAMPLING_SEED,order_seed=ORDER_SEED,task_pools=pools,
        requests=[dict(task=r['task'],source_index=r['source_index'],source_id=r['source_id']) for r in selected],
        scope='Uniform40/task sampled before token length or model output is inspected. Mixed order is independently fixed. No performance-based replacement.')
    if selection_path.exists():assert json.loads(selection_path.read_text())==frozen
    else:write(selection_path,frozen)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    rows=[];truncation=[];full_lengths=[]
    for index,item in enumerate(selected):
        task=item['task'];raw=item['source_record']
        prompt=templates[task].format(**raw)
        tokens=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert isinstance(tokens,list) and all(isinstance(t,int) for t in tokens)
        full_lengths.append(len(tokens));original_hash=token_hash(tokens)
        if len(tokens)>INPUT_LIMIT:
            before=len(tokens);left=INPUT_LIMIT//2;right=INPUT_LIMIT-left
            tokens=tokens[:left]+tokens[-right:]
            truncation.append(dict(index=index,task=task,source_id=item['source_id'],original_tokens=before,
                retained_tokens=len(tokens),head_tokens=left,tail_tokens=right,
                removed_middle_interval=[left,before-right],original_tokens_SHA256=original_hash,
                retained_tokens_SHA256=token_hash(tokens)))
        row=dict(sample_id=index,prompt_id=f'longbench_v1_{task}_{item["source_id"]}',
            task=task,category='QA' if task in TASKS[:3] else 'SUMMARIZATION',
            source_dataset='zai-org/LongBench',source_revision=DATA_REVISION,
            source_id=item['source_id'],source_index=item['source_index'],
            original_context=raw['context'],original_input=raw['input'],reference_answers=raw['answers'],
            rendered_prompt_full=prompt,prompt_text_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),
            context_SHA256=hashlib.sha256(raw['context'].encode()).hexdigest(),
            input_tokens_original=full_lengths[-1],
            input_tokens=len(tokens),input_ids=dict(qwen8=tokens),input_ids_SHA256=token_hash(tokens),
            generated_output_tokens=None,output_cap=OUTPUT_CAP,ignore_eos=False,thinking_disabled=True)
        rows.append(row)
    target=ROOT/'data/longbench_v1_mixed200.json'
    if target.exists():assert json.loads(target.read_text())==rows
    else:write(target,rows)
    old_manifest=json.loads((PREVIOUS/'data/length_control_manifest.json').read_text())
    previous_ids=set()
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('prompt_id','source_id','_id') and isinstance(v,str):previous_ids.add(v)
                collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
        elif isinstance(value,str) and value.startswith('longbench_'):
            previous_ids.add(value)
    collect(old_manifest)
    overlap=[r['prompt_id'] for r in rows if any(r['source_id'] in prior for prior in previous_ids)]
    lengths=[r['input_tokens'] for r in rows]
    buckets=collections.Counter('<=4096' if n<=4096 else '4097-8192' if n<=8192 else '8193-16384' if n<=16384 else '16385-32768' if n<=32768 else '>32768' for n in lengths)
    write(ROOT/'data/longbench200_truncation_manifest.json',dict(policy='Head-tail token truncation after the Qwen chat template, following the official LongBench head-tail principle.',
        context_limit=CONTEXT_LIMIT,output_cap=OUTPUT_CAP,input_limit=INPUT_LIMIT,
        no_context_based_exclusion=True,original_200_kept=True,requests=truncation,
        scope='Original full context and prompt are preserved. Token positions restart contiguously on the retained input. No filler. No performance-based prompt replacement.'))
    value=dict(time=time.time(),dataset='zai-org/LongBench',revision=DATA_REVISION,
        model_revision=MODEL_REVISION,task_counts=dict(collections.Counter(r['task'] for r in rows)),
        sampling_seed=SAMPLING_SEED,order_seed=ORDER_SEED,source_zip_SHA256=ZIP_SHA256,
        template_revision=TEMPLATE_REVISION,template_SHA256=digest(template_path),
        selection_SHA256=digest(selection_path),dataset_SHA256=digest(target),
        original_input=stats(full_lengths),actual_input=stats(lengths),
        original_over_model_context=sum(n>CONTEXT_LIMIT for n in full_lengths),
        truncated_requests=len(truncation),excluded_requests=0,input_buckets=dict(buckets),
        tasks={task:dict(original_input=stats([r['input_tokens_original'] for r in rows if r['task']==task]),
            actual_input=stats([r['input_tokens'] for r in rows if r['task']==task]),
            truncated_requests=sum(r['task']==task for r in truncation)) for task in TASKS},
        distinct_requests=len({r['prompt_id'] for r in rows}),distinct_contexts=len({r['context_SHA256'] for r in rows}),
        overlap_with_previous_144_source_documents=overlap,
        generated_output_distribution='NOT_GENERATED. Reference answer lengths are not model output lengths.',
        generation_contract=dict(greedy=True,thinking=False,ignore_eos=False,max_output_tokens=OUTPUT_CAP),
        context_contract=dict(model_limit=CONTEXT_LIMIT,input_limit=INPUT_LIMIT,KV_GiB_per_engine=42,
            single_request_feasibility='Fits model context and the configured total KV token capacity. Full memory/state GPU qualification is still required before Main Serving.'),
        evidence='PRE_SERVING_DATASET_ANALYSIS',
        scope='Five equal task strata, not a production-representative distribution. Selection is frozen before context analysis. Outputs and input/output joint distribution remain unmeasured until Natural Generation.')
    write(ROOT/'analysis/longbench200_pre_serving_contract.json',value)
    print(json.dumps({k:value[k] for k in ('task_counts','original_input','actual_input','truncated_requests','excluded_requests','input_buckets','dataset_SHA256')},indent=2))


if __name__=='__main__':main()
