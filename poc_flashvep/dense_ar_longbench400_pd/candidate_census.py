"""Census the official snapshot without changing any original prompt."""
import collections
import hashlib
import json
from pathlib import Path
import struct
import time
import zipfile
from .common import ROOT,REPO,MODEL,REVISION,initialize,write,digest

TEMPLATE_REVISION='2e00731f8d0bff23dc4325161044d0ed8af94c1e'
ZIP_SHA256='cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'


def token_hash(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()


def input_bucket(length):
    for index,(left,right) in enumerate(((3000,6000),(6000,10000),(10000,20000),(20000,36001))):
        if left<=length<right:return index
    return None


def main():
    initialize();target=ROOT/'data/candidate_pool_original.json'
    if target.exists():
        print('CANDIDATE_CENSUS_ALREADY_FROZEN',digest(target));return
    declaration=json.loads((ROOT/'configs/campaign_contract_v1.json').read_text())
    tasks=declaration['candidate_tasks_initial']+declaration['candidate_expansion_order']
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots')/REVISION/'data.zip'
    assert digest(archive)==ZIP_SHA256
    source=REPO/'poc_flashvep/reports/dense_ar_native_pd_real_workload_reproduction_20261119/references/longbench_v1_dataset2prompt.json'
    templates=json.loads(source.read_text());write(ROOT/'references/longbench_v1_dataset2prompt.json',templates)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    candidates=[];inventory=[];task_stats={};seen=set();dataset_counts={}
    with zipfile.ZipFile(archive) as bundle:
        members=[n for n in bundle.namelist() if n.endswith('.jsonl') and not Path(n).stem.endswith('_e')]
        for member in members:
            task=Path(member).stem
            raw=[json.loads(line) for line in bundle.read(member).decode().splitlines()]
            dataset_counts[task]=len(raw)
            if task not in templates:
                inventory.extend(dict(task=task,source_id=r.get('_id'),excluded='NO_OFFICIAL_TEMPLATE') for r in raw);continue
            counts=collections.Counter();lengths=[]
            for index,row in enumerate(raw):
                prompt=templates[task].format(**row)
                ids=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                length=len(ids);bucket=input_bucket(length);lengths.append(length)
                pid=f'longbench_v1_{task}_{row["_id"]}'
                original_sha=hashlib.sha256(prompt.encode()).hexdigest()
                context_sha=hashlib.sha256(row['context'].encode()).hexdigest()
                reason='OUTSIDE_DECLARED_TASKS' if task not in tasks else 'OUTSIDE_TARGET_INPUT_BUCKETS' if bucket is None else None
                if length+2048>40960 and task in tasks:reason='CONTEXT_INFEASIBLE_WITH_OUTPUT_RESERVE'
                duplicate=pid in seen;seen.add(pid)
                if duplicate:reason='DUPLICATE_REQUEST_ID'
                counts['total']+=1;counts[str(bucket) if bucket is not None else 'outside']+=1
                record=dict(prompt_id=pid,task=task,source_id=row['_id'],source_index=index,
                    source_member=member,input_tokens=length,input_bucket=bucket,
                    prompt_text_SHA256=original_sha,context_SHA256=context_sha,
                    excluded=reason,context_truncated=False)
                inventory.append(record)
                if reason is None:
                    candidates.append(dict(record,sample_id=len(candidates),source_dataset='zai-org/LongBench',source_revision=REVISION,
                        original_context=row['context'],original_input=row['input'],reference_answers=row['answers'],
                        rendered_prompt_full=prompt,input_tokens_original=length,input_ids=dict(qwen8=ids),
                        input_ids_SHA256=token_hash(ids),generated_output_tokens=None,output_cap=2048,ignore_eos=False,
                        thinking_disabled=True,category='SUMMARIZATION' if task in ('qmsum','samsum','multi_news','gov_report') else 'QA',
                        language='zh' if task in ('multifieldqa_zh','dureader') else 'en',
                        expansion_task=task not in declaration['candidate_tasks_initial']))
            task_stats[task]=dict(counts=counts,input_min=min(lengths),input_max=max(lengths))
            write(ROOT/'state/candidate_census_progress.json',dict(time=time.time(),completed_task=task,
                source_tasks_done=len(task_stats),feasible_candidates=len(candidates)))
            print('TASK_CENSUS',task,len(raw),len(candidates),flush=True)
    assert len(candidates)==len({r['prompt_id'] for r in candidates})
    write(target,candidates)
    contexts=collections.Counter(r['context_SHA256'] for r in candidates)
    write(ROOT/'data/candidate_input_inventory.json',dict(rows=inventory,source_revision=REVISION,
        source_zip_SHA256=ZIP_SHA256,template_revision=TEMPLATE_REVISION,template_SHA256=digest(source),
        candidate_tasks=tasks,official_task_counts=dataset_counts,task_input_statistics=task_stats,
        candidates=len(candidates),candidate_pool_SHA256=digest(target),
        unique_requests=len(candidates),unique_contexts=len(contexts),
        duplicate_context_requests=sum(n-1 for n in contexts.values()),
        actual_context_truncation_count=0,
        scope='All non-E LongBench tasks are tokenized and audited. Only the predeclared QA/summarization universe enters natural-reference generation. '
        'No new truncation,padding,request duplication or source text edits. Natural output availability and 4x4 balance remain NOT_MEASURED.'))
    for worker in range(8):
        rows=candidates[worker::8]
        write(ROOT/'data'/f'candidate_reference_shard{worker}.json',rows)
    write(ROOT/'state/candidate_input_census_complete.json',dict(time=time.time(),passed=True,candidates=len(candidates),SHA256=digest(target)))
    print('CANDIDATE_CENSUS_FROZEN',len(candidates),digest(target),flush=True)


if __name__=='__main__':main()
