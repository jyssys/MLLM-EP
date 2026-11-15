"""Freeze unseen public prompts without reading candidate performance."""
import random,zipfile
from .common import *


def main():
    initialize();manifest=ROOT/'data/unseen_manifest.json'
    if manifest.exists():print('UNSEEN_DATA_ALREADY_FROZEN');return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    seen=set();prefixes=set();prior=[]
    for directory in (PREVIOUS/'data',LEGACY/'data',STATIC/'data',ATOMIC/'data'):
        for path in directory.glob('*.json'):
            value=json.loads(path.read_text())
            if not isinstance(value,list):continue
            for row in value:
                if not isinstance(row,dict) or 'prompt_id' not in row:continue
                seen.add(row['prompt_id'])
                if 'qwen8' in row.get('input_ids',{}):prefixes.add(tuple(row['input_ids']['qwen8'][:7900]))
            prior.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip')
    candidates=[]
    with zipfile.ZipFile(archive) as stream:
        for task in ('gov_report','qmsum','narrativeqa','qasper','multifieldqa_en','hotpotqa','2wikimqa','musique'):
            paths=[n for n in stream.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl']
            if not paths:continue
            for index,line in enumerate(stream.read(paths[0]).decode().splitlines()):
                row=json.loads(line);identifier=f'longbench_{task}_{row.get("_id",index)}'
                if identifier in seen:continue
                messages=[dict(role='user',content='Document:\n'+row['context']+'\nQuestion:\n'+row.get('input',''))]
                candidates.append((identifier,task,messages))
    random.Random(2026111407).shuffle(candidates)
    tail=tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    check=tokenizer.apply_chat_template([dict(role='user',content='test')],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False)
    assert check[-len(tail):]==tail
    rows=[]
    for identifier,task,messages in candidates:
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=False)
        if len(ids)<8000:continue
        clipped=ids[:8000-len(tail)]+tail
        if tuple(clipped[:7900]) in prefixes:continue
        prefixes.add(tuple(clipped[:7900]));rows.append(dict(prompt_id=identifier,source_dataset='LongBench/'+task,
            source_SHA256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest(),
            input_ids={'qwen8':clipped},length=8000,original_tokenized_length=len(ids),
            source_type='CONTROLLED_PUBLIC_DOCUMENT_PREFIX_WITH_COMPLETE_GENERATION_SUFFIX',
            truncation_explicit=True,thinking_disabled_suffix_verified=True))
        if len(rows)==160:break
    assert len(rows)==160,'Insufficient unused real long documents'
    for name,group in (('unseen_calibration',rows[:32]),('unseen_confirmation',rows[32:])):
        write(ROOT/'data'/(name+'.json'),group)
    write(manifest,dict(time=time.time(),model_revision=MODEL_REVISION,selection_seed=2026111407,
        dataset_revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672',archive_SHA256=digest(archive),
        prior_files=prior,prior_prompt_ID_overlap=0,prior_7900_token_prefix_overlap=0,
        generation_suffix_ids=tail,groups={name:dict(count=len(group),
            SHA256=digest(ROOT/'data'/(name+'.json')),prompt_ids=[r['prompt_id'] for r in group])
            for name,group in (('unseen_calibration',rows[:32]),('unseen_confirmation',rows[32:]))},
        scope='Unseen prompt IDs remain separate from discovery. Cycled duration traces are not independent prompt samples.'))
    print('UNSEEN_DATA_FROZEN',len(rows))


if __name__=='__main__':main()
