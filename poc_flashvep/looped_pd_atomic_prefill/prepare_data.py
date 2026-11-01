"""Freeze real long ShareGPT prefixes. Never synthesize undisclosed filler."""
import hashlib
import random
import zipfile
from .common import *


def main():
    from transformers import AutoTokenizer
    setup();frozen=ROOT/'data/long_source_pool.json'
    if frozen.exists():raise FileExistsError('Preserve the existing frozen source pool')
    provenance=json.loads((PRIOR/'configs/dataset_provenance.json').read_text())['original']
    source=Path(provenance['cache_path']);assert digest(source)==provenance['source_SHA256']
    corpus=json.loads(source.read_text());random.Random(2026110101).shuffle(corpus)
    tokenizers={m:AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,local_files_only=True) for m,path in MODEL_PATHS.items()}
    rows=[];seen=set();prefixes={m:set() for m in tokenizers};scanned=0
    def accept(messages,identifier,dataset):
        nonlocal scanned
        encoded=json.dumps(messages,sort_keys=True);sha=hashlib.sha256(encoded.encode()).hexdigest()
        if sha in seen or len(encoded)<21000:return False
        scanned+=1;ids={}
        try:
            for m in ('qwen8','ouro','huginn'):
                v=tokenizers[m].apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False,
                    **({'enable_thinking':False} if m=='qwen8' else {}))
                ids[m]=list(v['input_ids'] if isinstance(v,dict) else v)
                if m=='qwen8' and len(ids[m])<8000:return False
                if m=='ouro' and len(ids[m])<8192:return False
        except Exception:return False
        if len(ids['huginn'])<3900:return False
        if any(tuple(v[:128]) in prefixes[m] for m,v in ids.items()):return False
        rows.append(dict(prompt_id=str(identifier),source_dataset=dataset,source_SHA256=sha,messages=messages,
            original_token_lengths={m:len(v) for m,v in ids.items()},input_ids={m:v[:16384 if m=='ouro' else 8000 if m=='qwen8' else 3900] for m,v in ids.items()}))
        seen.add(sha)
        for m,v in ids.items():prefixes[m].add(tuple(v[:128]))
        if len(rows)%16==0:print('LONG_SOURCES',len(rows),'scanned',scanned,flush=True)
        return True
    for entry in corpus:
        messages=[]
        for turn in entry.get('conversations',[]):
            role={'human':'user','gpt':'assistant','system':'system','user':'user','assistant':'assistant'}.get(turn.get('from'))
            if role and isinstance(turn.get('value'),str):messages.append(dict(role=role,content=turn['value']))
        positions=[i for i,m in enumerate(messages) if m['role']=='user']
        if not positions:continue
        messages=messages[:positions[-1]+1]
        accept(messages,entry.get('id','sharegpt_'+str(scanned)),'ShareGPT')
        if len(rows)>=184:break
    sharegpt_count=len(rows)
    supplement=None
    if len(rows)<184:
        from huggingface_hub import hf_hub_download
        revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
        archive=Path(hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',revision=revision))
        supplement=dict(dataset='THUDM/LongBench',revision=revision,filename='data.zip',source_SHA256=digest(archive),
            scope='Actual public long documents. Explicit controlled tokenizer prefixes, not Azure traffic.')
        with zipfile.ZipFile(archive) as z:
            for task in ('qasper','multifieldqa_en','narrativeqa','hotpotqa','2wikimqa','musique','gov_report','qmsum'):
                matches=[n for n in z.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl']
                if not matches:continue
                values=[json.loads(s) for s in z.read(matches[0]).decode().splitlines()]
                random.Random(2026110102).shuffle(values);accepted=0
                for i,value in enumerate(values):
                    content='Document:\n'+value['context']+'\nQuestion:\n'+value.get('input','')
                    accepted+=accept([dict(role='user',content=content)],f'longbench_{task}_{value.get("_id",i)}','LongBench/'+task)
                    if accepted>=64 or len(rows)>=184:break
                if len(rows)>=184:break
    assert len(rows)>=40,('INSUFFICIENT_LONG_SOURCE_POOL',len(rows))
    random.Random(2026110103).shuffle(rows)
    write(frozen,rows)
    calibration=rows[:24];test=rows[24:]
    audit=[]
    for model,cells in GRID.items():
        lengths=sorted({n for n,cap in cells}|{128,512,1536})
        for length in lengths:
            pools={split:[r for r in pool if len(r['input_ids'][model])>=length] for split,pool in (('calibration',calibration),('test',test))}
            for split,pool in pools.items():
                values=[dict(prompt_id=r['prompt_id'],source_SHA256=r['source_SHA256'],
                    input_ids={model:r['input_ids'][model][:length]},length=length,
                    source_type='LENGTH_MATCHED_CONTROLLED_PUBLIC_TEXT_PREFIX',source_dataset=r['source_dataset'],truncation_explicit=True) for r in pool]
                write(ROOT/'data'/f'{split}_{model}_L{length}.json',values)
            a={tuple(r['input_ids'][model][:length]) for r in pools['calibration']}
            b={tuple(r['input_ids'][model][:length]) for r in pools['test']}
            assert not a.intersection(b),('CALIBRATION_TEST_PREFIX_OVERLAP',model,length)
            audit.append(dict(model=model,length=length,calibration_sources=len(pools['calibration']),test_sources=len(pools['test']),
                calibration_unique_prefixes=len(a),test_unique_prefixes=len(b),actual_prefix_overlap=0,
                optional_16K=length==16384,qualified_pool=len(pools['calibration'])>=8 and len(pools['test'])>=8))
    write(ROOT/'configs/dataset_provenance.json',dict(original=provenance,supplement=supplement,sharegpt_records=sharegpt_count,
        selection_seed=2026110101,stratified_pool_shuffle_seed=2026110103,
        source_records=len(rows),calibration_source_records=len(calibration),test_source_records=len(test),audit=audit,
        scope='Actual ShareGPT and public LongBench text, explicit tokenizer prefixes. Controlled exact lengths. Not Azure or natural EOS.',
        cyclic_test_reuse='If offered count exceeds the frozen pool, record source index modulo pool size explicitly.'))
    print('DATA_FROZEN',len(rows),flush=True)


if __name__=='__main__':main()
