"""Select naturally long public documents without repetition or filler."""
import hashlib,json,random,zipfile
from .common import *


def main():
    initialize()
    from transformers import AutoTokenizer
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/blobs/cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64')
    tokenizers={m:AutoTokenizer.from_pretrained(str(MODEL_PATHS[m]),trust_remote_code=True,local_files_only=True) for m in ('ouro','huginn')}
    rows=[];seen=set();scanned=0
    with zipfile.ZipFile(archive) as z:
        for task in ('gov_report','qmsum','narrativeqa','musique'):
            path=next((n for n in z.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl'),None)
            if path is None:continue
            values=list(map(json.loads,z.read(path).decode().splitlines()));random.Random(2026111220).shuffle(values)
            for index,value in enumerate(values):
                content='Document:\n'+value['context']+'\nQuestion:\n'+value.get('input','')
                sha=hashlib.sha256(content.encode()).hexdigest()
                if sha in seen:continue
                messages=[dict(role='user',content=content)];scanned+=1
                ids=tokenizers['ouro'].apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
                if len(ids)<16384:continue
                other=tokenizers['huginn'].apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
                assert len(other)>=3900
                rows.append(dict(prompt_id=f'longbench_{task}_{value.get("_id",index)}',source_dataset='LongBench/'+task,
                    source_SHA256=sha,original_token_lengths=dict(ouro=len(ids),huginn=len(other)),
                    input_ids=dict(ouro=ids[:16384],huginn=other[:3900])))
                seen.add(sha)
                if len(rows)%16==0:print('NATURAL_LONG_SOURCES',len(rows),'SCANNED',scanned,flush=True)
                if len(rows)>=152:break
            if len(rows)>=152:break
    assert len(rows)>=152,('INSUFFICIENT_DISTINCT_NATURAL_LONG_DOCUMENTS',len(rows))
    random.Random(2026111221).shuffle(rows)
    inventory=[]
    for model,lengths in (('ouro',(12000,16384)),('huginn',(3900,))):
        for length in lengths:
            for split,pool in (('natural_calibration',rows[:24]),('natural_discovery',rows[24:152])):
                name=f'{split}_{model}_L{length}.json';path=ROOT/'data'/name
                result=[dict(prompt_id=r['prompt_id'],source_dataset=r['source_dataset'],source_SHA256=r['source_SHA256'],
                    input_ids={model:r['input_ids'][model][:length]},length=length,original_token_lengths=r['original_token_lengths'],
                    source_type='NATURALLY_LONG_PUBLIC_DOCUMENT_CONTROLLED_PREFIX',truncation_explicit=True,
                    extension_or_filler=False) for r in pool]
                assert len({r['prompt_id'] for r in result})==len(result)
                if path.exists():assert json.loads(path.read_text())==result
                else:write(path,result)
                inventory.append(dict(file=name,count=len(result),distinct_prompt_ids=len(result),SHA256=digest(path)))
    write(ROOT/'state/natural_long_dataset_manifest.json',dict(archive=str(archive),archive_SHA256=digest(archive),
        dataset='THUDM/LongBench',revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672',scanned=scanned,
        task_order=['gov_report','qmsum','narrativeqa','musique'],selection_seed=2026111220,split_seed=2026111221,
        model_revisions={m:str(p) for m,p in MODEL_PATHS.items() if m in tokenizers},files=inventory,
        scope='Distinct naturally long original documents. Exact chat-tokenizer prefixes. No token repetition, concatenation, or filler. Calibration IDs and discovery IDs do not overlap.'))
    print('NATURAL_DATASET_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
