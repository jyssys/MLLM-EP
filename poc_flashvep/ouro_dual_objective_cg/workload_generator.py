"""Freeze fresh natural input families. Never prescribe output token IDs."""
import gzip
import hashlib
import json
import random
import urllib.request
from .common import ROOT,PRIOR,MODEL,write,digest
from ..ouro_adaptive_chunk_loop.workload import fetch
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    assert not (ROOT/'workload_lock.json').exists(),'Preserve frozen inputs'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    url='https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/bdd27f4/databricks-dolly-15k.jsonl'
    with urllib.request.urlopen(url,timeout=180) as response: body=response.read()
    chat_rows=[json.loads(line) for line in body.splitlines() if line.strip()]
    a_source=dict(url=url,sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))
    articles=[];sources=[a_source]
    for offset in (4000,4100):
        payload,source=fetch(f'https://datasets-server.huggingface.co/rows?dataset=ccdv%2Farxiv-summarization&config=section&split=test&offset={offset}&length=100')
        articles.extend(payload['rows']);sources.append(source)
    old=json.loads((PRIOR/'source_family_manifest.json').read_text())
    old_families={r['family'] for r in old['cases']}
    partitions=[('TRAIN',0,3000,4000,4040,32),('VALIDATION',3000,6000,4040,4080,32),
        ('CALIBRATION',6000,9000,4080,4120,32),('FINAL',9000,len(chat_rows),4120,4200,96)]
    rng=random.Random(2026102301);cases=[];seen=set();families={}
    def add(source_id,family,split,domain,content):
        text=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        ids=tokenizer.encode(text)
        cases.append(dict(id=f'{source_id}_{domain}',source_id=source_id,family=family,split=split,
            domain=domain,input_ids=ids,prompt=text,input_tokens=len(ids),prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
        assert family not in families or families[family]==split
        families[family]=split
    for split,start,end,ar_start,ar_end,n in partitions:
        candidates=[]
        for i in range(start,end):
            row=chat_rows[i]
            content=row['instruction']+('\n'+row['context'] if row['context'] else '')+'\nAnswer clearly in one concise paragraph.'
            content_hash=hashlib.sha256(content.encode()).hexdigest()
            if content_hash in seen:continue
            count=len(tokenizer.encode(content))
            if 180<=count<=350:candidates.append((i,content,content_hash))
        assert len(candidates)>=n,(split,len(candidates),n)
        # Input-only selection. No generation or policy score chooses sources.
        rng.shuffle(candidates)
        for i,content,sha in candidates[:n]:
            family=hashlib.sha256(f'dolly:{i}'.encode()).hexdigest()
            assert family not in old_families
            seen.add(sha);add(f'dolly_{i}',family,split,'chat',content)
        for item in articles:
            index=item['row_idx']
            if not ar_start<=index<ar_end:continue
            article=item['row']['article'];sha=hashlib.sha256(article.encode()).hexdigest()
            if sha in seen:continue
            seen.add(sha)
            family=hashlib.sha256(f'arxiv_test:{index}'.encode()).hexdigest()
            assert family not in old_families
            tokens=tokenizer.encode(article)
            for domain,target in [('long',int(rng.lognormvariate(__import__('math').log(1760),.15))),('stress8k',7960)]:
                target=max(1200,min(2400,target)) if domain=='long' else target
                if len(tokens)<target:continue
                content='Read this scientific article excerpt. Summarize its method and main finding in one complete paragraph.\n\n'
                content+=tokenizer.decode(tokens[:target])
                content+='\n\nGive a concise summary, about 120 words.'
                add(f'arxiv_{index}',family,split,domain,content)
    payload=ROOT/'workload_payloads.json.gz'
    with gzip.open(payload,'wt') as stream:json.dump(dict(cases=cases),stream)
    csv_file(ROOT/'workload_manifest.csv',[{k:v for k,v in r.items() if k not in ('input_ids','prompt')} for r in cases])
    write(ROOT/'source_family_splits.json',dict(families=families,
        rule='Each source stays in one partition. Long/stress variants share a family.'))
    write(ROOT/'workload_lock.json',dict(sources=sources,payload_sha256=digest(payload),model_revision=MODEL.name,
        tokenizer_config_sha256=digest(MODEL/'tokenizer_config.json'),chat_template=tokenizer.chat_template,
        EOS=tokenizer.eos_token_id,frontend_stop_ids=[tokenizer.convert_tokens_to_ids('<|im_end|>')],
        greedy=True,static_R=4,natural_output=True,cap=1024,quality_cap=4096,
        target_outputs='Chat~96 and long~180 are intentions, not forced lengths. Record actual distributions.',
        workload_source_substitution='Dolly instruction/context paragraphs as chat-like inputs, arXiv excerpts as long inputs. Alpaca input-only filtering failed before registration.',
        source_selection_seed=2026102301,cases=len(cases),families=len(families),
        old_source_manifest_sha256=digest(PRIOR/'source_family_manifest.json')))
    print('Frozen fresh input variants',len(cases),'source families',len(families))

if __name__=='__main__':main()
