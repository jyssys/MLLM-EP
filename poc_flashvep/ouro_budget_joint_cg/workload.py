"""Freeze fresh source rows. Earlier held-out families never become new tests."""
import hashlib
import json
from .common import ROOT,PRIOR,MODEL,write
from ..ouro_adaptive_chunk_loop.workload import fetch

def main():
    target=ROOT/'source_family_manifest.json'
    assert not target.exists()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    alpaca,source_a=fetch('https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json')
    articles=[];sources=[source_a]
    for offset in (2000,2100):
        data,source=fetch(f'https://datasets-server.huggingface.co/rows?dataset=ccdv%2Farxiv-summarization&config=section&split=test&offset={offset}&length=100')
        articles.extend(data['rows']);sources.append(source)
    partitions=[('TRAIN',1000,1120,2000,2040),('VALIDATION',1120,1180,2040,2060),
        ('CALIBRATION',1180,1240,2060,2080),('NEW_HELDOUT',1240,1360,2080,2120),('FINAL_ONLINE',1360,1600,2120,2200)]
    cases=[]
    for split,start,stop,a,b in partitions:
        for i in range(start,stop):
            row=alpaca[i]
            content=row['instruction']+('\n'+row['input'] if row['input'] else '')+'\nGive a concise, complete answer.'
            text=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            cases.append(dict(id=f'alpaca_{i}',family=hashlib.sha256(f'alpaca:{i}'.encode()).hexdigest(),split=split,
                domain='conversation_proxy',input_ids=tokenizer.encode(text),source_row=i,text=text))
        for item in articles:
            i=item['row_idx']
            if not a<=i<b:
                continue
            tokens=tokenizer.encode(item['row']['article'])
            template=tokenizer.apply_chat_template([dict(role='user',content='Summarize this article excerpt in two complete sentences.\n<ARTICLE>\nSummary:')],tokenize=False,add_generation_prompt=True)
            left,right=template.split('<ARTICLE>');prefix,suffix=tokenizer.encode(left),tokenizer.encode(right)
            for length in (512,2033,8177,16384):
                count=length-len(prefix)-len(suffix)
                if len(tokens)<count:
                    continue
                ids=prefix+tokens[:count]+suffix
                cases.append(dict(id=f'arxiv_{i}_P{length}',family=hashlib.sha256(f'arxiv_test:{i}'.encode()).hexdigest(),
                    split=split,domain='article_excerpt',P=length,input_ids=ids,source_row=i,text=tokenizer.decode(ids)))
    previous=json.loads((PRIOR/'workload_manifest.json').read_text())
    old_families={r['family'] for r in previous['cases']}
    assert not old_families&{r['family'] for r in cases}
    write(target,dict(cases=cases,sources=sources,chat_template=tokenizer.chat_template,model_revision=MODEL.name,
        stop_ids=[tokenizer.convert_tokens_to_ids('<|im_end|>')],native_EOS=tokenizer.eos_token_id,
        selection='Input-only contiguous new rows. No source-text repetition. Every family remains in one partition.',
        previous_manifest_sha256=__import__('hashlib').sha256((PRIOR/'workload_manifest.json').read_bytes()).hexdigest(),
        greedy=True,static_R=4,serving_cap=512,quality_cap=4096))
    print('Frozen',len(cases),'new prompts')

if __name__=='__main__':
    main()
