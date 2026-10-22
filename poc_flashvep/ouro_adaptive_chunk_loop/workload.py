"""Input-only source-family partition. No candidate outputs select requests."""
import hashlib
import json
import urllib.request
from .common import ROOT, MODEL, write

def fetch(url):
    with urllib.request.urlopen(url,timeout=180) as response:
        data=response.read()
    return json.loads(data),dict(url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))

def main():
    target=ROOT/'workload_manifest.json'
    assert not target.exists(),'Never overwrite frozen input selection'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    alpaca,source_a=fetch('https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json')
    articles1,source_b=fetch('https://datasets-server.huggingface.co/rows?dataset=ccdv%2Farxiv-summarization&config=section&split=test&offset=40&length=100')
    sources=[source_a,source_b]
    cases=[]
    partitions=[('TRAIN',100,180,40,64),('VALIDATION',180,220,64,76),
        ('CALIBRATION',220,260,76,88),('HELDOUT',260,340,88,112),('FINAL_ONLINE',340,500,112,140)]
    for split,a_start,a_stop,ar_start,ar_stop in partitions:
        for i in range(a_start,a_stop):
            row=alpaca[i]
            content=row['instruction']+ ('\n'+row['input'] if row['input'] else '')+'\nGive a concise, complete answer.'
            text=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            cases.append(dict(id=f'alpaca_{i}',family=hashlib.sha256(f'alpaca:{i}'.encode()).hexdigest(),
                split=split,domain='conversation_proxy',input_ids=tokenizer.encode(text),source_row=i,text=text))
        for item in articles1['rows']:
            i=item['row_idx']
            if not ar_start<=i<ar_stop:
                continue
            tokens=tokenizer.encode(item['row']['article'])
            template=tokenizer.apply_chat_template([dict(role='user',content='Summarize this article excerpt in two complete sentences.\n<ARTICLE>\nSummary:')],tokenize=False,add_generation_prompt=True)
            left,right=template.split('<ARTICLE>')
            prefix,suffix=tokenizer.encode(left),tokenizer.encode(right)
            for length in (512,2033,4096,8177,16384):
                count=length-len(prefix)-len(suffix)
                if len(tokens)<count:
                    continue
                ids=prefix+tokens[:count]+suffix
                cases.append(dict(id=f'arxiv_{i}_P{length}',family=hashlib.sha256(f'arxiv_test:{i}'.encode()).hexdigest(),
                    split=split,domain='article_excerpt',P=length,input_ids=ids,source_row=i,text=tokenizer.decode(ids)))
    manifest=dict(cases=cases,sources=sources,selection='Input-only contiguous row ranges. No repetition to extend a short article. All lengths from one article share a partition.',
        model_revision=MODEL.name,chat_template=tokenizer.chat_template,native_EOS=tokenizer.eos_token_id,
        stop_ids=[tokenizer.convert_tokens_to_ids('<|im_end|>')],static_R=4,temperature=0,
        serving_cap=512,quantum_diagnostic_cap=32,quality_cap=4096)
    manifest['sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    write(target,manifest)
    print('FROZEN',len(cases),'cases',dict((s,sum(c['split']==s for c in cases)) for s,*_ in partitions))

if __name__=='__main__':
    main()
