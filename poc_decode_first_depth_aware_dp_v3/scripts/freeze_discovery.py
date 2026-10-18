"""Freeze diverse request metadata before decode or scheduling analysis."""
import hashlib
import json
import random
import urllib.request
from poc_decode_first_depth_aware_dp_v3.contract import *

def fetch(url):
    with urllib.request.urlopen(url,timeout=90) as response:
        raw=response.read()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()

def main():
    target=ROOT/'artifacts/discovery_frozen.json'
    assert not target.exists()
    rows=[]
    sources={}
    old=read(OLD_ROOT/'artifacts/quality_protocol_v3_pilot32.json')
    for domain in ['gsm8k','math500']:
        candidates=[r for r in old['rows'] if r['dataset']==domain and r['split']!='pilot']
        candidates=sorted(candidates,key=lambda r:r['id'])
        random.Random(20261018).shuffle(candidates)
        for i,r in enumerate(candidates[:32]):
            rows.append(dict(id='discovery/'+r['id'],dataset=domain,prompt=r['prompt'],
                original_id=r['id'],split='calibration' if i<8 else 'heldout',stop_strings=[]))
    info,_=fetch('https://api.github.com/repos/tatsu-lab/stanford_alpaca/commits/main')
    commit=info['sha']
    url=f'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/{commit}/alpaca_data.json'
    data,digest=fetch(url)
    sources['alpaca']=dict(url=url,revision=commit,response_sha256=digest)
    indices=list(range(len(data)))
    random.Random(20261018).shuffle(indices)
    for i,index in enumerate(indices[:32]):
        r=data[index]
        prompt='### Instruction:\n'+r['instruction']
        if r['input']:prompt+='\n\n### Input:\n'+r['input']
        prompt+='\n\n### Response:\n'
        rows.append(dict(id=f'alpaca/train/{index}',dataset='alpaca',prompt=prompt,
            split='calibration' if i<8 else 'heldout',stop_strings=[]))
    repo='learnanything/sharegpt_v3_unfiltered_cleaned_split'
    info,_=fetch('https://huggingface.co/api/datasets/'+repo)
    url='https://datasets-server.huggingface.co/rows?dataset=learnanything%2Fsharegpt_v3_unfiltered_cleaned_split&config=default&split=train&offset=0&length=100'
    data,digest=fetch(url)
    sources['sharegpt']=dict(url=url,revision=info['sha'],response_sha256=digest,
        limitation='Dataset viewer supplies the first 100 rows. The exact selected content is frozen here.')
    candidates=[]
    for r in data['rows']:
        value=r['row']
        messages=value.get('conversations',[])
        human=next((m.get('value','') for m in messages if m.get('from') in ['human','user']),None)
        if human and len(human)<=12000:
            candidates.append((r['row_idx'],human))
    assert len(candidates)>=32,len(candidates)
    random.Random(20261018).shuffle(candidates)
    for i,(index,human) in enumerate(candidates[:32]):
        rows.append(dict(id=f'sharegpt/train/{index}',dataset='sharegpt',
            prompt='### Human:\n'+human+'\n\n### Assistant:\n',
            split='calibration' if i<8 else 'heldout',stop_strings=[]))
    # Interleave domains. Partitions must not put one entire domain on one GPU.
    groups=[[r for r in rows if r['dataset']==d] for d in ['gsm8k','math500','alpaca','sharegpt']]
    rows=[groups[d][i] for i in range(32) for d in range(4)]
    for r in rows:r['prompt_sha256']=hashlib.sha256(r['prompt'].encode()).hexdigest()
    write(target,dict(rows=rows,sources=sources,seed=20261018,cap=256,
        natural_EOS=True,quality_claim=False,split_frozen_before_inference=True,
        prior_quality_protocol_sha256=sha(OLD_ROOT/'artifacts/quality_protocol_v3_pilot32.json')))
    print('FROZEN',len(rows))

if __name__=='__main__':main()
