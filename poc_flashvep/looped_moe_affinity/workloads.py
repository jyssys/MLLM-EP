"""Frozen public prompt IDs and explicitly labeled hardware-shape concatenation."""
from functools import lru_cache
import json
import torch
from pathlib import Path
from .common import ROOT,REPO,DATA,digest,write

@lru_cache(None)
def corpus():
    old=json.loads((REPO/'poc_flashvep/reports/looped_moe_headroom/workload_manifest.json').read_text())['rows']
    result=[]
    for row in old:
        if row['split'] not in ['calibration','heldout']:continue
        result.append(dict(id=row['id'],domain=dict(language='general',gsm8k='math',math500='math').get(row['dataset'],row['dataset']),
            text=row['text'],split=row['split']))
    # Public code only. No private workspace source becomes a benchmark prompt.
    paths=sorted((__import__('pathlib').Path('/home/esjung/external/xllm-loop-ep-poc/xllm')).rglob('*.py'))
    for i,path in enumerate(paths[:80]):
        text=path.read_text()
        if len(text)<256:continue
        result.append(dict(id='public_code/'+path.relative_to(Path('/home/esjung/external/xllm-loop-ep-poc')).as_posix(),
            domain='code',text=text,split='calibration' if i%4==0 else 'heldout'))
    for path in sorted((DATA/'paper_part1/mmlu').glob('*/*.jsonl')):
        split='calibration' if path.name=='dev.jsonl' else 'heldout'
        with path.open() as f:
            for i,line in enumerate(f):
                if i>=8:break
                row=json.loads(line)
                text=row['question']+'\n'+'\n'.join(f'{key}. {value}' for key,value in row['choices'].items())+'\nAnswer:'
                result.append(dict(id=f'mmlu/{path.parent.name}/{path.stem}/{i}',domain='knowledge',text=text,split=split))
    return result

def inputs(a,b,length,split='heldout',offset=0,domain_offset=0):
    items=corpus();domains=['general','knowledge','math','code']
    pools={d:[r for r in items if r['split']==split and r['domain']==d] for d in domains}
    selected=[pools[domains[(i+domain_offset)%len(domains)]][(offset+i//len(domains))%len(pools[domains[(i+domain_offset)%len(domains)]])] for i in range(b)]
    sequences=[]
    for item in selected:
        text=item['text']
        token=a.encode(text,length)
        while token.shape[-1]<length:
            text=text+'\n'+item['text'];token=a.encode(text,length)
        sequences.append(token[0])
    return torch.stack(sequences),selected

def provenance():
    items=corpus()
    write(ROOT/'runs/prompt_manifest.json',dict(rows=[dict(id=r['id'],domain=r['domain'],split=r['split'],
        text_sha256=__import__('hashlib').sha256(r['text'].encode()).hexdigest()) for r in items],
        shaping='Truncated or repeated public text to exact shape. Hardware workload, not natural task accuracy.'))

