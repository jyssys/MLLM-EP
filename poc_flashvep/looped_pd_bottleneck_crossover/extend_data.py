"""Add held-out prompts without changing the frozen calibration or screening sets."""
import hashlib
import random
from .common import *
from .prepare import GRID

def main():
    from transformers import AutoTokenizer
    frozen=json.loads((ROOT/'data/source_prompts.json').read_text())
    out=ROOT/'data/heldout_extension32.json'
    if not out.exists():
        original=json.loads((ROOT/'configs/dataset_provenance.json').read_text())['original']
        path=Path(original['cache_path']);assert digest(path)==original['source_SHA256']
        corpus=json.loads(path.read_text());random.Random(2026103109).shuffle(corpus)
        tokenizers={m:AutoTokenizer.from_pretrained(p[0],revision=p[1],trust_remote_code=True,local_files_only=True) for m,p in MODEL_PINS.items()}
        tokenizers['qwen']=AutoTokenizer.from_pretrained(QWEN_PIN[0],revision=QWEN_PIN[1],local_files_only=True)
        seen={r['prompt_sha256'] for r in frozen};rows=[]
        for item in corpus:
            messages=[]
            for turn in item.get('conversations',[]):
                role={'human':'user','gpt':'assistant','user':'user','assistant':'assistant','system':'system'}.get(turn.get('from'))
                if role and isinstance(turn.get('value'),str):messages.append(dict(role=role,content=turn['value']))
            positions=[i for i,v in enumerate(messages) if v['role']=='user']
            if not positions:continue
            messages=messages[:positions[-1]+1];text=json.dumps(messages,sort_keys=True);sha=hashlib.sha256(text.encode()).hexdigest()
            if sha in seen or not 13000<len(text)<180000:continue
            ids={}
            try:
                for m,tok in tokenizers.items():
                    values=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False,
                        **({'enable_thinking':False} if m=='qwen' else {}))
                    ids[m]=list(values['input_ids'] if isinstance(values,dict) else values)
            except Exception:continue
            if min(map(len,ids.values()))<3500:continue
            rows.append(dict(prompt_id=str(item.get('id',sha)),prompt_sha256=sha,
                original_token_lengths={m:len(v) for m,v in ids.items()},input_ids={m:v[:3500] for m,v in ids.items()},
                original_messages=messages,source='ShareGPT frozen source',workload_type='LENGTH_MATCHED_CONTROLLED_SHAREGPT_PREFIX',truncation_explicit=True))
            seen.add(sha)
            if len(rows)==32:break
        assert len(rows)==32;write(out,rows)
        write(ROOT/'configs/heldout_extension_provenance.json',dict(time=time.time(),count=32,source_SHA256=original['source_SHA256'],
            selection_seed=2026103109,disjoint_from_all_original_64=True,existing_source_prompts_SHA256=digest(ROOT/'data/source_prompts.json')))
    rows=frozen[32:]+json.loads(out.read_text());assert len(rows)==64
    for length in sorted({v[0] for v in GRID.values()}|{128,512,1536}):
        path=ROOT/'data'/f'test64_L{length}.json'
        if not path.exists():write(path,[dict(r,input_ids={m:x[:length] for m,x in r['input_ids'].items()},input_length=length) for r in rows])
    print('64_DISTINCT_HELDOUT_PROMPTS_READY')

if __name__=='__main__':main()
