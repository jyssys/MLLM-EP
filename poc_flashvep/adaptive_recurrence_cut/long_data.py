"""Prepare disjoint full-text common-support manifests. No GPU execution."""
import gzip
import hashlib
import json
import random
from .common import ROOT,REPO,MODELS,MODEL_PINS,write,digest


def main():
    from transformers import AutoTokenizer
    provenance=json.loads((REPO/'poc_flashvep/reports/looped_pd_b_slo_serving/configs/dataset_provenance.json').read_text())
    source=__import__('pathlib').Path(provenance['cache_path'])
    if digest(source)!=provenance['source_SHA256']:raise RuntimeError('Frozen ShareGPT checksum changed')
    conversations=json.loads(source.read_text())
    rng=random.Random(2026103069);rng.shuffle(conversations)
    tokenizers={m:AutoTokenizer.from_pretrained(MODELS[m],trust_remote_code=True,local_files_only=True) for m in ('ouro','huginn')}
    pools=[[],[],[]];limits=[(800,1500),(1500,3000),(3000,3585)];wanted=[140,160,100]
    seen=set();scanned=0
    for item in conversations:
        scanned+=1
        turns=item.get('conversations',[])
        positions=[i for i,t in enumerate(turns) if t.get('from') in ('human','user')]
        if not positions:continue
        end=rng.choice(positions);messages=[]
        for turn in turns[:end+1]:
            role={'human':'user','gpt':'assistant','user':'user','assistant':'assistant','system':'system'}.get(turn.get('from'))
            if role and isinstance(turn.get('value'),str):messages.append(dict(role=role,content=turn['value']))
        if not messages or messages[-1]['role']!='user':continue
        text=json.dumps(messages,sort_keys=True);sha=hashlib.sha256(text.encode()).hexdigest()
        if sha in seen:continue
        try:
            ids={}
            for m,tok in tokenizers.items():
                tokens=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
                ids[m]=list(tokens['input_ids'] if isinstance(tokens,dict) else tokens)
        except Exception:continue
        group=next((i for i,(a,b) in enumerate(limits) if all(a<=len(x)<b for x in ids.values())),None)
        if group is None or len(pools[group])>=wanted[group]:continue
        seen.add(sha)
        pools[group].append(dict(prompt_id=str(item.get('id',sha)),prompt_sha256=sha,messages=messages,
                                 input_ids=ids,input_lengths={m:len(v) for m,v in ids.items()},length_bin=group))
        if all(len(pools[i])>=wanted[i] for i in range(3)):break
    if any(len(pools[i])<wanted[i] for i in range(3)):
        raise RuntimeError(f'Insufficient complete prompts: {[len(x) for x in pools]}')
    for name,part in (('W1_calibration',0),('W1_test',1)):
        selected=[]
        for pool in pools:selected.extend(pool[len(pool)//2*part:len(pool)//2*(part+1)])
        random.Random(2026103070+part).shuffle(selected)
        caps=[384]*50+[448]*100+[512]*50;random.Random(2026103072+part).shuffle(caps)
        for row,cap in zip(selected,caps):row['controlled_output_cap']=cap
        dest=ROOT/'data'/f'{name}.json.gz'
        if dest.exists():raise FileExistsError(dest)
        dest.write_bytes(gzip.compress(json.dumps(selected).encode(),mtime=0))
    write(ROOT/'configs/long_data_provenance.json',dict(source=provenance,selection_seed=2026103069,
          scanned_conversations=scanned,counts={str(i):len(pool) for i,pool in enumerate(pools)},
          model_pins={m:MODEL_PINS[m] for m in tokenizers},common_support_prompt_limit=3584,
          rule='Complete dialogue through one selected user turn. Both tokenizers must satisfy the same bin. No token truncation.',
          controlled_generation='ignore_eos=True with caps384/448/512 in25/50/25percent proportions. Not natural output lengths.',
          natural_generation='Same prompts, EOS enabled, max512. NOT_RUN until later gates pass.',
          serving_status='PREPARED_NOT_EXECUTED',source_role='Real ShareGPT text. Not Azure request text or an Azure arrival trace.'))
    print('PREPARED_LONG_COMMON_SUPPORT',scanned,[len(x) for x in pools],flush=True)


if __name__=='__main__':main()
