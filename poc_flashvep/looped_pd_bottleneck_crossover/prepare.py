"""Freeze controlled ShareGPT prefixes and the initial discovery contract."""
import hashlib
import random
from .common import *

GRID={'G1':(2048,128),'G2':(3072,128),'G3':(3500,128),'G4':(3072,192),
      'G5':(3500,192),'G6':(3072,256),'C1':(1536,448),'C2':(1900,448),
      'U1':(1024,448),'U2':(2048,448),'U3':(512,128),'D32':(3500,32),'D64':(3500,64)}

def base(model,name):
    return dict(name=name,model=model,arm='PD_G1',depth=MODEL_PINS[model][2],cut=MODEL_PINS[model][2],
        state_capacity_tokens=4096,capacity=8,slots=8,chunk=256,graphs=True,wavefront=True,
        asynchronous=model=='ouro',kv_blocks=8192 if model=='ouro' else 18432,
        seed=2026103107,gpus=[0] if model=='ouro' else [4],phase='capacity_gate',
        timeout=1800,barrier=True,verify_native=True,state_probe=False,groups=[],worker='capacity_worker')

def main():
    from transformers import AutoTokenizer
    setup();frozen=ROOT/'data/source_prompts.json'
    if not frozen.exists():
        provenance=json.loads((REPO/'poc_flashvep/reports/looped_pd_b_slo_serving/configs/dataset_provenance.json').read_text())
        path=Path(provenance['cache_path']);assert digest(path)==provenance['source_SHA256']
        corpus=json.loads(path.read_text());random.Random(2026103109).shuffle(corpus)
        tokenizers={m:AutoTokenizer.from_pretrained(p[0],revision=p[1],trust_remote_code=True,local_files_only=True) for m,p in MODEL_PINS.items()}
        tokenizers['qwen']=AutoTokenizer.from_pretrained(QWEN_PIN[0],revision=QWEN_PIN[1],local_files_only=True)
        selected=[];seen=set()
        for item in corpus:
            messages=[]
            for turn in item.get('conversations',[]):
                role={'human':'user','gpt':'assistant','user':'user','assistant':'assistant','system':'system'}.get(turn.get('from'))
                if role and isinstance(turn.get('value'),str):messages.append(dict(role=role,content=turn['value']))
            user_positions=[i for i,v in enumerate(messages) if v['role']=='user']
            if not user_positions:continue
            messages=messages[:user_positions[-1]+1]
            encoded=json.dumps(messages,sort_keys=True);sha=hashlib.sha256(encoded.encode()).hexdigest()
            if sha in seen or not 13000<len(encoded)<180000:continue
            ids={}
            try:
                for m,tok in tokenizers.items():
                    values=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                        return_dict=False,**({'enable_thinking':False} if m=='qwen' else {}))
                    ids[m]=list(values['input_ids'] if isinstance(values,dict) else values)
            except Exception:continue
            if min(map(len,ids.values()))<3500:continue
            selected.append(dict(prompt_id=str(item.get('id',sha)),prompt_sha256=sha,
                original_token_lengths={m:len(v) for m,v in ids.items()},input_ids={m:v[:3500] for m,v in ids.items()},
                original_messages=messages,source='ShareGPT frozen source',workload_type='LENGTH_MATCHED_CONTROLLED_SHAREGPT_PREFIX',
                truncation_explicit=True,model_generation_context_limit=4096))
            seen.add(sha)
            if len(selected)==64:break
        assert len(selected)==64,'Insufficient real long-prompt sources'
        write(frozen,selected)
        write(ROOT/'configs/dataset_provenance.json',dict(original=provenance,selection_seed=2026103109,
            calibration_count=32,test_count=32,calibration_test_disjoint=True,
            scope='Explicit tokenizer-prefix length matching from real ShareGPT text. Not natural full-dialogue lengths or Azure requests.',
            Qwen_prompt_mode='Official chat template, enable_thinking=False, then explicitly selected prefix.'))
    source=json.loads(frozen.read_text())
    for split,rows in (('calibration',source[:32]),('test',source[32:])):
        for length in sorted({v[0] for v in GRID.values()}|{128,512,1536}):
            name=ROOT/'data'/f'{split}_L{length}.json'
            if not name.exists():write(name,[dict(r,input_ids={m:x[:length] for m,x in r['input_ids'].items()},input_length=length) for r in rows])
    for model in ('ouro','huginn'):
        name=f'gate_{model}_barrier_v1';cfg=base(model,name);cfg.update(state_probe=True)
        cfg['groups']=[dict(chunk=256,kv_blocks=cfg['kv_blocks'],jobs=[
            dict(dataset='test_L512.json',batch=2,cap=16,repeats=1),
            dict(dataset='test_L128.json',batch=4,cap=8,repeats=1),
            dict(dataset='test_L1536.json',batch=2,cap=16,repeats=1)])]
        if not (ROOT/'configs'/f'{name}.json').exists():write(ROOT/'configs'/f'{name}.json',cfg)
    contract=ROOT/'configs/discovery_contract.json'
    if not contract.exists():write(contract,dict(time=time.time(),core_grid={k:v for k,v in GRID.items() if k.startswith('G') or k in ('C1','C2')},
        optional_controls={k:v for k,v in GRID.items() if k.startswith('U') or k.startswith('D')},
        phase_A_boundary='Measure both pre-CODA and post-first-token service. D-rest times exactly output_target-1 tokens from valid resident KV.',
        phase_A_chunks=[256,512,1024,2048],phase_A_batches=[1,2,4,8],default_KV_blocks=dict(ouro=8192,huginn=18432),
        diagnostic_expanded_Huginn_blocks=26624,clean_lane='One GPU or one2GPU pair at a time. No other managed GPU load.',
        phase_B_rate_factors=[.4,.6,.8,1.,1.2],minimum_screening_requests=24,selected_requests=64,
        classification='Separate D reservation waiting from actual P compute queue. A finite trace does not establish infinite-horizon stability.',
        C_gate='Actual P compute queue growth near independently measured P capacity plus D compute, KV, and Decode SLO slack.',
        SLO_rule='Freeze from disjoint low-rate PD calibration before held-out serving. Record TTFT, average TPOT, and ITL sensitivity independently.',
        budget_reduction='If pilot runtime exceeds allocation, freeze G2/G3/G5/C1 symmetrically for both Looped models.'))
    print('DATA_AND_INITIAL_GATES_READY',flush=True)

if __name__=='__main__':main()
