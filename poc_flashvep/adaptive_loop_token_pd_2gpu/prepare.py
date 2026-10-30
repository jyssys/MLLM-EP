"""Frozen real-prompt manifests and diagnostic migration requests."""
import gzip
import json
import random
from .common import *

def base(model,arm,name,dataset,count,gpus):
    rows=json.loads((ROOT/'data'/dataset).read_text())[:count]
    depth=MODEL_PINS[model][2]
    return dict(name=name,model=model,arm=arm,dataset=dataset,count=count,gpus=gpus,
        depth=depth,cut=depth,axis='token' if arm.startswith('Token') else 'loop',
        dynamic=arm.endswith('Dynamic'),prefix_fraction=.5,capacity=4,state_capacity_tokens=4096,
        chunk=256,token_budget=256,graphs=True,asynchronous=model=='ouro',wavefront=True,
        kv_blocks=18432 if model=='huginn' else 8192,
        warmup_count=2,warmup_cap=2,cap=max(r['cap'] for r in rows),request_caps=[r['cap'] for r in rows],
        prompt_ids=[r['prompt_id'] for r in rows],arrival_offsets=[i*.004 for i in range(count)],
        seed=2026103101,timeout=3600,estimated_seconds=600,verify_transfer=False,state_probe=False,
        cut_candidates=[0,1,2,3,4] if model=='ouro' else [0,8,16,24,28,32],
        handoff_mode='native_G1' if arm=='PD_G1' else 'snapshot',ignore_eos=True)

def main():
    setup();prior=REPO/'poc_flashvep/reports/adaptive_recurrence_cut/data'
    for split in ('calibration','test'):
        pool=json.loads(gzip.decompress((prior/f'W1_{split}.json.gz').read_bytes()))
        rng=random.Random(2026103100+(split=='test'));rng.shuffle(pool)
        def pick(predicate,n):
            selected=[dict(r) for r in pool if predicate(r)]
            if len(selected)<n:raise ValueError((split,n,len(selected)))
            return selected[:n]
        bins=[pick(lambda r:all(800<=v<1500 for v in r['input_lengths'].values()),11),
              pick(lambda r:all(1500<=v<2800 for v in r['input_lengths'].values()),13),
              pick(lambda r:all(2800<=v<=3584 for v in r['input_lengths'].values()),8)]
        w1=sum(bins,[]);rng.shuffle(w1)
        w2=pick(lambda r:all(2000<=v<=3500 for v in r['input_lengths'].values()),24)
        w3=pick(lambda r:all(800<=v<=2000 for v in r['input_lengths'].values()),24)+pick(lambda r:all(3000<=v<=3500 for v in r['input_lengths'].values()),8)
        # Interleave eight long-prompt arrivals into a central burst segment.
        w3=w3[:12]+w3[24:]+w3[12:24]
        w4=pick(lambda r:all(800<=v<1500 for v in r['input_lengths'].values()),24)
        caps={'W1':[384]*8+[448]*16+[512]*8,'W2':[128,192,256]*8,
              'W3':[256]*8+[384]*16+[512]*8,'W4':[400,448,512]*8}
        for name,rows in zip(('W1','W2','W3','W4'),(w1,w2,w3,w4)):
            rng.shuffle(caps[name])
            for row,cap in zip(rows,caps[name]):
                row['cap']=cap
                if any(v+cap>4096 for v in row['input_lengths'].values()):raise ValueError('Context exceeded')
            write(ROOT/'data'/f'{name}_{split}.json',rows)
    diagnostic=[]
    for i,length in enumerate([128,512,1536,128,512,1536,512,1536]):
        source=json.loads((prior/f'prefix_{length}.json').read_text())[i]
        diagnostic.append(dict(prompt_id=f'gate_{length}_{i}',input_ids=source['input_ids'],cap=8))
    write(ROOT/'data/gate8.json',diagnostic)
    for model,gpus in [('ouro',[0,1]),('huginn',[4,5])]:
        for arm in ARMS:
            name=f'gate8_{model}_{arm.lower()}_v1'
            cfg=base(model,arm,name,'gate8.json',8,gpus)
            cfg.update(verify_transfer=True,state_probe=True,cut=MODEL_PINS[model][2]//2 if arm.startswith('Loop') else MODEL_PINS[model][2],
                       chunk=128,token_budget=128,timeout=1800)
            if arm=='DP':cfg.update(chunk=4096,token_budget=4096,wavefront=False)
            write(ROOT/'configs'/f'{name}.json',cfg)
    write(ROOT/'configs/dataset_contract.json',dict(prior_manifests=str(prior),source='Real ShareGPT full dialogue text. No Azure original prompts.',
        source_hashes={str(p):digest(p) for p in prior.glob('W1_*.json.gz')},
        bins_W1=[11,13,8],rounding='32 requests require integer bin counts. Actual percentages are34.375/40.625/25.',
        controlled_output='ignore_eos=True. No natural-EOS claim.',
        disjoint_calibration_test=True,diagnostic='Only gate8 uses previously frozen token prefixes. Main W1-W4 uses complete text.'))

if __name__=='__main__':main()
