"""Freeze disjoint public prompts and matched arrival traces."""
import random,zipfile
from .common import *


def offsets(count,rate,seed):
    rng=random.Random(seed);values=[0.]
    for _ in range(count-1):values.append(values[-1]+rng.expovariate(rate))
    return values


def base(name,arm='native',count=64,cap=16,rate=4.8,seed=2026111301,dataset='discovery.json'):
    cfg=json.loads((PREVIOUS/'configs/primed_confirm_qwen8_L8000_O16_N128_s31_native_v32.json').read_text())
    cfg.update(name=name,arm='DP2' if arm.startswith('coloc') else 'PD_G1',count=count,cap=cap,
        dataset=dataset,request_caps=[cap]*count,arrival_offsets=offsets(count,rate,seed),arrival_rate=rate,
        arrival_seed=seed,timeout=600,wall_timeout=1000,phase='CROSSOVER_DISCOVERY',
        warmup_batch_sizes=[1,2,4],warmup_output_cap=16,
        layer_mode=None,layer_cut=28,hidden_source_backpressure=False,reuse_prefill_token=arm=='native',
        serialize_D_imports=not arm.startswith('coloc'),cohort_barrier=not arm.startswith('coloc'),
        colocated_cohorts=arm=='coloc_cohort',D_admission_batch=4,chunk=8000,token_budget=32000,
        native_ACK_hold=True,
        scope='Fresh matched text-enabled experiment. Controlled public prompt prefixes. Finite-cohort rates are not sustainable capacity.')
    if arm.startswith('coloc'):cfg.update(reuse_prefill_token=False)
    return cfg


def prepare_data():
    initialize();target=ROOT/'data/discovery.json'
    if target.exists():return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    pool=json.loads((ATOMIC/'data/long_source_pool.json').read_text());seen={r['prompt_id'] for r in pool}
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip')
    extra=[]
    with zipfile.ZipFile(archive) as stream:
        for task in ('gov_report','qmsum','narrativeqa','qasper'):
            paths=[n for n in stream.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl']
            for index,line in enumerate(stream.read(paths[0]).decode().splitlines()):
                value=json.loads(line);identifier=f'longbench_{task}_{value.get("_id",index)}'
                if identifier in seen:continue
                messages=[dict(role='user',content='Document:\n'+value['context']+'\nQuestion:\n'+value.get('input',''))]
                ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
                if len(ids)<8000:continue
                sha=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
                extra.append(dict(prompt_id=identifier,source_dataset='LongBench/'+task,source_SHA256=sha,input_ids={'qwen8':ids}))
                seen.add(identifier)
                if len(extra)>=40:break
            if len(extra)>=40:break
    pool+=extra;random.Random(2026111300).shuffle(pool)
    tail=tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    check=tokenizer.apply_chat_template([dict(role='user',content='test')],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    assert check[-len(tail):]==tail,'Verify the actual disabled-thinking generation suffix'
    rows=[];unique=set()
    for row in pool:
        ids=row['input_ids']['qwen8'];assert len(ids)>=8000
        ids=ids[:8000-len(tail)]+tail
        if tuple(ids) in unique:continue
        unique.add(tuple(ids));rows.append(dict(prompt_id=row['prompt_id'],source_SHA256=row['source_SHA256'],
            source_dataset=row['source_dataset'],input_ids={'qwen8':ids},length=8000,
            source_type='CONTROLLED_PUBLIC_DOCUMENT_PREFIX_WITH_COMPLETE_GENERATION_SUFFIX',truncation_explicit=True,
            thinking_disabled_suffix_verified=True))
    assert len(rows)>=216
    groups=dict(qualification=rows[:8],calibration=rows[8:24],discovery=rows[24:88],heldout=rows[88:216])
    for name,values in groups.items():write(ROOT/'data'/(name+'.json'),values)
    assert not {r['prompt_id'] for r in groups['discovery']} & {r['prompt_id'] for r in groups['heldout']}
    write(ROOT/'data/manifest.json',dict(old_pool_SHA256=digest(ATOMIC/'data/long_source_pool.json'),
        LongBench_revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672',archive_SHA256=digest(archive),
        additional_sources=len(extra),tokenizer_revision=MODEL_REVISION,thinking_enabled=False,
        generation_suffix_ids=tail,generation_suffix_text=tokenizer.decode(tail),
        rule='Keep the real document prefix. Append the verified user-close and disabled-thinking generation suffix within exactly 8000 tokens.',
        groups={k:dict(count=len(v),prompt_ids=[r['prompt_id'] for r in v],SHA256=digest(ROOT/'data'/(k+'.json'))) for k,v in groups.items()},
        discovery_heldout_prompt_overlap=0,scope='New disjoint sets for this campaign. Earlier research can have used some original documents. No filler-token repetition.'))


if __name__=='__main__':prepare_data();print('DATA_READY',ROOT/'data/manifest.json')
