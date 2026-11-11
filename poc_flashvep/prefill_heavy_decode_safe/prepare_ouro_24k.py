"""Extend natural context length within the qualified model limit."""
import hashlib,json,random,zipfile
from .common import *


def main():
    initialize()
    from transformers import AutoTokenizer
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_PATHS['ouro']),trust_remote_code=True,local_files_only=True)
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/blobs/cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64')
    rows=[];seen=set();scanned=0;length=24576
    with zipfile.ZipFile(archive) as z:
        for task in ('narrativeqa','qmsum','gov_report','musique','qasper','triviaqa','multifieldqa_en','multi_news','repobench-p'):
            path=next((n for n in z.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl'),None)
            if path is None:continue
            values=list(map(json.loads,z.read(path).decode().splitlines()));random.Random(2026111270).shuffle(values)
            for index,value in enumerate(values):
                content='Document:\n'+value['context']+'\nQuestion:\n'+value.get('input','')
                sha=hashlib.sha256(content.encode()).hexdigest();scanned+=1
                if sha in seen:continue
                ids=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=True,
                    add_generation_prompt=True,return_dict=False)
                if len(ids)<length:continue
                rows.append(dict(prompt_id=f'longbench_{task}_{value.get("_id",index)}',source_dataset='LongBench/'+task,
                    source_SHA256=sha,original_token_lengths=dict(ouro=len(ids)),input_ids=dict(ouro=ids[:length]),
                    length=length,source_type='NATURALLY_LONG_PUBLIC_DOCUMENT_CONTROLLED_PREFIX',
                    truncation_explicit=True,extension_or_filler=False))
                seen.add(sha)
                if len(rows)%16==0:print('NATURAL_24K_SOURCES',len(rows),'SCANNED',scanned,flush=True)
                if len(rows)>=152:break
            if len(rows)>=152:break
    if len(rows)<152:
        write(ROOT/'state/natural_24k_insufficient_sources.json',dict(status='FAILED_DATA_PREPARATION',
            available=len(rows),required=152,scanned=scanned,
            scope='No filler or repeated prompts replace missing natural documents.'))
    assert len(rows)>=152,('INSUFFICIENT_DISTINCT_NATURAL_24K_DOCUMENTS',len(rows))
    random.Random(2026111271).shuffle(rows)
    for split,pool in (('calibration',rows[:24]),('discovery',rows[24:152])):
        path=ROOT/'data'/f'extended_natural_{split}_ouro_L24576.json'
        if path.exists():assert json.loads(path.read_text())==pool
        else:write(path,pool)
    write(ROOT/'state/natural_24k_dataset_manifest.json',dict(archive=str(archive),archive_SHA256=digest(archive),
        revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672',scanned=scanned,
        task_order=['narrativeqa','qmsum','gov_report','musique','qasper','triviaqa','multifieldqa_en','multi_news','repobench-p'],
        selection_seed=2026111270,split_seed=2026111271,distinct_calibration=24,distinct_discovery=128,
        scope='Natural source documents. No context extension, filler, repetition, or model modification. This is a new workload, not the 4096-token anchor.'))
    parent='async_ouro_L16384_native_reference_state8'
    cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());state='extended_ouro_L24576_native_reference_state8'
    cfg.update(name=state,length=length,cap=16,state_capacity_tokens=length,chunk=length,
        kv_blocks=20480,D_kv_blocks=20480,token_budget=2*length,D_token_budget=length,
        P_prefill_graph_rows=2*length,D_prefill_graph_rows=length,
        dataset='extended_natural_discovery_ouro_L24576.json',prompt_ids=[r['prompt_id'] for r in rows[24:32]],
        qualification_fresh_native=True,required_qualification=parent,
        phase='NATURAL_24K_NATIVE_FULL_STATE_REFERENCE',
        selection_reason='At 16384 tokens, the instrumented D maximum gap was 159.517 ms. That trace is diagnostic only. Extend natural input length to test larger service separation without reducing Output below 16.')
    cfg.pop('campaign_state_reference',None);cfg.pop('qualification_output_reference',None)
    assert cfg['length']+cfg['cap']<=cfg['context_limit'];save(cfg)
    cap=json.loads((ROOT/'configs'/'capacity_ouro_L16384_O16_B124_start1.json').read_text())
    name='extended_capacity_ouro_L24576_O16_B12_start1'
    cap.update(name=name,length=length,cap=16,state_capacity_tokens=length,chunk=length,kv_blocks=20480,
        D_kv_blocks=20480,token_budget=2*length,P_prefill_graph_rows=2*length,D_prefill_graph_rows=length,
        count=2,capacity=2,slots=2,P_slots=2,graph_rows=2,dataset='extended_natural_calibration_ouro_L24576.json',
        prompt_ids=[r['prompt_id'] for r in rows[:2]],arrival_offsets=[0.,0.],request_caps=[16,16],
        jobs=[dict(batch=b,cap=16,repeats=3) for b in (1,2)],required_qualification=state,
        phase='NATURAL_24K_MATCHED_NATIVE_FINITE_CAPACITY')
    save(cap)
    write(ROOT/'state/extended_24k_plan.json',dict(state8=state,capacity=name,jobs=[state,name],
        scope='Native discovery. Three complete D targets fit the legal 20480-block pool. Four do not. All SLO, model, and precision constraints remain frozen.'))
    print(state,name,flush=True)


if __name__=='__main__':main()
