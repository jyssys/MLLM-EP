"""Audit original candidates before any v2 serving-result selection."""
import argparse
import collections
import hashlib
import json
import time
import zipfile
from pathlib import Path
from .common import ROOT,PREVIOUS,REPO,MODEL,REVISION,write,digest,initialize

OUTPUT_BINS=((1,10),(11,50),(51,100),(101,200))
EXPANSION=('vcsum','lcc','repobench-p','trec','lsht','passage_count','passage_retrieval_en','passage_retrieval_zh')
SEED=2026112201

def bucket(length):
    return next((i for i,(low,high) in enumerate(OUTPUT_BINS) if low<=length<=high),None)

def initial():
    initialize()
    target=ROOT/'configs/dataset_selection_contract_v2.json'
    assert not target.exists(),'Do not overwrite a frozen selection rule'
    pool=PREVIOUS/'data/candidate_pool_original.json';ref=PREVIOUS/'data/natural_reference_all_candidates.json'
    inventory=PREVIOUS/'data/candidate_input_inventory.json'
    declaration=dict(time=time.time(),name='LongBench-Balanced400-v2',source_revision=REVISION,
        input_bins=[[3000,5999],[6000,9999],[10000,19999],[20000,36000]],output_bins=OUTPUT_BINS,
        target_per_cell=25,target_per_input=100,selection_seed=SEED,
        prior_pool_SHA256=digest(pool),prior_reference_SHA256=digest(ref),inventory_SHA256=digest(inventory),
        expansion_order=EXPANSION,expansion='All feasible original prompts in all eight remaining non-E official tasks. No outcome-based stopping within a task.',
        reference='Reuse BI0 C1 original full36 Mode0 prompt plus native decode graph natural EOS references. Generate expansion with the same contract. C3 compiled lengths may differ.',
        output_cap=2048,ignore_eos=False,thinking_disabled=True,
        ranking='Initial13 tasks before expansion,then expansion order,then SHA256(seed+prompt_id). No quality-score or serving-speed filtering.',
        shortage='Within each input bin take up to25 in every output bin. For each deficient output bin,fill from remaining original requests in the same input bin by output-bin distance,then deterministic rank. Never borrow from another input bin unless fewer than100 eligible originals remain; report a dataset shortfall instead.',
        split='Context-disjoint hash split. Original candidates have prior C1 exposure and may have prior C0/C3 exposure. Do not call the entire dataset globally unseen.',
        quality='Official task metrics are reported separately. Classification,code and Chinese tasks can enter the expanded pool; task mix and bias remain explicit.')
    write(target,declaration)
    p=json.loads(pool.read_text());r=json.loads(ref.read_text())['rows'];b={x['prompt_id']:x for x in r}
    counts=[[0]*4 for _ in range(4)]
    for x in p:
        y=b[x['prompt_id']];assert x['input_ids_SHA256']==y['input_ids_SHA256']
        j=bucket(y['output_tokens'])
        if j is not None:counts[x['input_bucket']][j]+=1
    write(ROOT/'analysis/pre_expansion_candidate_availability.json',dict(time=time.time(),n=len(p),cells=counts,
        row_totals=list(map(sum,counts)),outside_output_bins=len(p)-sum(map(sum,counts)),evidence='REANALYZED_NATURAL_REFERENCE',
        source_pool_SHA256=digest(pool),source_reference_SHA256=digest(ref),selected400=False))
    print('PRE_EXPANSION_AVAILABILITY',counts,flush=True)

def census():
    initialize();assert (ROOT/'configs/dataset_selection_contract_v2.json').exists()
    assert not (ROOT/'data/candidate_pool_original.json').exists()
    old=json.loads((PREVIOUS/'data/candidate_input_inventory.json').read_text())
    eligible={r['prompt_id']:r for r in old['rows'] if r.get('task') in EXPANSION and r.get('input_bucket') is not None}
    source=REPO/'poc_flashvep/reports/dense_ar_native_pd_real_workload_reproduction_20261119/references/longbench_v1_dataset2prompt.json'
    templates=json.loads(source.read_text())
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots')/REVISION/'data.zip'
    assert digest(archive)==old['source_zip_SHA256']
    from transformers import AutoTokenizer
    from ..dense_ar_longbench400_pd.candidate_census import token_hash
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    rows=[]
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            task=Path(member).stem
            if task not in EXPANSION or not member.endswith('.jsonl'):continue
            for index,line in enumerate(bundle.read(member).decode().splitlines()):
                row=json.loads(line);pid=f'longbench_v1_{task}_{row["_id"]}'
                if pid not in eligible:continue
                before=eligible[pid];prompt=templates[task].format(**row)
                ids=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                assert len(ids)==before['input_tokens'] and hashlib.sha256(prompt.encode()).hexdigest()==before['prompt_text_SHA256']
                assert len(ids)+2048<=40960
                value=dict(before,excluded=None,sample_id=len(rows),source_dataset='zai-org/LongBench',source_revision=REVISION,
                    original_context=row['context'],original_input=row['input'],reference_answers=row['answers'],
                    rendered_prompt_full=prompt,input_tokens_original=len(ids),input_ids=dict(qwen8=ids),input_ids_SHA256=token_hash(ids),
                    generated_output_tokens=None,output_cap=2048,ignore_eos=False,thinking_disabled=True,
                    category='CODE' if task in ('lcc','repobench-p') else 'CLASSIFICATION' if task in ('trec','lsht') else 'SUMMARIZATION' if task=='vcsum' else 'QA',
                    language='zh' if task in ('vcsum','lsht','passage_retrieval_zh') else 'en',expansion_task=True)
                rows.append(value)
            print('V2_EXPANSION_TASK',task,len(rows),flush=True)
    assert len(rows)==len(eligible)==len({x['prompt_id'] for x in rows})
    write(ROOT/'data/candidate_pool_original.json',rows)
    for gpu in range(8):write(ROOT/'data'/f'candidate_reference_shard{gpu}.json',rows[gpu::8])
    write(ROOT/'state/candidate_input_census_complete.json',dict(time=time.time(),passed=True,count=len(rows),
        SHA256=digest(ROOT/'data/candidate_pool_original.json'),scope='Expansion only. Prior2165 natural references remain separate and immutable.'))
    print('V2_EXPANSION_FROZEN',len(rows),flush=True)

def merged():
    initialize();assert json.loads((ROOT/'state/candidate_natural_reference_complete.json').read_text())['passed']
    prior=json.loads((PREVIOUS/'data/candidate_pool_original.json').read_text())
    extra=json.loads((ROOT/'data/candidate_pool_original.json').read_text())
    refs=json.loads((PREVIOUS/'data/natural_reference_all_candidates.json').read_text())['rows']
    for path in sorted((ROOT/'raw/CandidateNaturalReference_v2extra').glob('GPU*/natural_reference.jsonl')):
        refs.extend(json.loads(line) for line in path.read_text().splitlines())
    pool=prior+extra;byid={r['prompt_id']:r for r in refs}
    assert len(pool)==len(refs)==len(byid)==len({r['prompt_id'] for r in pool})
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots')/REVISION/'data.zip'
    classes={}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            task=Path(member).stem
            if task not in ('trec','lsht') or not member.endswith('.jsonl'):continue
            for line in bundle.read(member).decode().splitlines():
                row=json.loads(line);classes[f'longbench_v1_{task}_{row["_id"]}']=row['all_classes']
    cells=[[0]*4 for _ in range(4)]
    for row in pool:
        ref=byid[row['prompt_id']];assert ref['input_ids_SHA256']==row['input_ids_SHA256']
        if row['task'] in ('trec','lsht'):row['all_classes']=classes[row['prompt_id']]
        j=bucket(ref['output_tokens'])
        if j is not None:
            assert ref['natural_EOS'] and not ref['cap_hit']
            cells[row['input_bucket']][j]+=1
    write(ROOT/'data/combined_original_candidate_pool_v2.json',pool)
    write(ROOT/'data/natural_reference_all_candidates.json',dict(rows=refs,source_version='prior_v1_plus_v2extra',
        evidence='NATURAL_OUTPUT_REFERENCE',scope='C1 output lengths,not C3 serving metrics.'))
    record=dict(time=time.time(),n=len(pool),cells=cells,row_totals=list(map(sum,cells)),selected400=False,
        task_counts=dict(collections.Counter(r['task'] for r in pool)),pool_cap_hits=sum(r['cap_hit'] for r in refs),
        outside_output_bins=len(refs)-sum(map(sum,cells)),pool_SHA256=digest(ROOT/'data/combined_original_candidate_pool_v2.json'),
        reference_SHA256=digest(ROOT/'data/natural_reference_all_candidates.json'),evidence='MEASURED_NATURAL_REFERENCE')
    write(ROOT/'analysis/post_expansion_candidate_availability.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)

def freeze():
    import numpy as np
    initialize();assert not (ROOT/'data/longbench_balanced400_v2.json').exists()
    available=json.loads((ROOT/'analysis/post_expansion_candidate_availability.json').read_text())
    assert (ROOT/'state/candidate_counts_reported_before_freeze.json').exists(),'Report actual counts before selecting400'
    assert all(n>=100 for n in available['row_totals'])
    pool=json.loads((ROOT/'data/combined_original_candidate_pool_v2.json').read_text())
    refs=json.loads((ROOT/'data/natural_reference_all_candidates.json').read_text())['rows'];byid={r['prompt_id']:r for r in refs}
    def rank(row):return (1+EXPANSION.index(row['task']) if row['task'] in EXPANSION else 0,
        hashlib.sha256((str(SEED)+row['prompt_id']).encode()).hexdigest())
    cells=collections.defaultdict(list)
    for row in pool:
        j=bucket(byid[row['prompt_id']]['output_tokens'])
        if j is not None:cells[row['input_bucket'],j].append(row)
    for rows in cells.values():rows.sort(key=rank)
    selected=[];used=set();moves=[]
    for i in range(4):
        shortage=[]
        for j in range(4):
            take=cells[i,j][:25];selected.extend(take);used.update(r['prompt_id'] for r in take)
            if len(take)<25:shortage.append((j,25-len(take)))
        for j,count in shortage:
            rest=[r for jj in range(4) for r in cells[i,jj] if r['prompt_id'] not in used]
            rest.sort(key=lambda r:(abs(bucket(byid[r['prompt_id']]['output_tokens'])-j),rank(r)))
            take=rest[:count];assert len(take)==count
            selected.extend(take);used.update(r['prompt_id'] for r in take)
            moves.append(dict(input_bucket=i,target_output=j,shortage=count,
                allocations=[dict(prompt_id=r['prompt_id'],output_bucket=bucket(byid[r['prompt_id']]['output_tokens'])) for r in take]))
    assert len(selected)==len(used)==400
    selected.sort(key=lambda r:hashlib.sha256(('v2-arrival-'+str(SEED)+r['prompt_id']).encode()).hexdigest())
    for index,row in enumerate(selected):
        ref=byid[row['prompt_id']]
        row.update(sample_id=index,natural_output_tokens=ref['output_tokens'],natural_output_bucket=bucket(ref['output_tokens']),
            natural_output_ids_SHA256=hashlib.sha256(json.dumps(ref['output_ids']).encode()).hexdigest(),
            evaluation_split='HELDOUT' if int(hashlib.sha256(('v2-holdout-'+str(SEED)+row['context_SHA256']).encode()).hexdigest(),16)%2 else 'DISCOVERY')
    # Check mechanism feasibility before publishing the frozen main manifest.
    from ..dense_ar_longbench400_pd.mechanism_sets import exact_subset
    for split in ('DISCOVERY','HELDOUT'):
        for i in range(4):
            rows=[r for r in selected if r['evaluation_split']==split and r['input_bucket']==i]
            assert len(rows)>=16,(split,i,len(rows))
            _,ok=exact_subset(rows);assert ok,('No exact16 subset in the declared context split',split,i)
    target=ROOT/'data/longbench_balanced400_v2.json';write(target,selected)
    for split in ('discovery','heldout'):
        part=[r for r in selected if r['evaluation_split']==split.upper()]
        write(ROOT/'data'/f'balanced400_v2_{split}.json',part)
        for i in range(4):write(ROOT/'data'/f'mechanism_input{i}_{split}.json',[r for r in part if r['input_bucket']==i])
    contexts={r['context_SHA256'] for r in selected};remainder=[r for r in pool if r['context_SHA256'] not in contexts]
    remainder.sort(key=lambda r:hashlib.sha256(('v2-calibration-'+r['prompt_id']).encode()).hexdigest())
    q8=[r for i in range(4) for r in [x for x in remainder if x['input_bucket']==i][:2]]
    q32=q8+[r for r in remainder if r['prompt_id'] not in {x['prompt_id'] for x in q8}][:24]
    for name,rows in [('qualification1',[min(remainder,key=lambda r:r['input_tokens'])]),('qualification8',q8),('qualification32',q32)]:
        assert len(rows)==int(name.removeprefix('qualification'));write(ROOT/'data'/(name+'.json'),rows)
    from ..dense_ar_longbench400_pd.mechanism_sets import exact_subset
    exact=[]
    for split in ('discovery','heldout'):
        for i in range(4):
            rows=[r for r in selected if r['evaluation_split']==split.upper() and r['input_bucket']==i]
            assert len(rows)>=16,(split,i,len(rows))
            chosen,ok=exact_subset(rows);assert ok,('Exact aggregate rows unavailable without prompt modification',split,i)
            name=f'input{i}_{split}_exact16.json';write(ROOT/'data'/name,chosen)
            exact.append(dict(split=split,input_bucket=i,dataset=name,n=16,input_sum=sum(r['input_tokens'] for r in chosen),SHA256=digest(ROOT/'data'/name)))
    write(ROOT/'data/mechanism_exact16_sets_frozen_v2.json',dict(rows=exact,scope='Original prompts; selected using identity and aggregate input-length residue,not serving outcomes.'))
    def stats(v):
        a=np.asarray(v);return dict(n=len(v),mean=float(a.mean()),median=float(np.median(a)),min=int(a.min()),max=int(a.max()),
            **{'p'+str(p):float(np.percentile(a,p)) for p in (10,25,75,90,95,99)})
    counts=[[sum(r['input_bucket']==i and r['natural_output_bucket']==j for r in selected) for j in range(4)] for i in range(4)]
    oldids={r['prompt_id'] for r in json.loads((PREVIOUS/'data/longbench_balanced400_v1.json').read_text())}
    manifest=dict(time=time.time(),name='LongBench-Balanced400-v2',n=400,SHA256=digest(target),available_cells=available['cells'],selected_cells=counts,
        selection_contract_SHA256=digest(ROOT/'configs/dataset_selection_contract_v2.json'),seed=SEED,source_revision=REVISION,
        input=stats([r['input_tokens'] for r in selected]),output=stats([r['natural_output_tokens'] for r in selected]),
        tasks=dict(collections.Counter(r['task'] for r in selected)),languages=dict(collections.Counter(r['language'] for r in selected)),
        input_totals=list(map(sum,counts)),output_totals=[sum(row[j] for row in counts) for j in range(4)],shortage_reallocation=moves,
        unique_requests=400,unique_contexts=len(contexts),request_duplicate_count=0,context_duplicate_requests=400-len(contexts),
        new_truncation_count=0,selected_EOS=400,selected_cap_hits=0,total_input_tokens=sum(r['input_tokens'] for r in selected),
        total_output_tokens=sum(r['natural_output_tokens'] for r in selected),v1_request_overlap=sum(r['prompt_id'] in oldids for r in selected),
        split_counts=dict(collections.Counter(r['evaluation_split'] for r in selected)),
        split_input_counts={s:[sum(r['evaluation_split']==s and r['input_bucket']==i for r in selected) for i in range(4)] for s in ('DISCOVERY','HELDOUT')},
        discovery_heldout_context_overlap=0,calibration_main_context_overlap=0,
        evidence='MEASURED_NATURAL_REFERENCE',scope='Lengths condition on original-model C1 reference. Online C3 lengths are reported separately. This balanced benchmark is not production traffic. All prior reference/performance exposure remains disclosed.')
    write(ROOT/'data/balanced400_v2_freeze_manifest.json',manifest)
    write(ROOT/'configs/quality_gate_frozen_v1.json',dict(reference='prior_C1_plus_v2extra',paired_task_macro_margin_percentage_points=2.,
        confidence=.95,request_unit='Within-task context cluster; performance unit is independent server start.',
        expanded_task_metrics='Official task map,including classification,retrieval,count,Chinese summarization and code. No quality-based sample selection.'))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('initial','census','merged','freeze'));args=parser.parse_args()
    globals()[args.action]()
