"""Freeze the measured natural-length population without serving-result selection."""
import collections
import hashlib
import json
import time
import numpy as np
from .common import ROOT,REPO,REVISION,initialize,write,digest


def output_bucket(length):
    for i,(left,right) in enumerate(((1,8),(9,24),(25,49),(50,99))):
        if left<=length<=right:return i
    return None


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
        min=float(values.min()),max=float(values.max()),
        **{'p'+str(p):float(np.percentile(values,p)) for p in (10,25,50,75,90,95,99)})


def main():
    initialize();contract=json.loads((ROOT/'configs/campaign_contract_v1.json').read_text())
    finished=json.loads((ROOT/'state/candidate_natural_reference_complete.json').read_text());assert finished['passed']
    candidate_path=ROOT/'data/candidate_pool_original.json';pool=json.loads(candidate_path.read_text())
    source=ROOT/'raw'/('CandidateNaturalReference_'+finished['version'])
    reference=[json.loads(line) for path in sorted(source.glob('GPU*/natural_reference.jsonl')) for line in path.read_text().splitlines()]
    by_id={row['prompt_id']:row for row in reference};assert len(by_id)==len(reference)==len(pool)
    write(ROOT/'data/natural_reference_all_candidates.json',dict(rows=reference,source_version=finished['version']))
    seed=contract['selection_seed'];initial=set(contract['candidate_tasks_initial']);expansion=contract['candidate_expansion_order']
    def rank(row):
        task_rank=0 if row['task'] in initial else 1+expansion.index(row['task'])
        return task_rank,hashlib.sha256((str(seed)+row['prompt_id']).encode()).hexdigest()
    cells=collections.defaultdict(list)
    for row in pool:
        value=by_id[row['prompt_id']]
        assert value['input_ids_SHA256']==row['input_ids_SHA256']
        bucket=output_bucket(value['output_tokens'])
        if bucket is not None:cells[row['input_bucket'],bucket].append(row)
    for rows in cells.values():rows.sort(key=rank)
    available=[[len(cells[i,j]) for j in range(4)] for i in range(4)]
    selected=[];selected_ids=set();deficits=[];moves=[]
    for i in range(4):
        for j in range(4):
            chosen=cells[i,j][:25]
            selected.extend(chosen);selected_ids.update(r['prompt_id'] for r in chosen)
            if len(chosen)<25:deficits.append((i,j,25-len(chosen)))
    # The distance rule is frozen before any online measurement.
    for i,j,remaining in deficits:
        candidates=[r for rows in cells.values() for r in rows if r['prompt_id'] not in selected_ids]
        candidates.sort(key=lambda r:(abs(r['input_bucket']-i),abs(output_bucket(by_id[r['prompt_id']]['output_tokens'])-j),rank(r)))
        take=candidates[:remaining]
        selected.extend(take);selected_ids.update(r['prompt_id'] for r in take)
        moves.append(dict(target_input=i,target_output=j,shortage=remaining,
            allocations=[dict(prompt_id=r['prompt_id'],input_bucket=r['input_bucket'],output_bucket=output_bucket(by_id[r['prompt_id']]['output_tokens'])) for r in take]))
    assert len(selected)==min(400,sum(sum(row) for row in available))
    selected.sort(key=lambda r:hashlib.sha256(('arrival-order-'+str(seed)+r['prompt_id']).encode()).hexdigest())
    for index,row in enumerate(selected):
        ref=by_id[row['prompt_id']]
        row.update(sample_id=index,natural_output_tokens=ref['output_tokens'],
            natural_output_ids_SHA256=hashlib.sha256(json.dumps(ref['output_ids']).encode()).hexdigest(),
            natural_output_bucket=output_bucket(ref['output_tokens']),
            evaluation_split='HELDOUT' if int(hashlib.sha256(('holdout-'+str(seed)+row['context_SHA256']).encode()).hexdigest(),16)%2 else 'DISCOVERY')
    target=ROOT/'data/longbench_balanced400_v1.json'
    assert not target.exists(),'The frozen dataset cannot be overwritten'
    write(target,selected)
    selected_reference=[by_id[r['prompt_id']] for r in selected]
    write(ROOT/'data/longbench_balanced400_v1_natural_reference.json',dict(rows=selected_reference))
    counts=[[sum(r['input_bucket']==i and r['natural_output_bucket']==j for r in selected) for j in range(4)] for i in range(4)]
    tasks=dict(collections.Counter(r['task'] for r in selected))
    # Context-group assignment prevents context leakage between discovery and held-out.
    for split in ('DISCOVERY','HELDOUT'):
        write(ROOT/'data'/('balanced400_'+split.lower()+'.json'),[r for r in selected if r['evaluation_split']==split])
    remainder=[r for r in pool if r['context_SHA256'] not in {s['context_SHA256'] for s in selected}]
    remainder.sort(key=lambda r:hashlib.sha256(('calibration-'+str(seed)+r['prompt_id']).encode()).hexdigest())
    assert len(remainder)>=32
    q8=[]
    for bucket in range(4):q8.extend([r for r in remainder if r['input_bucket']==bucket][:2])
    q32=q8+[r for r in remainder if r['prompt_id'] not in {x['prompt_id'] for x in q8}][:24]
    q1=[min(remainder,key=lambda r:r['input_tokens'])]
    for name,rows in (('qualification1',q1),('qualification8',q8),('qualification32',q32)):
        write(ROOT/'data'/(name+'.json'),rows)
    for bucket in range(4):
        rows=[r for r in selected if r['input_bucket']==bucket and r['evaluation_split']=='DISCOVERY']
        write(ROOT/'data'/f'mechanism_input{bucket}_discovery.json',rows)
        held=[r for r in selected if r['input_bucket']==bucket and r['evaluation_split']=='HELDOUT']
        write(ROOT/'data'/f'mechanism_input{bucket}_heldout.json',held)
    summary=dict(time=time.time(),dataset_name='LongBench-Balanced400-v1',dataset_SHA256=digest(target),
        source_revision=REVISION,selection_seed=seed,n=len(selected),available_cells=available,selected_cells=counts,
        exact_25_per_cell=all(n==25 for row in counts for n in row),shortage_reallocation=moves,
        input=stats([r['input_tokens'] for r in selected]),output=stats([r['natural_output_tokens'] for r in selected]),
        pool_input=stats([r['input_tokens'] for r in pool]),pool_output=stats([r['output_tokens'] for r in reference]),
        input_bucket_counts=[sum(row) for row in counts],output_bucket_counts=[sum(counts[i][j] for i in range(4)) for j in range(4)],
        tasks=tasks,languages=dict(collections.Counter(r['language'] for r in selected)),
        candidate_n=len(pool),eligible_n=sum(sum(row) for row in available),
        outside_output_bins=sum(output_bucket(r['output_tokens']) is None for r in reference),
        pool_cap_hits=sum(r['cap_hit'] for r in reference),selected_cap_hits=sum(r['cap_hit'] for r in selected_reference),
        EOS=sum(r['natural_EOS'] for r in selected_reference),total_input_tokens=sum(r['input_tokens'] for r in selected),
        total_output_tokens=sum(r['natural_output_tokens'] for r in selected),
        unique_requests=len({r['prompt_id'] for r in selected}),unique_contexts=len({r['context_SHA256'] for r in selected}),
        truncations=0,split_counts=dict(collections.Counter(r['evaluation_split'] for r in selected)),
        discovery_heldout_context_overlap=0,calibration_main_context_overlap=0,
        evidence='MEASURED_NATURAL_OUTPUT_REFERENCE',
        scope='The balanced sample is not production traffic. Natural-length conditioning changes the population. '
            'Generation uses original prompts and a common cap2048,not official task-specific output caps. '
            'Task quality never determines sample selection. C1 length references are not online serving results.')
    write(ROOT/'data/balanced400_freeze_manifest.json',summary)
    lines=['# LongBench-Balanced400-v1 데이터셋 계약','',
        '증거 수준: `MEASURED_NATURAL_OUTPUT_REFERENCE`. Online Serving 성능은 아직 `NOT_RUN`이다.','',
        f'원문 요청 {len(selected)}개를 동결했다. Input 구간별 요청 수는 {summary["input_bucket_counts"]}이다.',
        '16개 Cell에 각각 25개를 배정하는 목표는 달성하지 못했다. 부족분은 사전에 정한 인접 Cell 규칙으로 재배분했다.','',
        '| Input Token 구간 | Output1–8 후보/선정 | Output9–24 후보/선정 | Output25–49 후보/선정 | Output50–99 후보/선정 |',
        '|---|---:|---:|---:|---:|']
    for i,label in enumerate(('3000–5999','6000–9999','10000–19999','20000–36000')):
        lines.append('| '+label+' | '+' | '.join(f'{available[i][j]}/{counts[i][j]}' for j in range(4))+' |')
    lines.extend(['',f'Dataset SHA256: `{summary["dataset_SHA256"]}`.',
        f'Dataset Revision: `{REVISION}`. 선정 Seed: `{seed}`.',
        '모든 Input 길이는 Qwen3-8B Chat Template 적용 후 Token 수다. Thinking은 비활성화했다.',
        '원문 변경, 새 Truncation, Padding, 요청 복제는 사용하지 않았다.',
        'Natural Output은 BI0 Single-GPU Concurrency1에서 측정했다. Greedy, EOS 허용, 공통 Cap2048을 사용했다.',
        '이 Cap은 LongBench의 Task별 기본 Generation Cap과 다르다.','',
        f'후보 요청 수는 {len(pool)}개다. 후보 전체의 Cap Hit는 {summary["pool_cap_hits"]}개다.',
        f'선정 요청의 Cap Hit는 {summary["selected_cap_hits"]}개다. EOS 종료는 {summary["EOS"]}/{len(selected)}개다.',
        f'고유 Request는 {summary["unique_requests"]}개다. 고유 Context는 {summary["unique_contexts"]}개다.',
        '같은 Context의 Request는 동일 Discovery/Held-out 집합에 배정했다.',
        'QA Score와 Online 성능은 선정 기준에 사용하지 않았다. 긴 Output을 제외한 선택 편향은 남는다.',
        '균형 데이터셋을 일반 Production 분포로 해석하지 않는다.','',
        '## 실제 길이 통계','', '| 통계 | Input Tokens | Natural Output Tokens |','|---|---:|---:|'])
    for key in ('mean','median','min','max','p10','p25','p75','p90','p95','p99'):
        lines.append(f'| {key} | {summary["input"][key]:.3f} | {summary["output"][key]:.3f} |')
    lines.extend(['','## 재현 자료','',
        '전체 Manifest와 Task별 통계는 캠페인 Namespace의 `data/`에 저장했다.',
        '후보와 선정 집합의 분포 차이, Cell 부족분, Context 중복, Split 수는 `balanced400_freeze_manifest.json`에 기록했다.',
        'Task 품질 검사는 별도 결과로 저장한다. 품질 결과가 낮아도 동결 요청을 교체하지 않는다.'])
    # Report text uses short,scope-qualified sentences. Frozen raw evidence remains unchanged.
    from subprocess import run
    patch='*** Begin Patch\n*** Add File: '+str(REPO/'poc_flashvep/reports/dense_ar_longbench400_dataset_contract.md')+'\n'+'\n'.join('+'+line for line in lines)+'\n*** End Patch\n'
    # The report is emitted through the repository patch mechanism by the calling agent.
    print(json.dumps(dict(summary=summary,report_patch=patch),ensure_ascii=False))


if __name__=='__main__':main()
