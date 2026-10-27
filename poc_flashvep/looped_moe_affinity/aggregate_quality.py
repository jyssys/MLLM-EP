"""Aggregate full official evaluations without changing their scoring rules."""
import csv
import json
import math
from .common import ROOT,write

def interval(k,n):
    z=1.959963984540054
    p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(mid-half),100*(mid+half)

def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def summary(model,benchmark,k,n,reference,paths,ep=1):
    low,high=interval(k,n);score=100*k/n
    threshold=max(2.,(high-low)/2)
    return dict(model=model,ep=ep,benchmark=benchmark,correct=k,total=n,accuracy_pct=score,
        ci95_low_pct=low,ci95_high_pct=high,reference_pct=reference,delta_pp=score-reference,
        compatibility='WITHIN_DECLARED_TOLERANCE' if abs(score-reference)<=threshold else 'REFERENCE_MISMATCH',
        evidence='LIVE_FULL_BENCHMARK',source_paths='|'.join(str(p.relative_to(ROOT)) for p in paths))

def collect():
    native=[];ep_rows=[]
    for model,mref,aref in [('ifm_ouro',64.03,51.33),('ifm_huginn',66.81,50.47)]:
        for ep in [1,4,8]:
            if ep==1:
                mpaths=list((ROOT/'raw').glob(f'{model}_official_mmlu_bucket*/official/results.native.json'))
                apaths=list((ROOT/'raw').glob(f'{model}_official_arc_full_v*/official/results.native.json'))
            else:
                mpaths=list((ROOT/'raw').glob(f'{model}_ep{ep}_official_quality_v*/official_summary.rank0.json'))
                mpaths+=list((ROOT/'raw').glob(f'{model}_ep{ep}_official_quality_v*/quality_progress.rank0.json'))
                apaths=mpaths
            by_task={}
            for p in set(mpaths+apaths):
                payload=json.loads(p.read_text());data=payload['tasks'] if isinstance(payload,dict) else payload
                for val in (data.values() if isinstance(data,dict) else data):
                    by_task[val['task_name']]=val
            target=native if ep==1 else ep_rows
            subjects=[v for k,v in by_task.items() if k.startswith('mmlu/')]
            if len(subjects)==57:
                n=sum(v['num_items'] for v in subjects)
                k=round(sum(v['metrics']['acc']*v['num_items']/100 for v in subjects))
                assert n==14042
                target.append(summary(model,'MMLU_5shot_micro_acc',k,n,mref,mpaths,ep))
            else:
                target.append(dict(model=model,ep=ep,benchmark='MMLU_5shot_micro_acc',planned_total=14042,
                    completed_subjects=len(subjects),status='NOT_COMPLETE',evidence='NOT_MEASURED_FULL_BENCHMARK'))
            if 'arc_challenge' in by_task:
                v=by_task['arc_challenge'];n=v['num_items'];assert n==1165
                target.append(summary(model,'ARC_Challenge_0shot_acc_token',round(v['metrics']['acc_token']*n/100),n,aref,apaths,ep))
            else:
                target.append(dict(model=model,ep=ep,benchmark='ARC_Challenge_0shot_acc_token',planned_total=1165,
                    status='NOT_COMPLETE',evidence='NOT_MEASURED_FULL_BENCHMARK'))
            target.append(dict(model=model,ep=ep,benchmark='HellaSwag_0shot',total=10042,
                reference_pct=54.82 if model=='ifm_ouro' else 55.10,status='UNAVAILABLE_OFFICIAL_DISTRIBUTION',
                evidence='NOT_AVAILABLE',reason='Original GitHub distribution removed by DMCA. Official HF dataset unavailable.'))
    for ep in [1,4,8]:
        target=native if ep==1 else ep_rows
        for task,n,ref in [('lambada_standard',5153,31.83),('lambada_openai',5153,39.98),('xwinograd_en',2325,68.82)]:
            pattern=f'foil1_official_{task}_full_v*/foil_results.rank0.json' if ep==1 else f'foil1_ep{ep}_official_quality_v*/foil_results.rank0.json'
            paths=list((ROOT/'raw').glob(pattern))
            found=False
            for p in paths:
                data=json.loads(p.read_text())
                if task not in data['results']:continue
                value=data['results'][task]['acc,none']
                target.append(summary('foil1',task+'_0shot_acc',round(value*n),n,ref,[p],ep));found=True;break
            if not found:target.append(dict(model='foil1',ep=ep,benchmark=task+'_0shot_acc',planned_total=n,
                status='NOT_COMPLETE',evidence='NOT_MEASURED_FULL_BENCHMARK'))
    csv_write(ROOT/'accuracy_native.csv',native)
    csv_write(ROOT/'accuracy_ep.csv',ep_rows)
    write(ROOT/'quality/aggregate_snapshot.json',dict(native=native,ep=ep_rows,
        note='MMLU uses item-weighted accuracy. ARC uses official acc_token. Compatibility is not equivalence.'))
    return native,ep_rows

if __name__=='__main__':
    n,e=collect()
    for row in n+e:
        if 'accuracy_pct' in row:print(row['model'],row['ep'],row['benchmark'],row['correct'],row['total'],round(row['accuracy_pct'],4),row['compatibility'])
