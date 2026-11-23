"""Compare output identity and task quality with the matching Native start."""
import collections
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .critical_analysis import records,csv_save


def paired_rows(actual,reference,resamples=4000):
    by_id={row['prompt_id']:row for row in reference};groups=collections.defaultdict(lambda:collections.defaultdict(list))
    rows=[]
    assert {row['prompt_id'] for row in actual}==set(by_id)
    for row in actual:
        other=by_id[row['prompt_id']]
        assert row['task']==other['task'] and row['context_SHA256']==other['context_SHA256']
        delta=row['actual_score']-other['actual_score']
        groups[row['task']][row['context_SHA256']].append(delta)
        rows.append(dict(prompt_id=row['prompt_id'],task=row['task'],context_SHA256=row['context_SHA256'],
            actual_score=row['actual_score'],native_score=other['actual_score'],delta_pp=delta))
    point=float(np.mean([np.mean([x for values in cluster.values() for x in values]) for cluster in groups.values()]))
    rng=np.random.default_rng(2026112201);boot=[]
    for _ in range(resamples):
        values=[]
        for clusters in groups.values():
            v=list(clusters.values());chosen=rng.integers(0,len(v),len(v))
            values.append(float(np.mean([x for index in chosen for x in v[index]])))
        boot.append(float(np.mean(values)))
    return dict(task_macro_delta_pp=point,context_cluster_CI95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
        lower_score_requests=sum(row['delta_pp']<0 for row in rows),higher_score_requests=sum(row['delta_pp']>0 for row in rows),
        equal_score_requests=sum(row['delta_pp']==0 for row in rows),rows=rows)


def analyze():
    families=('c3_credit_confirmation','p_geometry_confirmation','coordinator_hygiene_confirmation',
        'coordinator_hygiene_stable_stock_confirmation','coordinator_hygiene_fixed_ports_confirmation',
        'coordinator_hygiene_warm_regime_confirmation','p_geometry_D_cap4_confirmation','output192_D_cap_intervention')
    starts=[];requests=[];sources={}
    for family in families:
        source=ROOT/'analysis'/(family+'_analysis.json')
        if not source.exists():continue
        sources[family]=digest(source);analysis=json.loads(source.read_text())
        index={(row['mode'],row['start'],row['label']):row for row in analysis['starts']}
        for row in analysis['starts']:
            if row['mode']!='Natural' or row['label']=='NativeStrong':continue
            native=index.get(('Natural',row['start'],'NativeStrong'))
            if native is None:continue
            names=[row['name'],native['name']]
            paths=[ROOT/'analysis'/(name+'_quality_gate.json') for name in names]
            if not all(path.exists() for path in paths):continue
            quality=[json.loads(path.read_text()) for path in paths]
            outputs=[{r['prompt_id']:r for r in records(ROOT/'raw'/name/'requests.jsonl')} for name in names]
            assert set(outputs[0])==set(outputs[1])
            result=paired_rows(quality[0]['rows'],quality[1]['rows'])
            for item in result.pop('rows'):requests.append(dict(family=family,start=row['start'],label=row['label'],**item))
            starts.append(dict(family=family,start=row['start'],label=row['label'],name=row['name'],reference=native['name'],
                requests=len(outputs[0]),exact_output_requests=sum(outputs[0][key]['output_ids']==outputs[1][key]['output_ids'] for key in outputs[0]),
                same_output_length_requests=sum(outputs[0][key]['output_tokens']==outputs[1][key]['output_tokens'] for key in outputs[0]),
                actual_output_tokens=sum(value['output_tokens'] for value in outputs[0].values()),
                native_output_tokens=sum(value['output_tokens'] for value in outputs[1].values()),
                frozen_C1_quality_passed=quality[0]['passed'],frozen_C1_quality_margin_pp=quality[0]['margin_percentage_points'],
                **result))
    value=dict(time=time.time(),sources_SHA256=sources,starts=starts,
        evidence='TRACE_ANALYSIS',scope='Same-start Natural EOS outputs and frozen task scores. NativeStrong is a comparison,not ground truth. '
            'A different token sequence does not prove an incorrect answer. A lower task score is a measured per-request regression. '
            'Context-cluster95% CIs describe paired task quality,not independent server-start uncertainty. '
            'The unchanged compiled C1 reference and2pp gate remain primary. This Native-relative audit does not replace them. '
            'No new GPU job,request selection,output correction or quality threshold is introduced.')
    write(ROOT/'analysis/native_relative_accuracy_analysis.json',value)
    csv_save(ROOT/'figures/native_relative_accuracy_per_start.csv',starts)
    csv_save(ROOT/'figures/native_relative_accuracy_per_request.csv',requests)
    print('NATIVE_RELATIVE_ACCURACY',len(starts),flush=True)
    return value


if __name__=='__main__':analyze()
