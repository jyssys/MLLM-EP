"""Apply the frozen paired task-quality gate without changing selected prompts."""
import collections
import json
import numpy as np
from .common import ROOT,write,digest
from .quality import scores


def compare(name,resamples=4000):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    assert not cfg.get('ignore_eos'),'Task quality requires the natural generation contract'
    assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    actual=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    reference=json.loads((ROOT/'data/natural_reference_all_candidates.json').read_text())['rows']
    by_id={row['prompt_id']:row for row in reference}
    expected=[by_id[row['prompt_id']] for row in data]
    assert len(actual)==len(data) and {r['prompt_id'] for r in actual}==set(by_id)&{r['prompt_id'] for r in data}
    aa={r['prompt_id']:r for r in scores(actual,data)}
    bb={r['prompt_id']:r for r in scores(expected,data)}
    raw={r['prompt_id']:r for r in actual};rows=[]
    for prompt in data:
        key=prompt['prompt_id'];gold=by_id[key];observed=raw[key]
        rows.append(dict(prompt_id=key,task=prompt['task'],context_SHA256=prompt['context_SHA256'],
            actual_score=aa[key]['score'],reference_score=bb[key]['score'],
            delta_percentage_points=aa[key]['score']-bb[key]['score'],
            greedy_exact=observed['output_ids']==gold['output_ids'],
            actual_output_tokens=observed['output_tokens'],reference_output_tokens=gold['output_tokens'],
            cap_hit=observed.get('cap_hit',False),empty_text=aa[key]['empty_text']))
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in rows:groups[row['task']][row['context_SHA256']].append(row['delta_percentage_points'])
    observed_delta=float(np.mean([np.mean([r['delta_percentage_points'] for r in rows if r['task']==task]) for task in groups]))
    rng=np.random.default_rng(2026112181);boot=[]
    for _ in range(resamples):
        tasks=[]
        for task,clusters in groups.items():
            values=list(clusters.values());chosen=rng.integers(0,len(values),len(values))
            tasks.append(float(np.mean([v for index in chosen for v in values[index]])))
        boot.append(float(np.mean(tasks)))
    ci=np.percentile(boot,[2.5,97.5]).tolist()
    contract=json.loads((ROOT/'configs/quality_gate_frozen_v1.json').read_text())
    margin=contract['paired_task_macro_margin_percentage_points']
    result=dict(name=name,n=len(rows),task_count=len(groups),rows=rows,
        task_macro_delta_percentage_points=observed_delta,CI95_percentage_points=ci,
        exact_greedy_requests=sum(r['greedy_exact'] for r in rows),cap_hits=sum(r['cap_hit'] for r in rows),
        empty_text=sum(r['empty_text'] for r in rows),passed=ci[0]>=-margin,
        margin_percentage_points=margin,resamples=resamples,bootstrap_unit='Within-task context cluster',
        dataset_SHA256=cfg['dataset_SHA256'],contract_SHA256=digest(ROOT/'configs/quality_gate_frozen_v1.json'),
        evidence='TASK_QUALITY_NONINFERIORITY',
        scope='This gate compares actual natural outputs with the frozen BI0 single-GPU reference. '
            'It does not prove KV correctness. Its request-cluster CI is not a server-start performance CI.')
    write(ROOT/'analysis'/(name+'_quality_gate.json'),result)
    return {k:v for k,v in result.items() if k!='rows'}


if __name__=='__main__':
    import sys
    print(json.dumps(compare(sys.argv[1]),indent=2))
