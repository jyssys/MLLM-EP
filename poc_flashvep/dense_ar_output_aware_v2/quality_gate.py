"""Compare C3 quality with an independent original compiled C1 reference."""
import collections
import json
import time
import numpy as np
from .common import ROOT,write,digest,dispatch_allowed,resources
from .quality import scores

def paired_quality(actual,expected,data,resamples):
    aa={r['prompt_id']:r for r in scores(actual,data)};bb={r['prompt_id']:r for r in scores(expected,data)}
    raw={r['prompt_id']:r for r in actual};gold={r['prompt_id']:r for r in expected};rows=[]
    for prompt in data:
        key=prompt['prompt_id'];a=raw[key];b=gold[key]
        rows.append(dict(prompt_id=key,task=prompt['task'],context_SHA256=prompt['context_SHA256'],
            actual_score=aa[key]['score'],reference_score=bb[key]['score'],delta_percentage_points=aa[key]['score']-bb[key]['score'],
            greedy_exact=a['output_ids']==b['output_ids'],actual_output_tokens=a['output_tokens'],reference_output_tokens=b['output_tokens'],
            cap_hit=a.get('cap_hit',False),empty_text=aa[key]['empty_text']))
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:groups[r['task']][r['context_SHA256']].append(r['delta_percentage_points'])
    point=float(np.mean([np.mean([r['delta_percentage_points'] for r in rows if r['task']==task]) for task in groups]))
    rng=np.random.default_rng(2026112201);boot=[]
    for _ in range(resamples):
        values=[]
        for clusters in groups.values():
            v=list(clusters.values());chosen=rng.integers(0,len(v),len(v))
            values.append(float(np.mean([x for index in chosen for x in v[index]])))
        boot.append(float(np.mean(values)))
    return dict(rows=rows,task_macro_delta_percentage_points=point,CI95_percentage_points=np.percentile(boot,[2.5,97.5]).tolist(),
        exact_greedy_requests=sum(r['greedy_exact'] for r in rows),cap_hits=sum(r['cap_hit'] for r in rows),empty_text=sum(r['empty_text'] for r in rows))

def compare(name,resamples=4000):
    contract=ROOT/'configs/c3_quality_contract_v2.json';assert contract.exists()
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text());assert not cfg['ignore_eos']
    marker=ROOT/'state/official_compiled_c1_reference_complete_v2.json'
    while not marker.exists():
        assert dispatch_allowed();resources().restore(cfg['gpus']);time.sleep(5)
    reference_state=json.loads(marker.read_text());assert reference_state['passed']
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    actual=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    refs=[json.loads(s) for s in (ROOT/'raw'/reference_state['name']/'requests.jsonl').read_text().splitlines()]
    byid={r['prompt_id']:r for r in refs};expected=[byid[r['prompt_id']] for r in data]
    assert {r['prompt_id'] for r in actual}=={r['prompt_id'] for r in data} and len(actual)==len(data)
    result=paired_quality(actual,expected,data,resamples)
    secondary={r['prompt_id']:r for r in json.loads((ROOT/'data/natural_reference_all_candidates.json').read_text())['rows']}
    secondary_result=paired_quality(actual,[secondary[r['prompt_id']] for r in data],data,resamples)
    margin=json.loads(contract.read_text())['paired_task_macro_margin_percentage_points']
    result.update(name=name,n=len(data),passed=bool(result['CI95_percentage_points'][0]>=-margin),margin_percentage_points=margin,
        resamples=resamples,dataset_SHA256=cfg['dataset_SHA256'],contract_SHA256=digest(contract),
        reference=reference_state['name'],secondary_Mode0_reference={k:v for k,v in secondary_result.items() if k!='rows'},
        evidence='TASK_QUALITY_NONINFERIORITY',
        scope='Original compiled C1 reference is primary for C3. Original Mode0 C1 quality remains secondary. '
            'Dataset selection and its frozen lengths remain Mode0 C1. Context-cluster quality CI is not a server-start performance CI.')
    write(ROOT/'analysis'/(name+'_quality_gate.json'),result)
    return {k:v for k,v in result.items() if k!='rows'}
