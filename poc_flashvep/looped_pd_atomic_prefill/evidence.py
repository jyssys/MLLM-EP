"""Evaluate actual output, transfer, and atomic-span evidence without GPU work."""
from .common import *


def records(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def rows(name):return [r for r in records(ROOT/'raw'/name/'requests.jsonl') if not r.get('warmup')]


def status(name):
    folder=ROOT/'raw'/name;p=folder/'status.json';result=folder/'dispatcher_result.json'
    if result.exists() and json.loads(result.read_text())['returncode']!=0:return 'FAILED'
    if p.exists():return json.loads(p.read_text())['status']
    return 'NOT_RUN'


def atomic(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    window=json.loads((folder/'window.json').read_text())
    spans=[]
    if cfg['model']=='qwen8':
        for p in folder.glob('native_prefill_*.json'):
            spans.extend(r for r in json.loads(p.read_text())['prompt_spans'] if r['request_id'].startswith('q') and window['start']<=r['time']<=window['end'])
        keys={r['request_id'].split('-',1)[0] for r in spans}
    else:
        for p in folder.glob('atomic_spans_*.json'):
            data=json.loads(p.read_text())
            if data['role']!='prefill':continue
            spans.extend(r for r in data['prompt_spans'] if window['start']<=r['time']<=window['end'])
        keys={r['seed'] for r in spans}
    exact=len(keys)==cfg['count'] and len(spans)==cfg['count'] and all(r['start']==0 and r['count']==r['prompt_tokens']==cfg['length'] for r in spans)
    if cfg['model']!='qwen8':exact &= all(r['loop_count']==cfg['depth'] for r in spans)
    return dict(run=name,passed=bool(exact),recorded_prompt_spans=len(spans),distinct_requests=len(keys),expected_requests=cfg['count'],
        scope='Actual scheduled token spans. Atomic does not mean one kernel or one transfer fragment.')


def compare(a,b):
    x={r['sample_id']:r for r in rows(a)};y={r['sample_id']:r for r in rows(b)}
    failures=[]
    matched_inputs=sum(i in y and all(r[k]==y[i][k] for k in ('input_ids','seed')) for i,r in x.items())
    for i,r in x.items():
        other=y.get(i)
        if other is None or any(r[k]!=other[k] for k in ('input_ids','output_ids','seed')):
            failures.append(dict(sample_id=i,reference=r.get('output_ids'),candidate=other.get('output_ids') if other else None))
    return dict(reference=a,candidate=b,compared=len(x),candidate_rows=len(y),passed=bool(x) and len(x)==len(y) and not failures,
        matched_input_seed_rows=matched_inputs,comparison_applicable=bool(x) and len(x)==len(y)==matched_inputs,
        failures=failures,scope='Internal exact output sequence parity. Not original-HF fidelity or downstream accuracy.')


def full_work(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    a=json.loads((folder/'before_measured.json').read_text());b=json.loads((folder/'after_measured.json').read_text())
    observed=sum(v['issued_loop_rows']-a[k]['issued_loop_rows'] for k,v in b.items())
    expected=sum((r['input_tokens']+r['output_tokens']-1)*cfg['depth'] for r in rows(name))
    return dict(run=name,expected=expected,observed=observed,passed=observed==expected,
        scope='Active recurrent rows, not padded physical GEMM work.')


def transfer(name):
    folder=ROOT/'raw'/name
    metrics=json.loads((folder/'closed_stats.json').read_text())['metrics']
    tids={r['transfer_id'] for r in metrics if r['request_id'].startswith('r0_')}
    checks=[v for p in folder.glob('KV_checks-*.jsonl') for v in records(p) if v['tid'] in tids]
    return dict(run=name,expected_requests=len(tids),valid_KV_hidden_byte_checks=len(checks),
        passed=bool(tids) and len({v['tid'] for v in checks})==len(tids) and all(v['passed'] for v in checks))


def qualify(model,reference,candidate):
    result=dict(time=time.time(),model=model,comparison=compare(reference,candidate),atomic=atomic(candidate))
    if model!='qwen8':result.update(transfer=transfer(candidate),work=full_work(candidate))
    result['passed']=all(v['passed'] for k,v in result.items() if isinstance(v,dict))
    return result
