"""Check submitted D quanta. Configuration intent alone is insufficient."""
from collections import Counter
from .common import *


def validate_spans(cfg,spans):
    loops=cfg.get('D_prefill_loop_cap') or cfg['depth'];tokens=cfg.get('D_prefill_token_cap') or cfg['length']
    bad=[r for r in spans if r['loop_count']>loops or r['count']>tokens]
    covered={r['seed'] for r in spans if r['loop_count']>0}
    hist=Counter((r['phase'],r['count'],r['loop_count']) for r in spans)
    return dict(loop_limit=loops,token_limit=tokens,request_coverage=len(covered),
        expected_requests=cfg['count'],violations=len(bad),passed=not bad and len(covered)==cfg['count'],
        histogram=[dict(phase=p,tokens=t,loops=l,spans=n) for (p,t,l),n in sorted(hist.items())],
        scope='Actual measured D prompt submissions. This is not a hardware kernel-preemption or Decode-safety test.')


def audit(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    begin=json.loads((folder/'measurement_begin.json').read_text())['time'];spans=[]
    for p in folder.glob('atomic_spans_*.json'):
        value=json.loads(p.read_text())
        if value['role']=='decode':spans += [r for r in value['prompt_spans'] if r['time']>=begin and 0<=r['seed']-cfg['seed']<cfg['count']]
    result=dict(name=name,**validate_spans(cfg,spans));write(folder/'actual_quantum_contract.json',result)
    return result
