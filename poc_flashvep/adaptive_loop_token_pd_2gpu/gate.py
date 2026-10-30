"""Finite exact parity, received bytes, and prescribed recurrent work checks."""
import argparse
import json
from .common import ROOT,write

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]

def compare(reference,candidate):
    a=ROOT/'raw'/reference;b=ROOT/'raw'/candidate
    left={r['seed']:r for r in lines(a/'requests.jsonl') if not r['warmup']}
    right={r['seed']:r for r in lines(b/'requests.jsonl') if not r['warmup']}
    checks=[]
    for seed in sorted(left.keys()|right.keys()):
        x=left.get(seed);y=right.get(seed)
        fields=('input_ids','output_ids','exit_depths','finish_reason')
        exact=bool(x and y and all(x[k]==y[k] for k in fields))
        divergence=next((i for i,(p,q) in enumerate(zip(x['output_ids'],y['output_ids'])) if p!=q),None) if x and y else None
        checks.append(dict(seed=seed,exact=exact,first_divergent_token=divergence))
    def states(folder):
        result={}
        for path in folder.glob('prompt_states_*.jsonl'):
            for row in lines(path):result[row['seed']]=row
        return result
    u=states(a);v=states(b)
    state_checks=[]
    for seed in sorted(right):
        x=u.get(seed);y=v.get(seed)
        state_checks.append(dict(seed=seed,KV_exact=bool(x and y and x['planes']==y['planes']),
                                 hidden_exact=bool(x and y and x['hidden_sha256']==y['hidden_sha256'])))
    transfers=[r for path in b.glob('KV_checks*.jsonl') for r in lines(path)]
    if not transfers:
        transfers=[r for path in b.glob('kv_check*.jsonl') for r in lines(path)]
    before=json.loads((b/'before_measured.json').read_text());after=json.loads((b/'after_measured.json').read_text())
    issued=sum(after[k]['issued_loop_rows']-before[k]['issued_loop_rows'] for k in before)
    required=sum((r['input_tokens']+r['output_tokens']-1)*r['recurrent_depth'] for r in right.values())
    cfg=json.loads((b/'arc_contract.json').read_text())
    result=dict(reference=reference,candidate=candidate,output_checks=checks,prompt_state_checks=state_checks,
                output_exact=bool(checks) and all(r['exact'] for r in checks),
                prompt_state_exact=bool(state_checks) and all(r['KV_exact'] and r['hidden_exact'] for r in state_checks),
                receiver_checks=len(transfers),receiver_exact=all(r.get('passed',r.get('exact',False)) for r in transfers),
                issued_loop_rows=issued,required_loop_rows=required,work_exact=issued==required,
                limitation='Finite same-runtime migration qualification. Not original-HF equivalence or benchmark accuracy.')
    needs_bytes=cfg['arm'] not in ('DP','DP_Chunked','PD_G1') and cfg.get('cut',1)!=0
    result['passed']=result['output_exact'] and result['prompt_state_exact'] and result['work_exact'] and result['receiver_exact'] and (bool(transfers) or not needs_bytes)
    if candidate.startswith('qualification32_'):
        model=cfg['model'];arm=cfg['arm'];axis='token' if arm.startswith('Token') else 'loop'
        required=[]
        if arm not in ('DP','DP_Chunked','PD_G1'):
            for endpoint in ('zero','full'):
                required.append((f'gate8_{model}_pd_g1_v1',f'endpoint8_{model}_{axis}_static_{endpoint}_pairs_v2'))
            required.append((f'long_gate8_{model}_pd_g1_v1',f'long_gate8_{model}_{axis}_static_v1'))
        else:required.append((f'long_gate8_{model}_pd_g1_v1',f'long_gate8_{model}_pd_g1_v1'))
        result['endpoint_and_long_gates']={}
        for ref,cand in required:
            try:passed=compare(ref,cand)['passed']
            except (OSError,KeyError,ValueError):passed=False
            result['endpoint_and_long_gates'][cand]=passed
        result['passed']=result['passed'] and all(result['endpoint_and_long_gates'].values())
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');a=p.parse_args()
    value=compare(a.reference,a.candidate)
    write(ROOT/'state'/f'gate_{a.candidate}.json',value)
    print({k:v for k,v in value.items() if k not in ('output_checks','prompt_state_checks')})
    if not value['passed']:raise SystemExit(1)

if __name__=='__main__':main()
