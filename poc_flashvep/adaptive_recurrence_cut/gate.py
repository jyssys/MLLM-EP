"""Compare actual output sequences and full-prompt KV/state hashes."""
import argparse
import json
from .common import ROOT,write


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare(reference,candidate):
    left=ROOT/'raw'/reference;right=ROOT/'raw'/candidate
    cfg=json.loads((right/'arc_contract.json').read_text())
    a={r['seed']:r for r in read(left/'requests.jsonl') if not r['warmup']}
    a={seed:r for seed,r in a.items() if r['sample_id']<cfg['count']}
    b={r['seed']:r for r in read(right/'requests.jsonl') if not r['warmup']}
    rows=[]
    for seed in sorted(a.keys()|b.keys()):
        x=a.get(seed);y=b.get(seed)
        same=bool(x and y and all(x[k]==y[k] for k in ('input_ids','output_ids','exit_depths','finish_reason')))
        divergence=next((i for i,(p,q) in enumerate(zip(x['output_ids'],y['output_ids'])) if p!=q),None) if x and y else None
        rows.append(dict(seed=seed,exact=same,first_divergent_output_index=divergence))
    def states(directory):
        out={}
        for path in directory.glob('prompt_states_*.jsonl'):
            for r in read(path):out[r['seed']]=r
        return out
    u,v=states(left),states(right)
    u={seed:r for seed,r in u.items() if seed in a}
    state_rows=[]
    for seed in sorted(u.keys()|v.keys()):
        x=u.get(seed);y=v.get(seed)
        state_rows.append(dict(seed=seed,hidden_exact=bool(x and y and x['hidden_sha256']==y['hidden_sha256']),
                               KV_exact=bool(x and y and x['planes']==y['planes'])))
    transfers=[r for p in right.glob('KV_checks*.jsonl') for r in read(p)]
    before=json.loads((right/'before_measured.json').read_text())
    after=json.loads((right/'after_measured.json').read_text())
    issued=sum(after[k]['issued_loop_rows']-before[k]['issued_loop_rows'] for k in before)
    required=sum((r['input_tokens']+r['output_tokens']-1)*r['recurrent_depth'] for r in b.values())
    result=dict(reference=reference,candidate=candidate,output_checks=rows,prompt_state_checks=state_rows,
                receiver_byte_checks=len(transfers),receiver_bytes_exact=all(r['passed'] for r in transfers),
                issued_recurrent_rows=issued,required_recurrent_rows=required,full_depth_work_passed=issued==required,
                token_passed=bool(rows) and all(r['exact'] for r in rows),
                state_passed=bool(state_rows) and all(r['hidden_exact'] and r['KV_exact'] for r in state_rows),
                limitation='Finite new-migration gate. Not downstream accuracy or universal original-HF equivalence.')
    result['passed']=result['token_passed'] and result['state_passed'] and result['receiver_bytes_exact'] and result['full_depth_work_passed']
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--output',required=True)
    a=p.parse_args();result=compare(a.reference,a.candidate);write(ROOT/'state'/a.output,result)
    print({k:v for k,v in result.items() if k not in ('output_checks','prompt_state_checks')})
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
