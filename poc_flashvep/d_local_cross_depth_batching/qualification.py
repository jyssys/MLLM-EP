"""Exact outputs and all valid prompt state. Never treat missing evidence as pass."""
import json
from .common import ROOT, write


def measured(folder):
    path = folder / 'requests.jsonl'
    return {v['sample_id']:v for line in path.read_text().splitlines()
            if (v:=json.loads(line)).get('warmup') is False}


def states(folder):
    values = {}
    for path in folder.glob('prompt_states_*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['seed'],row['prompt_tokens'])
            description = dict(hidden_sha256=row['hidden_sha256'], planes=row['planes'])
            if key in values:
                # Keep contradictory snapshots invalid. Never select the
                # convenient snapshot or stop qualification of other methods.
                if values[key] != description:
                    values[key] = None
            else:
                values[key] = description
    return values


def compare(reference, candidate, count):
    ref, target = ROOT/'raw'/reference, ROOT/'raw'/candidate
    a, b = measured(ref), measured(target)
    sa, sb = states(ref), states(target)
    result = []
    for index in range(count):
        left, right = a.get(index), b.get(index)
        exact = bool(left and right and all(left[k]==right[k]
            for k in ('input_ids','output_ids','exit_depths','finish_reason')))
        key = (left['seed'],left['input_tokens']) if left else None
        state_exact = key in sa and key in sb and sa[key] is not None and sb[key] is not None and sa[key] == sb[key]
        result.append(dict(sample_id=index, output_exact=exact, valid_KV_and_hidden_exact=state_exact,
            reference_state_present=key in sa, candidate_state_present=key in sb))
    passed = len(a)==len(b)==count and all(r['output_exact'] and r['valid_KV_and_hidden_exact'] for r in result)
    value = dict(reference=reference,candidate=candidate,count=count,passed=passed,requests=result,
        conflicting_state_keys=dict(reference=[list(k) for k,v in sa.items() if v is None],
            candidate=[list(k) for k,v in sb.items() if v is None]),
        scope='All generated tokens and every hashed valid prompt KV plane plus prompt-last hidden state. Internal runtime parity, not original HF equivalence.')
    write(target/'local_qualification.json',value)
    return value


def compare_outputs(reference,candidate,count):
    a,b=measured(ROOT/'raw'/reference),measured(ROOT/'raw'/candidate)
    rows=[]
    for index in range(count):
        left,right=a.get(index),b.get(index)
        rows.append(dict(sample_id=index,exact=bool(left and right and all(left[k]==right[k]
            for k in ('input_ids','output_ids','exit_depths','finish_reason')))))
    result=dict(reference=reference,candidate=candidate,count=count,
        passed=len(b)==count and all(r['exact'] for r in rows),requests=rows,
        scope='Clean serving output-token and fixed-depth comparison. No KV/hidden hashes are collected in this run.')
    write(ROOT/'raw'/candidate/'serving_output_parity.json',result)
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(compare(sys.argv[1],sys.argv[2],int(sys.argv[3]))))
