"""Compare queried valid layer states, not unused KV allocation cells."""
import json,re
from .common import *
from .analyze import records,compare


def events(name,role):
    return [r for p in (ROOT/'raw'/name).glob(f'extension_{role}_*.jsonl') for r in records(p)]


def states(name,role):
    result={}
    for row in events(name,role):
        if row['event']!='VALID_QUERY_STATE' or row['query_start']!=0:continue
        if row['request_id'].startswith('_warmup_'):continue
        external=row.get('external_request_id')
        if external is None:
            # Frozen v2 controls precede explicit ID-map logging. Their Native
            # InputProcessor appends one 8-hex suffix to the controlled ID.
            match=re.fullmatch(r'((?:q\d+)|(?:warm_\d+))-[0-9a-f]{8}',row['request_id'])
            assert match,('Unrecognized Native identity',row['request_id'])
            external=match.group(1)
        if not re.fullmatch(r'q\d+',external):continue
        assert external not in result,'Do not merge duplicate Prefill execution'
        result[external]=row
    return result


def state_compare(reference,candidate,role,cut=None,allow_reference_subset=False):
    a=states(reference,'prefill');b=states(candidate,role);failures=[]
    for rid,y in b.items():
        x=a.get(rid)
        if x is None:failures.append(dict(request=rid,reason='MISSING_REFERENCE'));continue
        expected=x['KV'][:cut] if cut else x['KV']
        if y['query_tokens']!=x['query_tokens'] or y['KV']!=expected:
            failures.append(dict(request=rid,reason='VALID_KV_HASH_MISMATCH',reference=expected,candidate=y['KV']))
        if cut is None and y['final_hidden_SHA256']!=x['final_hidden_SHA256']:
            failures.append(dict(request=rid,reason='FINAL_HIDDEN_HASH_MISMATCH'))
    return dict(reference=reference,candidate=candidate,role=role,compared=len(b),
        layers_per_request=cut or 36,passed=bool(a) and bool(b) and (allow_reference_subset or len(a)==len(b)) and not failures,failures=failures,
        scope='Bitwise hashes of every queried valid KV cell and, for complete Prefill, final Hidden. Not arbitrary unqueried cache storage.')


def main():
    plan=json.loads((ROOT/'state/layer_recovery_plan.json').read_text());a,b,c=plan['jobs'];checks=[]
    for n in (b,c):
        receipt=ROOT/'raw'/n/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        checks.append(compare(a,n));checks.append(state_compare(a,n,'prefill',32 if n==c else None))
        if n==c:checks.append(state_compare(a,n,'decode'))
    out=dict(checks=checks,passed=len(checks)==5 and all(x['passed'] for x in checks),
        scope='N8 Layer qualification. Full N32 mixed arrival and lifecycle checks are separate gates.')
    write(ROOT/'analysis/layer_N8_state_qualification.json',out)
    print(json.dumps(dict(passed=out['passed'],checks=[dict({k:v for k,v in x.items() if k!='failures'},
        failure_count=len(x['failures'])) for x in checks]),indent=2))


if __name__=='__main__':main()
