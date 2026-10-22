"""A red-capable check of the actual saved batched mismatch symptom."""
import argparse
import json
from .common import LEGACY, ROOT, digest, write

def audit():
    rows=[]
    for folder in sorted((LEGACY/'raw').iterdir()):
        for reference in folder.glob('atomic_C1024_P8192_N8_T*.json'):
            candidate=folder/reference.name.replace('atomic_', 'grouped_')
            if not candidate.exists():
                continue
            a=json.loads(reference.read_text())
            b=json.loads(candidate.read_text())
            by_id={r['id']:r for r in b['requests']}
            for r in a['requests']:
                s=by_id[r['id']]
                prefix=next((i for i,(x,y) in enumerate(zip(r['output_ids'],s['output_ids'])) if x!=y),None)
                rows.append(dict(id=r['id'],trial=a['trial'],exact=r['output_ids']==s['output_ids'],
                    first_divergent_token=prefix,reference=str(reference),candidate=str(candidate),
                    reference_sha256=digest(reference),candidate_sha256=digest(candidate),
                    reference_tokens=len(r['output_ids']),candidate_tokens=len(s['output_ids']),
                    depth_exact=r['depths']==s['depths']))
    unique=[]
    seen=set()
    for row in rows:
        if not row['exact'] and row['id'] not in seen:
            unique.append(row)
            seen.add(row['id'])
    matched=[r for r in rows if r['exact']][:24]
    value=dict(scope='LEGACY_FROZEN_TRACE_RECHECK_NOT_NEW_GPU_EXECUTION',
        comparisons=len(rows),exact=sum(r['exact'] for r in rows),mismatches=unique,matched_controls=matched,
        all_pairs=rows)
    write(ROOT/'legacy_parity_registration.json',value)
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--assert-exact',action='store_true')
    a=p.parse_args()
    value=audit()
    print(json.dumps({k:v for k,v in value.items() if k not in ('all_pairs','matched_controls')},indent=2))
    if a.assert_exact:
        assert not value['mismatches'],'BATCH_TOKEN_PARITY_FAILURE_REPRODUCED_FROM_FROZEN_GPU_TRACE'
