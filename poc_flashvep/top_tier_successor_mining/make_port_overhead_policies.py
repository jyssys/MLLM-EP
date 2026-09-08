"""Matched implementation-cost controls; paper method/thresholds unchanged."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
original=json.loads(a.input.read_text());rows=[]
for policy in original:
    rows.append(policy)
    if policy['method']!='vanilla':
        rows.append(dict(policy,name=policy['name']+'_metadata_cached',cache_token_metadata=True))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(policies=len(rows),path=str(a.out),scope='EQUIVALENT_TOKEN_METADATA_LIFETIME_CONTROL_NOT_SUCCESSOR')))
