"""Strict recorded-trajectory feedback loop; never weakens acceptance parity."""
import argparse
import collections
import json
from pathlib import Path
from poc_multibd_moe_ep.contract import write
from poc_multibd_moe_ep.scripts.analyze_conversion import acceptance_signature


def inspect(directory):
    groups=collections.defaultdict(list)
    for path in sorted(Path(directory).glob('request*.json')):
        result=json.loads(path.read_text())
        if result.get('is_warmup'):continue
        request=result['request']
        groups[(request['domain'],request['selected_index'],result['R'],
                result.get('deterministic_ops',False))].append((path,result))
    rows=[]
    for key,values in groups.items():
        reference=acceptance_signature(values[0][1])
        tests=[]
        for path,result in values:
            actual=acceptance_signature(result)
            first=next((i for i,(a,b) in enumerate(zip(reference['steps'],actual['steps'])) if a!=b),None)
            tests.append(dict(artifact=path.name,mode=result['mode'],
                final_tokens_equal=actual['tokens']==reference['tokens'],
                NFE_equal=len(actual['steps'])==len(reference['steps']),
                exact_acceptance_equal=actual['steps']==reference['steps'],
                pass_all=actual==reference,first_discrepant_step=first))
        rows.append(dict(domain=key[0],selected_index=key[1],R=key[2],deterministic_ops=key[3],
            comparisons=tests,pass_all=all(t['pass_all'] for t in tests)))
    return dict(groups=rows,pass_all=bool(rows) and all(r['pass_all'] for r in rows),
        note='Final token equality alone is insufficient; acceptance schedule and NFE are mandatory.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    value=inspect(args.directory);write(args.output,value)
    print(json.dumps(dict(pass_all=value['pass_all'],groups=[
        {k:r[k] for k in ['domain','selected_index','R','deterministic_ops','pass_all']}
        for r in value['groups']]),indent=2))
    raise SystemExit(0 if value['pass_all'] else 1)


if __name__=='__main__':main()
