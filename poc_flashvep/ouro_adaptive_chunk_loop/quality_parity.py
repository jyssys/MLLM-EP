"""Compare actual free-running quality outputs, not aggregate token counts."""
import csv
import json
from .common import ROOT,write

def main():
    by_policy={}
    for folder in (ROOT/'raw').glob('p1_quality_pilot*'):
        for path in folder.glob('*.json'):
            row=json.loads(path.read_text())
            if isinstance(row,dict) and 'gold' in row and 'output_ids' in row:
                by_policy.setdefault(row['policy'],{})[row['id']]=row
    rows=[]
    for mode in ('grouped','mixed'):
        for rid,a in by_policy.get('atomic',{}).items():
            if rid not in by_policy.get(mode,{}):
                continue
            b=by_policy[mode][rid]
            rows.append(dict(id=rid,dataset=a['dataset'],candidate=mode,prompt_exact=a['prompt_ids']==b['prompt_ids'],
                token_exact=a['output_ids']==b['output_ids'],depth_exact=a['depths']==b['depths'],
                length_exact=len(a['output_ids'])==len(b['output_ids']),stop_exact=a['finish_reason']==b['finish_reason'],
                first_divergent_token=next((i for i,(x,y) in enumerate(zip(a['output_ids'],b['output_ids'])) if x!=y),None)))
    if rows:
        with (ROOT/'quality_output_parity.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={mode:dict(N=sum(r['candidate']==mode for r in rows),token_exact=sum(r['candidate']==mode and r['token_exact'] for r in rows),
        prompt_exact=sum(r['candidate']==mode and r['prompt_exact'] for r in rows)) for mode in ('grouped','mixed')}
    write(ROOT/'quality_output_parity.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
