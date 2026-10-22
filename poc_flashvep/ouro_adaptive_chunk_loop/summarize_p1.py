"""Summarize completed numerical jobs without claiming untested quality."""
import csv
import json
from .common import ROOT,write

def main():
    isolated=[]
    parity=[]
    for folder in (ROOT/'raw').glob('p1_isolated*'):
        if not (folder/'completed.json').exists():
            continue
        for path in folder.glob('P*.json'):
            d=json.loads(path.read_text())
            if 'comparison' in d:
                c=d['comparison']
                isolated.append(dict(run=folder.name,P=d['P'],C=d['C'],mode=d['mode'],
                    token_exact=c['token_exact'],depth_exact=c['depth_exact'],hidden_exact=c['hidden']['exact'],
                    logits_exact=c['logits']['exact'],KV_exact=c['KV_all_exact'],KV_max_relative_L2=c['KV_max_relative_L2']))
    for folder in (ROOT/'raw').glob('p1_legacy_reproduction*'):
        if not (folder/'completed.json').exists():
            continue
        references=list(folder.glob('atomic_C1024_*.json'))
        for ref in references:
            a=json.loads(ref.read_text())
            for path in folder.glob('*_C*_P*.json'):
                if path==ref:
                    continue
                b=json.loads(path.read_text())
                other={r['id']:r for r in b['requests']}
                for row in a['requests']:
                    candidate=other[row['id']]
                    parity.append(dict(run=folder.name,id=row['id'],candidate=b['mode'],C=b['chunk_tokens'],
                        token_exact=row['output_ids']==candidate['output_ids'],depth_exact=row['depths']==candidate['depths'],
                        first_divergent_token=next((i for i,(x,y) in enumerate(zip(row['output_ids'],candidate['output_ids'])) if x!=y),None),
                        scope='LIVE_BOUNDED128_LEGACY_CONTINUITY_NOT_ACCURACY'))
    for name,rows in [('isolated_numerical_results.csv',isolated),('batch_parity.csv',parity)]:
        if rows:
            with (ROOT/name).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(isolated_comparisons=len(isolated),isolated_all_exact=all(r['token_exact'] and r['hidden_exact'] and r['logits_exact'] and r['KV_exact'] for r in isolated),
        bounded_pair_counts={m:dict(pairs=sum(r['candidate']==m for r in parity),exact=sum(r['candidate']==m and r['token_exact'] for r in parity)) for m in sorted({r['candidate'] for r in parity})},
        quality='NOT_RUN_UNTIL_NEW_QUALITY_CASES_COMPLETE')
    write(ROOT/'phase1_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
