"""Compare captured prompt rows by absolute token position, not row offset."""
import argparse
import json
import torch
from .common import ROOT,write
from .diagnostic import compare

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--reference',default='atomic')
    p.add_argument('--candidate',default='grouped')
    a=p.parse_args()
    torch.set_num_threads(1)
    folder=ROOT/'raw'/a.run
    tensors=[torch.load(folder/f'{mode}_selected_tensors.pt',map_location='cpu',mmap=True,weights_only=True) for mode in [a.reference,a.candidate]]
    metadata=[json.loads((folder/f'{mode}_core_metadata.json').read_text()) for mode in [a.reference,a.candidate]]
    results=[]
    for depth in range(4):
        selected=[[r for r in rows if r['token_index']==0 and r['depth']==depth][-1] for rows in metadata]
        maps=[{position:i for i,position in enumerate(r['target_positions'])} for r in selected]
        positions=sorted(set(maps[0])&set(maps[1]))
        keys=sorted(k for k in tensors[0] if k.startswith(f't0_d{depth}_l0_') and k.endswith('all_target_rows'))
        for key in keys:
            x,y=[t[key][[m[p] for p in positions]] for t,m in zip(tensors,maps)]
            delta=(x.float()-y.float()).flatten(1).abs().max(dim=1).values
            changed=[positions[i] for i in range(len(positions)) if delta[i]!=0]
            results.append(dict(key=key,depth=depth,matched_absolute_positions=len(positions),
                reference_query_rows=selected[0]['actual_rows'],candidate_query_rows=selected[1]['actual_rows'],
                reference_max_q=selected[0]['max_q'],candidate_max_q=selected[1]['max_q'],
                nonexact_positions=len(changed),first_nonexact_position=changed[0] if changed else None,
                max_abs=float(delta.max()),exact=torch.equal(x,y)))
    write(ROOT/f'aligned_{a.run}_{a.reference}_vs_{a.candidate}.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
