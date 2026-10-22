"""Reconstruct every captured prompt row across all chunk boundaries."""
import argparse
import json
import re
import torch
from .common import ROOT,write

KINDS=['input_norm','Q_projection','K_projection','V_projection','post_rope_Q',
    'attention_pre_output_projection','attention','post_attention_norm','MLP','layer_output']

def collect(tensors,depth,layer,kind):
    if kind in ('post_rope_Q','attention_pre_output_projection'):
        pattern=rf'^t0_d{depth}_l{layer}_p(\d+)_(\d+)_{kind}$'
    else:
        pattern=rf'^t0_d{depth}_l{layer}_{kind}_p(\d+)_(\d+)$'
    result={}
    for key,tensor in tensors.items():
        match=re.match(pattern,key)
        if match:
            start,end=map(int,match.groups())
            assert len(tensor)==end-start+1,(key,tensor.shape)
            for offset,pos in enumerate(range(start,end+1)):
                result[pos]=tensor[offset]
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--reference',default='atomic')
    p.add_argument('--candidate',default='grouped')
    a=p.parse_args()
    torch.set_num_threads(1)
    folder=ROOT/'raw'/a.run
    tensors=[torch.load(folder/f'{mode}_selected_tensors.pt',map_location='cpu',mmap=True,weights_only=True) for mode in (a.reference,a.candidate)]
    records=[]
    for depth in range(4):
        for layer in range(2):
            for kind in KINDS:
                values=[collect(t,depth,layer,kind) for t in tensors]
                positions=sorted(set(values[0])&set(values[1]))
                if not positions:
                    continue
                x,y=[torch.stack([v[i] for i in positions]) for v in values]
                delta=(x.float()-y.float()).flatten(1).abs().max(dim=1).values
                changes=[positions[i] for i in range(len(positions)) if delta[i]!=0]
                records.append(dict(depth=depth,layer=layer,kind=kind,positions_compared=len(positions),
                    nonexact_positions=len(changes),first_nonexact_position=changes[0] if changes else None,
                    max_abs=float(delta.max()),exact=torch.equal(x,y),
                    relative_L2=float(torch.linalg.vector_norm(x.float()-y.float())/torch.linalg.vector_norm(x.float()).clamp_min(1e-12))))
    first=next((r for r in records if not r['exact']),None)
    write(ROOT/f'attention_boundary_{a.run}_{a.reference}_vs_{a.candidate}.json',dict(rows=records,
        first_captured_difference=first,scope='All captured prompt chunks aligned by absolute position. Not an accuracy evaluation.'))
    print(json.dumps(dict(run=a.run,comparisons=len(records),first_captured_difference=first),indent=2))

if __name__=='__main__':
    main()
