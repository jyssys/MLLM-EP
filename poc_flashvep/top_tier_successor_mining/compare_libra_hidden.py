"""CPU-only first-divergence audit of explicit HF/native diagnostic captures."""
import argparse
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--hf',type=Path,required=True)
p.add_argument('--native',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(4)
rows=[]
for rank in range(4):
    hf=torch.load(a.hf/f'gqa_0_source{rank}.pt',map_location='cpu',weights_only=True)['hidden_diagnostic']
    native=torch.load(a.native/f'hidden_rank{rank}.pt',map_location='cpu',weights_only=True)
    for name,x in hf.items():
        y=native[name].float();x=x.float()
        assert x.shape==y.shape,(rank,name,x.shape,y.shape)
        row=dict(rank=rank,boundary=name,shape=list(x.shape),
                 relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-12)),
                 max_abs=float((x-y).abs().max()),
                 cosine=float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0)))
        rows.append(row)
        print(json.dumps(row))
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(rows,indent=2))
