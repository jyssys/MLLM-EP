"""No-op routing decisions: cost plus honest autoregressive parity boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
df=pd.DataFrame([json.loads(s) for f in a.input.glob('requests_dp*.jsonl') for s in f.read_text().splitlines()])
df=df[~df.warmup].copy()
base=df[df.policy=='vanilla']
def before_eos(tokens):
    return tokens[:next((i+1 for i,t in enumerate(tokens) if t in (151645,151643)),len(tokens))]
def describe(pair):
    pairs=list(zip(pair.output_tokens,pair.output_tokens_ref))
    return dict(paired_requests=len(pair),
        full32_equal_fraction=float(np.mean([x==y for x,y in pairs])),
        first_token_equal_fraction=float(np.mean([x[:1]==y[:1] for x,y in pairs])),
        through_first_eos_equal_fraction=float(np.mean([before_eos(x)==before_eos(y) for x,y in pairs])),
        paired_e2e_reduction_percent=float(np.median(100*(1-pair.e2e_ms/pair.e2e_ms_ref))))
rows=[]
for policy in ('sere_noop_s8_metadata_cached','modes_noop_metadata_cached'):
    pair=df[df.policy==policy].merge(base,on=['cohort','dp_rank','dataset_request'],
        suffixes=('','_ref'),validate='one_to_one')
    rows.append(dict(policy=policy,reference='vanilla_same_cohort',**describe(pair)))
base=base.copy();base['repetition']=base.cohort.str.extract(r'rep(\d+)').astype(int)
for i in (1,2):
    pair=base[base.repetition==i].merge(base[base.repetition==0],
        on=['dp_rank','dataset_request'],suffixes=('','_ref'),validate='one_to_one')
    rows.append(dict(policy=f'vanilla_rep{i}',reference='vanilla_rep0',**describe(pair)))
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'summary.json').write_text(json.dumps(dict(comparisons=rows,
    scope='Algorithmic no-op identity, not assumed bitwise cross-run generation identity; fixed32 is not benchmark quality.',
    source_identity='SERE retains all8; MoDES thresholds zero; official primitive identity tests separately passed.'),indent=2))
print(json.dumps(rows,indent=2))
