"""Verify real, untruncated W1 prompts before the serving expansion."""
import json
from .common import *
from .prepare import base

def main():
    source=json.loads((ROOT/'data/W1_calibration.json').read_text())
    # Include both short and longest-bin full dialogue prompts.
    selected=source[:4]+[r for r in source[4:] if all(v>=2800 for v in r['input_lengths'].values())][:4]
    selected=[dict(r,cap=8) for r in selected]
    if len(selected)!=8:raise RuntimeError('Missing full-text long gate prompts')
    write(ROOT/'data/long_gate8.json',selected)
    for model,gpus in [('ouro',[0,1]),('huginn',[4,5])]:
        for arm in ('PD_G1','Loop_Static','Token_Static','Loop_Dynamic','Token_Dynamic'):
            name=f'long_gate8_{model}_{arm.lower()}_v1'
            cfg=base(model,arm,name,'long_gate8.json',8,gpus)
            cfg.update(state_probe=True,verify_transfer=True,timeout=2400,
                       cut=MODEL_PINS[model][2]//2 if arm.startswith('Loop') else MODEL_PINS[model][2],
                       seed=2026103171)
            write(ROOT/'configs'/f'{name}.json',cfg)

if __name__=='__main__':main()
