"""Preserve ordinary chunk-streaming PD_G1 as a separate strong control."""
import json
from .common import ROOT,write


def main():
    for model,depth in (('ouro',4),('huginn',32)):
        cfg=json.loads((ROOT/'configs'/f'gate_{model}_L512_cut{depth}.json').read_text())
        name=f'gate_{model}_native_G1_L512'
        cfg.update(name=name,count=2,arrival_offsets=[0.,.02],handoff_mode='native_G1',
                   verify_transfer=True,state_probe=True)
        destination=ROOT/'configs'/f'{name}.json'
        if destination.exists():raise FileExistsError(destination)
        write(destination,cfg)
        source=json.loads((ROOT/'configs'/f'map_{model}_L512_frozen_choices_start3.json').read_text())
        name=f'map_{model}_L512_native_G1_start3'
        source.update(name=name,handoff_mode='native_G1',
                      cohorts=[c for c in source['cohorts'] if c['cut']==depth])
        destination=ROOT/'configs'/f'{name}.json'
        if destination.exists():raise FileExistsError(destination)
        write(destination,source)


if __name__=='__main__':main()
