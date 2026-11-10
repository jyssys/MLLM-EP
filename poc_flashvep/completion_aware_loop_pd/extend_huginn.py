"""Qualify earlier exact frontiers before using them in a Loop candidate set."""
import argparse
import copy
from .common import *
from .secondary import recovery_config


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('static8','mixed32','selection'));a=p.parse_args()
    initialize();jobs=[]
    if a.stage=='static8':
        original=json.loads((ROOT/'configs/secondary_huginn_loop_state8.json').read_text())
        for cut in (16,21):
            cfg=copy.deepcopy(original)
            cfg.update(name=f'earlier_huginn_loop{cut}_state8',intervention_method='static_intervention',
                native_partial_cut=cut,boundary_checkpoints=[cut],require_actual_offload_qualification=False,
                phase='EARLIER_FRONTIER_EXACT_STATE_NOT_PERFORMANCE')
            save(cfg);jobs.append(cfg['name'])
    else:
        for cut in (16,21):
            path=ROOT/'raw'/f'earlier_huginn_loop{cut}_state8'/'qualification_with_frozen_reference.json'
            assert path.exists() and json.loads(path.read_text())['passed'],'An earlier frontier lacks exact GPU qualification'
        if a.stage=='mixed32':
            cfg=json.loads((ROOT/'configs/secondary_huginn_loop_state32.json').read_text())
            cfg.update(name='earlier_huginn_dynamic_mixed_state32',boundary_checkpoints=[16,21,24,28,30,31],
                phase='EARLIER_FRONTIER_ACTUAL_MIXED_N32_EXACT_STATE_NOT_PERFORMANCE')
        else:
            name='earlier_huginn_loop_selection32';cfg=recovery_config('huginn',name,'loop',32,2026110920)
            cfg.update(boundary_checkpoints=[16,21,24,28,30,31],
                required_qualification='earlier_huginn_dynamic_mixed_state32',
                phase='EARLIER_QUALIFIED_FRONTIER_SELECTION_NOT_PRIMARY')
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state'/f'earlier_{a.stage}_jobs.json',jobs);print(' '.join(jobs))


if __name__=='__main__':main()
