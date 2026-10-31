"""Declare independent service cells before inspecting their timing results."""
from .common import *
from .prepare import GRID,base

def main():
    for model in ('ouro','huginn'):
        name=f'phase_a_{model}_grid_start0';cfg=base(model,name)
        cfg.update(state_probe=False,phase='independent_capacity',estimated_seconds=5000)
        cfg['groups']=[dict(chunk=256,kv_blocks=cfg['kv_blocks'],jobs=[
            dict(cell=cell,dataset=f'test_L{length}.json',batch=b,cap=cap,repeats=3,instrument=True)
            for cell,(length,cap) in GRID.items() if cell.startswith('G') or cell in ('C1','C2')
            for b in (1,2,4)])]
        write(ROOT/'configs'/f'{name}.json',cfg)
        name=f'phase_a_{model}_chunk_start0';cfg=base(model,name)
        cfg.update(phase='independent_chunk_tuning',estimated_seconds=3000)
        cfg['groups']=[dict(chunk=chunk,kv_blocks=cfg['kv_blocks'],jobs=[
            dict(cell='G2',dataset='test_L3072.json',batch=b,cap=128,repeats=3,instrument=True)
            for b in (1,2)]) for chunk in (512,1024,2048)]
        write(ROOT/'configs'/f'{name}.json',cfg)
        name=f'phase_a_{model}_event_overhead_start0';cfg=base(model,name)
        cfg.update(phase='instrumentation_ablation',estimated_seconds=600)
        cfg['groups']=[dict(chunk=256,kv_blocks=cfg['kv_blocks'],jobs=[
            dict(cell='G2',dataset='test_L3072.json',batch=2,cap=128,repeats=3,instrument=on)
            for on in (False,True,False)])]
        write(ROOT/'configs'/f'{name}.json',cfg)
    cfg=dict(name='phase_a_qwen_grid_start0',model='qwen',gpus=[2],seed=2026103107,
        worker='ar_worker',estimated_seconds=1800,jobs=[
        dict(cell=cell,dataset=f'test_L{length}.json',batch=b,cap=cap,repeats=3)
        for cell,(length,cap) in GRID.items() if cell.startswith('G') or cell in ('C1','C2')
        for b in (1,2,4,8)])
    write(ROOT/'configs'/f'{cfg["name"]}.json',cfg)
    cfg=base('huginn','phase_a_huginn_expanded_pool_start0')
    cfg.update(phase='KV_capacity_ablation',estimated_seconds=1800)
    cfg['groups']=[dict(chunk=256,kv_blocks=26624,jobs=[
        dict(cell=cell,dataset=f'test_L{length}.json',batch=b,cap=cap,repeats=3,instrument=True)
        for cell,(length,cap) in (('G2',GRID['G2']),('C1',GRID['C1'])) for b in (2,4)])]
    write(ROOT/'configs'/f'{cfg["name"]}.json',cfg)
    print('PHASE_A_CONFIGS_FROZEN')

if __name__=='__main__':main()
