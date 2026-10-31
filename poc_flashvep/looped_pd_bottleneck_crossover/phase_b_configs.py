"""Initial native-PD controls. Rates use measured independent service only."""
import random
import statistics
from .common import *
from .prepare import base,GRID

def pd_config(model,name,cell='G2',count=24,rate=.1,calibration=False):
    length,cap=GRID[cell];c=base(model,name)
    split='calibration' if calibration else 'test64' if count>32 else 'test'
    c.update(worker='serving_worker',axis='loop',dynamic=False,count=count,workload=cell,handoff_mode='native_G1',
        phase='calibration' if calibration else 'crossover',dataset=f'{split}_L{length}.json',
        capacity=8,chunk=256,token_budget=256,timeout=5400,estimated_seconds=1800,
        gpus=[0,1] if model=='ouro' else [4,5],request_caps=[cap]*count,
        arrival_rate=rate,arrival_seed=2026103111,warmup_count=2,warmup_cap=2)
    rng=random.Random(c['arrival_seed']);times=[0.]
    for i in range(count-1):times.append(times[-1]+rng.expovariate(rate))
    c['arrival_offsets']=times
    rows=json.loads((ROOT/'data'/c['dataset']).read_text())[:count]
    assert len(rows)==count,'The declared cohort needs enough frozen Prompt records'
    c['prompt_ids']=[r['prompt_id'] for r in rows]
    return c

def main():
    for model in ('ouro','huginn'):
        c=pd_config(model,f'gate_{model}_native_pd_v1',count=2,rate=100.)
        c.update(dataset='test_L512.json',request_caps=[16]*2,arrival_offsets=[0.,0.],
            verify_transfer=True,state_probe=True,phase='transport_correctness',estimated_seconds=300)
        write(ROOT/'configs'/(c['name']+'.json'),c)
    print('INITIAL_PD_GATES_READY')

if __name__=='__main__':main()
