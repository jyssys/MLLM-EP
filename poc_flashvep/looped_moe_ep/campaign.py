import argparse
from .common import *
from .launch import launch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['ep8_p0','baseline','final_quality','ready_probe'])
    a=p.parse_args()
    results=[]
    if a.phase=='ep8_p0':
        configs=[dict(id=f'{model}_ep8_p0_v1',model=model,gpus=list(range(8)),phase='p0',timeout_s=1800)
            for model in ['foil1','ifm_ouro','ifm_huginn']]
    elif a.phase=='baseline':
        import time,json
        while not (ROOT/'campaign_ep8_p0.json').exists():time.sleep(5)
        assert all(r['status'] in ('PASS','ALREADY_COMPLETE') for r in json.loads((ROOT/'campaign_ep8_p0.json').read_text()))
        configs=[]
        # First cover every model/degree. Repeat primary cells only after this coverage.
        for start in range(3):
            for world in [4,8]:
                for model in ['foil1','ifm_ouro','ifm_huginn']:
                    configs.append(dict(id=f'{model}_ep{world}_baseline_s{start}_v1',model=model,
                        gpus=list(range(world)) if world==8 or start%2==0 else list(range(4,8)),
                        phase='baseline',quality_examples=32 if start==0 else 0,
                        warmups=10,repeats=20,lengths=[128,1024,2048] if start==0 else [128],
                        batches=[1,2,4],generation=start==0,generation_lengths=[32,128],
                        micro=start==0,replica_memory=start==0,profile=start==0,start_index=start*4,timeout_s=21600))
        write(ROOT/'configs/baseline_plan.json',dict(configs=configs,
            repetition_scope='3 starts at prefix128. Long prefixes and full generation matrix use 1 start with 20 forward repetitions.'))
    elif a.phase=='final_quality':
        import time,json
        while not (ROOT/'campaign_baseline.json').exists() or len(json.loads((ROOT/'campaign_baseline.json').read_text()))<18:
            time.sleep(5)
        configs=[dict(id=f'{model}_ep{world}_shape_quality_v2',model=model,gpus=list(range(world)),phase='quality',
            quality_examples=32,batches=[1,2,4],lengths=[128,1024,2048],timeout_s=1800)
            for world in [4,8] for model in ['foil1','ifm_ouro','ifm_huginn']]
    else:
        import time,json
        while not (ROOT/'campaign_final_quality.json').exists() or len(json.loads((ROOT/'campaign_final_quality.json').read_text()))<6:
            time.sleep(5)
        configs=[dict(id=f'{model}_ep{world}_ready_probe_v1',model=model,gpus=list(range(world)),phase='ready_probe',
            probe_tokens=8,probe_timeout_s=600,timeout_s=3600)
            for world in [4,8] for model in ['foil1','ifm_ouro','ifm_huginn']]
    for config in configs:
        try:results.append(launch(config))
        except Exception as e:
            results.append(dict(id=config['id'],status='FAILED',error=repr(e)))
        write(ROOT/f'campaign_{a.phase}.json',results)

if __name__=='__main__':main()
