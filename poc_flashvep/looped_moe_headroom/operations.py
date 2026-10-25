"""Generate frozen run manifests. These files are plans, not execution evidence."""
import argparse
import json
from .common import ROOT, write

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['p0','p1','calibration','heldout','timing','generation_fidelity','generation_quality'])
    a=p.parse_args()
    workloads=json.loads((ROOT/'workload_manifest.json').read_text())['rows']
    jobs=[]
    models=['ifm_ouro','ifm_huginn','foil1']
    if a.phase=='p0':
        for model,gpu in zip(models,[0,3,6]):
            jobs.append(dict(id=f'{model}_p0_gpu{gpu}_native_v5',model=model,phase='p0',gpus=[gpu]))
    elif a.phase=='p1':
        for model in models:
            for domain in ['language','gsm8k','math500','synthetic']:
                selected=[r for r in workloads if r['dataset']==domain and r['split'] in ('calibration','smoke')]
                selected=selected[:8] if domain!='synthetic' else selected
                jobs.append(dict(id=f'{model}_p1_{domain}_v3',model=model,phase='p1',ids=[r['id'] for r in selected],
                    lengths=[128,512,2048] if domain=='synthetic' else [512],
                    decode_steps=4 if domain in ('gsm8k','math500') else 0,timeout_s=14400))
    elif a.phase in ('calibration','heldout'):
        split='calibration' if a.phase=='calibration' else 'heldout'
        selected=[r for r in workloads if r['dataset']=='language' and r['split']==split]
        for model in models:
            for shard in range(8):
                subset=selected[shard::8]
                jobs.append(dict(id=f'{model}_p2_{split}_shard{shard}_v1',model=model,phase='p2',
                    ids=[r['id'] for r in subset],length=128 if split=='calibration' else 512,timeout_s=21600))
    elif a.phase.startswith('generation'):
        from .study import arms
        fidelity=a.phase=='generation_fidelity'
        selected=[]
        for domain in (['language','gsm8k','math500'] if fidelity else ['gsm8k','math500']):
            selected.extend([r for r in workloads if r['dataset']==domain and r['split']=='heldout'][:2 if fidelity else 8])
        for model in models:
            for shard,row in enumerate(selected):
                jobs.append(dict(id=f'{model}_{a.phase}_{shard}_v1',model=model,phase='generation',ids=[row['id']],
                    arms=[v['id'] for v in arms()] if fidelity else ['O1_drop_0.1_renorm0','C1_router_0.2','C2_reuse_guardNone'],
                    prefix_cap=128 if fidelity else 2048,cap=16 if fidelity else 256,timeout_s=21600))
                if not fidelity:
                    jobs[-1].update(prompt_style='official_ifm_gsm8k_8shot',stop_strings=['\nQ:','\nProblem:'])
    else:
        ids=[r['id'] for r in workloads if r['dataset']=='language' and r['split']=='smoke'][:1]
        for model in models:
            for repeat in range(3):
                jobs.append(dict(id=f'{model}_timing_start{repeat}_v1',model=model,phase='timing',ids=ids,
                    lengths=[128,512,2048],repeats=20,warmups=5,generation_lengths=[16,64],
                    generation_prefix_lengths=[128,512],generation_repeats=1,timeout_s=14400))
    target=ROOT/('plan_p0_v5.json' if a.phase=='p0' else f'plan_{a.phase}.json')
    if target.exists():
        assert json.loads(target.read_text())==dict(phase=a.phase,jobs=jobs)
    else:
        write(target,dict(phase=a.phase,jobs=jobs))
    print(target,len(jobs))

if __name__=='__main__':
    main()
