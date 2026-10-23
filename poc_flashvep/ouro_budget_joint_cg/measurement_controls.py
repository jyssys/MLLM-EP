"""Pair clean and ready-stream observer lanes on the same native path."""
import gc
import json
import random
import torch
from .common import ROOT,write,guard
from .benchmark import engine,select_cases
from .fairness import attach
from .service_observer import ServiceObserver

def register():
    target=ROOT/'p1_observer_control_plan.json';assert not target.exists()
    groups=[]
    for gpu in (0,1):
        cells=[dict(mode=mode,C=512,fairness='completion') for mode in ('atomic','staged4','grouped','grouped3','split','mixed')]
        random.Random(20261023+gpu+1300).shuffle(cells)
        job=dict(id=f'cb_p1_observer_control_gpu{gpu}_v1',kind='observer_control',gpu=gpu,
            shape=[2033,4] if gpu==0 else [8177,8],split='TRAIN',cells=cells,trials=[4,5],cap=64,
            budget=2048,quiet=True,timeout_s=2400)
        groups.append(dict(jobs=[job],barrier=None))
    write(target,dict(gpus=list(range(8)),groups=groups,
        scope='Quiet same-GPU clean/event crossover. Other GPUs are isolation-reserved. Not idle allocation.'))

@torch.inference_mode()
def run_job(model,job,target):
    from ..ouro_chunk_loop_slo import worker as frontend
    frontend.ROOT=ROOT
    manifest=json.loads((ROOT/'source_family_manifest.json').read_text());reports=[]
    P,N=job['shape']
    for cell in job['cells']:
        mode=cell['mode'];C=cell['C'];e=engine(model,mode,C,N,job)
        e.frontend_stop_token_ids=manifest['stop_ids'];attach(e,True)
        cases,arrivals=select_cases(manifest,'TRAIN',P,N,job['trials'][0])
        frontend.serve(e,cases,16,arrivals,.251,observe=False)
        for trial in job['trials']:
            cases,arrivals=select_cases(manifest,'TRAIN',P,N,trial)
            lanes=['clean','events'] if (trial+guard())%2==0 else ['events','clean']
            pair={}
            for lane in lanes:
                metadata=dict(run_id=job['id'],policy=mode,C=C,P=P,configured_N=N,trial=trial,
                    physical_GPU=guard(),fairness='completion',split='TRAIN')
                observer=ServiceObserver(e,metadata) if lane=='events' else None
                try:
                    data=frontend.serve(e,cases,job['cap'],arrivals,.251,observe=False)
                    data.update(metadata,service_observations=[] if observer is None else observer.finish(),
                        lane='MEASURED_CLEAN' if lane=='clean' else 'MEASURED_OBSERVER',
                        evidence='LIVE_MATCHED_OBSERVER_CONTROL_NOT_CAPACITY')
                    write(target/f'{mode}_C{C}_{lane}_T{trial}.json',data);pair[lane]=data
                finally:
                    if observer:
                        observer.close()
            a={r['id']:r for r in pair['clean']['requests']};b={r['id']:r for r in pair['events']['requests']}
            reports.append(dict(policy=mode,trial=trial,requests=len(a),
                output_exact=sum(a[k]['output_ids']==b[k]['output_ids'] for k in a),
                length_exact=sum(len(a[k]['output_ids'])==len(b[k]['output_ids']) for k in a),
                depths_exact=sum(a[k]['depths']==b[k]['depths'] for k in a),
                clean_wall_s=pair['clean']['end']-pair['clean']['start'],
                event_wall_s=pair['events']['end']-pair['events']['start']))
        e.close();del e,pair,data
        gc.collect();torch.cuda.empty_cache()
    write(target/'observer_pairs.json',reports)
    return dict(pairs=reports,scope='Observer parity can fail from batch geometry. No automatic equivalence claim.')

if __name__=='__main__':
    register()
