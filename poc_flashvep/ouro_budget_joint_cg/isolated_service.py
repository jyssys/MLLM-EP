"""Controlled endpoints for C/G interaction. This is a component benchmark."""
import gc
import json
import random
import torch
from .common import ROOT,write,guard
from .benchmark import engine
from .fairness import attach
from .service_observer import ServiceObserver

def register():
    target=ROOT/'p2_controlled_service_plan.json';assert not target.exists()
    jobs=[]
    for gpu in range(8):
        P=2048 if gpu%2==0 else 8192
        cells=[dict(mode=mode,C=C,fairness='completion') for C in (128,512,2048)
            for mode in ('split','grouped','grouped3','staged4')]
        random.Random(20261023+1700+gpu).shuffle(cells)
        jobs.append(dict(id=f'cb_p2_controlled_gpu{gpu}_v1',kind='isolated_service',gpu=gpu,
            shape=[P,1],split='TRAIN',cells=cells,trials=[6,7,8],cap=1,budget=2048,timeout_s=2400))
    write(target,dict(gpus=list(range(8)),jobs=jobs,
        scope='Single-request component cost. All C values divide the exact final KV endpoint. Cross-GPU concurrent diagnostic, not clean serving.'))

@torch.inference_mode()
def run_job(model,job,target):
    from ..ouro_chunk_loop_slo import worker as frontend
    frontend.ROOT=ROOT
    manifest=json.loads((ROOT/'source_family_manifest.json').read_text());P=job['shape'][0]
    sourceP=8177 if P==2048 else 16384
    sources=[c for c in manifest['cases'] if c['split']=='TRAIN' and c.get('P')==sourceP]
    assert len(sources)>=3,'Controlled replication requires at least three source families'
    results=[]
    for cell in job['cells']:
        mode=cell['mode'];C=cell['C'];e=engine(model,mode,C,1,job);attach(e,True)
        for trial in [-1]+job['trials']:
            source=sources[(trial if trial>=0 else 0)%len(sources)]
            case=dict(source,id=source['id']+f'_component_P{P}',input_ids=source['input_ids'][:P])
            meta=dict(run_id=job['id'],policy=mode,C=C,P=P,configured_N=1,trial=trial,
                physical_GPU=guard(),fairness='completion',split='TRAIN',families=[case['family']])
            observer=ServiceObserver(e,meta) if trial>=0 else None
            try:
                data=frontend.serve(e,[case],1,[0.],.001,observe=False)
                if observer:
                    rows=observer.finish()
                    assert sum(r['prompt_token_loop_work'] for r in rows)==4*P
                    endpoint=[r for r in rows if any(s['phase']=='prefilling_recurrent' and s['start']+s['tokens']==P for s in r['spans'])]
                    assert endpoint and all(r['context_max']==P for r in endpoint)
                    data.update(meta,service_observations=rows,endpoint_rows=[r['index'] for r in endpoint],
                        lane='EXPLORATORY_SHARED_HOST',full_R4_prompt_accounting=True,
                        scope='Component prompt excerpt. One bounded natural output token. Not answer quality or serving.')
                    write(target/f'{mode}_C{C}_component_T{trial}.json',data)
                    results.append(dict(mode=mode,C=C,trial=trial,P=P,endpoint_quanta=len(endpoint)))
            finally:
                if observer:
                    observer.close()
        e.close();del e,data
        gc.collect();torch.cuda.empty_cache()
    return dict(completed=True,endpoint_cases=results)

if __name__=='__main__':
    register()
