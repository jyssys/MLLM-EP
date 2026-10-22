"""Matched clean/event pairs and a separate heavy CUDA trace."""
import gc
import json
import random
import time
import torch
from .common import ROOT,write,digest
from .surface import QuantumObserver,select_cases

@torch.inference_mode()
def run_job(model,job,target):
    from .worker import engine,dispose,bind_legacy
    old=bind_legacy(job)
    manifest=json.loads((ROOT/'workload_manifest.json').read_text())
    summaries=[]
    for mode,chunk in job['cells']:
        e=engine(model,mode,chunk,job['shape'][1],job)
        e.frontend_stop_token_ids=manifest['stop_ids']
        cases,arrivals=select_cases(manifest,'TRAIN',*job['shape'],0)
        old.serve(e,cases,16,arrivals,.251,observe=False)
        lanes=['clean','events']
        random.Random(job['seed']+chunk).shuffle(lanes)
        for rep in range(job.get('repetitions',3)):
            for lane in lanes if rep%2==0 else lanes[::-1]:
                meta=dict(run_id=job['id'],physical_gpu=int(__import__('os').environ['CUDA_VISIBLE_DEVICES']),
                    split='TRAIN',policy=mode,configured_C=chunk,P=job['shape'][0],N=job['shape'][1],
                    trial=rep,source_families=[c['family'] for c in cases])
                observer=QuantumObserver(e,meta) if lane=='events' else None
                data=old.serve(e,cases,32,arrivals,.251,observe=False)
                if observer:
                    data['quantum_observations']=observer.finish(data)
                    del observer
                data.update(meta,lane=lane,evidence='LIVE_MATCHED_OBSERVER_CONTROL',
                    metric_scope='Bounded natural cohort, not sustainable capacity.')
                path=target/f'{mode}_C{chunk}_{lane}_r{rep}.json'
                write(path,data)
                summaries.append(dict(policy=mode,C=chunk,lane=lane,rep=rep,wall_s=data['end']-data['start']))
        if job.get('heavy_profile'):
            original=e.executor.submit
            def annotated(batch):
                label='+'.join(sorted({batch.phase_for(r).value for r in batch.requests}))
                with torch.profiler.record_function('ouro_quantum/'+label):
                    return original(batch)
            e.executor.submit=annotated
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=False,with_stack=False) as profiler:
                data=old.serve(e,cases,8,arrivals,.251,observe=False)
            e.executor.submit=original
            trace=target/f'{mode}_C{chunk}_heavy_trace.json'
            profiler.export_chrome_trace(str(trace))
            write(target/f'{mode}_C{chunk}_heavy_result.json',dict(data,
                evidence='HEAVY_PROFILER_NOT_HEADLINE_LATENCY',trace=trace.name,sha256=digest(trace)))
            # A bound submit method retains the old runner and its KV tensors.
            # Release diagnostic references before the next engine.
            del profiler,original,annotated
        dispose(e)
        del e,data
        gc.collect();torch.cuda.empty_cache()
    return dict(status='MATCHED_MEASUREMENT_CONTROLS_COMPLETE',pairs=summaries)

def register():
    cells=[['atomic',512],['staged4',512],['grouped',512],['grouped3',512],['split',512],['mixed',512]]
    jobs=[]
    for gpu in range(8):
        rotated=cells[gpu%6:]+cells[:gpu%6]
        jobs.append(dict(id=f'p2_observer_control_gpu{gpu}_v1',kind='measurement_controls',gpu=gpu,
            cells=rotated,shape=[2033,4] if gpu%2==0 else [8177,8],repetitions=3,
            heavy_profile=gpu in (0,1),budget=2048,seed=20261023+gpu,timeout_s=3600))
    path=ROOT/'measurement_control_plan.json'
    assert not path.exists()
    write(path,dict(gpus=list(range(8)),jobs=jobs,scope='All policies rotate on each device. Heavy traces are separate from clean/event pairs.'))
    print(path)

if __name__=='__main__':
    register()
