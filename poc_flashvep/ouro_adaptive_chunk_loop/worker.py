"""A single physical GPU worker with explicit attempt provenance."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import torch
import numpy as np
from .common import ROOT, LEGACY, FORK, MODEL, UUIDS, config, guard, write, digest

def identity():
    import vllm_rlt
    gpu = guard()
    loaded=Path(vllm_rlt.__file__).resolve()
    assert loaded.parent==(FORK/'vllm_rlt').resolve(),('WRONG_RUNTIME_IMPORT',loaded)
    assert torch.cuda.device_count() == 1
    torch.cuda.init()
    assert not torch.distributed.is_initialized(),'This worker must not form a TP/PP process group'
    actual = str(torch.cuda.get_device_properties(0).uuid)
    assert actual.removeprefix('GPU-') == UUIDS[gpu].removeprefix('GPU-'), (gpu, actual)
    return dict(physical_GPU=gpu, UUID=UUIDS[gpu], torch_UUID=actual,
        pid=os.getpid(), uid=os.getuid(), CVD=os.environ['CUDA_VISIBLE_DEVICES'],
        actual_runtime_import=str(loaded),loader_sha256=digest(FORK/'vllm_rlt/models/ouro.py'),
        logical_GPU=0, affinity=sorted(os.sched_getaffinity(0)), scope='INDEPENDENT_SINGLE_GPU')

def engine(model, mode, chunk, width, job):
    from vllm_rlt.engine.stage_engine import StageEngine
    result=StageEngine(model, **config(mode, chunk, width, blocks=job.get('blocks',18432),
        budget=job.get('budget',2048), graphs=job.get('graphs',True), streams=job.get('streams',True)))
    assert result.cache_manager.layout=='last_exited' and result.cache_manager.storage_depths==4
    return result

def dispose(e):
    e.close()

def bind_legacy(job):
    # Reuse immutable measurement code. Changes are process-local globals only.
    from poc_flashvep.ouro_chunk_loop_slo import worker as old
    old.ROOT = ROOT
    old.UUIDS = UUIDS
    old.guard = guard
    old.config = lambda mode,chunk,width=8,token_budget=2048: config(mode,chunk,width,
        blocks=job.get('blocks',18432),budget=job.get('budget',2048),
        graphs=job.get('graphs',True),streams=job.get('streams',True))
    old.new_engine = lambda model,mode,chunk,width=8,budget=2048: engine(model,mode,chunk,width,job)
    return old

def await_barrier(job, target):
    if not job.get('barrier'):
        return
    write(target/'READY.json', dict(time=time.time(), pid=os.getpid()))
    release = ROOT/'state'/f"release_{job['barrier']}.json"
    deadline = time.monotonic()+1800
    while not release.exists():
        assert time.monotonic() < deadline, 'Warm barrier timeout'
        time.sleep(.1)
    start = json.loads(release.read_text())['start_wall_s']
    while time.time()<start:
        time.sleep(.001)

def calibration(model, job, target):
    old = bind_legacy(job)
    summaries = []
    # Two contexts in the same start. A separate warmup precedes each shape.
    for shape_i,(length,width) in enumerate(job.get('shapes',[[2048,4],[8192,8]])):
        e = engine(model,'atomic',1024,width,job)
        cases = old.old_cases(0,length,width)
        warm = old.serve(e,cases,48,[0]*width,.001,observe=False)
        write(target/f'warm_P{length}_N{width}.json',warm)
        if shape_i == 0:
            await_barrier(job,target)
        for rep in range(job.get('repetitions',3)):
            data = old.serve(e,cases,96,[0]*width,.001,observe=True)
            ready = max(r['token_times'][0] for r in data['requests'])
            itls = [b-a for r in data['requests'] for a,b in zip(r['token_times'],r['token_times'][1:]) if a>=ready]
            data.update(job_id=job['id'], physical_GPU=guard(), P_bin=length,N=width,repetition=rep,
                purpose='CONTENTION_CALIBRATION_LEGACY_PROMPTS_NOT_HELDOUT',
                median_ITL_ms=float(np.median(itls))*1000,
                p95_ITL_ms=float(np.percentile(itls,95))*1000,
                p99_ITL_ms=float(np.percentile(itls,99))*1000,
                output_tokens=sum(len(r['output_ids']) for r in data['requests']))
            data['drained_output_tokens_s']=data['output_tokens']/(data['end']-data['start'])
            write(target/f'cal_P{length}_N{width}_r{rep}.json',data)
            summaries.append({k:data[k] for k in ('P_bin','N','repetition','median_ITL_ms','p95_ITL_ms','p99_ITL_ms','drained_output_tokens_s')})
        dispose(e)
        del e,warm,data
        gc.collect()
        torch.cuda.empty_cache()
    return dict(calibration=summaries)

def continuity(model,job,target):
    old = bind_legacy(job)
    from poc_flashvep.ouro_chunk_loop_prefill.bench import run
    for mode,chunk in job['cells']:
        length,width=job['shape']
        e=engine(model,mode,chunk,width,job)
        run(model,mode,chunk,length,width,0,False,job.get('graphs',True),False,e)
        for trial in job['trials']:
            path=target/f'{mode}_C{chunk}_P{length}_N{width}_T{trial}.json'
            if path.exists():
                continue
            data=run(model,mode,chunk,length,width,trial,job.get('trace',False),job.get('graphs',True),True,e)
            data.update(physical_GPU=guard(),job_id=job['id'],split='LEGACY',evidence='LIVE_MEASURED_CONTINUITY_NOT_NEW_HELDOUT')
            write(path,data)
        dispose(e)
        del e,data
        gc.collect()
        torch.cuda.empty_cache()
    return dict(status='CONTINUITY_COMPLETE')

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    a=p.parse_args()
    job=json.loads(Path(a.job).read_text())
    target=ROOT/'raw'/job['id']
    target.mkdir(parents=True,exist_ok=True)
    assert not (target/'completed.json').exists(),'Do not repeat a completed attempt'
    who=identity()
    write(target/'identity.json',dict(**who,job=job,start=time.time(),job_sha=digest(a.job),
        runtime_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=FORK,text=True).strip(),
        runtime_diff_sha=__import__('hashlib').sha256(subprocess.check_output(['git','diff','--binary'],cwd=FORK)).hexdigest()))
    try:
        from vllm_rlt.models import OuroForCausalLM
        torch.set_num_threads(1)
        model=OuroForCausalLM.from_pretrained(MODEL,device='cuda:0',dtype=torch.bfloat16)
        assert len(model.model.layers)==24 and model.config.total_ut_steps==4
        assert all(p.dtype==torch.bfloat16 for p in model.parameters())
        tensors=list(model.parameters())+list(model.buffers())
        storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
        write(target/'loaded_model_accounting.json',dict(parameter_numel=sum(p.numel() for p in model.parameters()),
            unique_parameter_buffer_storage_bytes=sum(storages.values()),physical_layers=24,static_R=4,
            parameter_dtypes=sorted(set(str(p.dtype) for p in model.parameters())),
            buffer_dtypes=sorted(set(str(p.dtype) for p in model.buffers())),
            allocated_before_KV=torch.cuda.memory_allocated(),reserved_before_KV=torch.cuda.memory_reserved(),
            scope='Loaded parameter/buffer storage and allocator state. Not measured HBM read traffic.'))
        del tensors,storages
        if job['kind']=='calibration':
            summary=calibration(model,job,target)
        elif job['kind']=='continuity':
            summary=continuity(model,job,target)
        elif job['kind']=='diagnostic':
            from .diagnostic import run_job
            summary=run_job(model,job,target)
        elif job['kind']=='surface':
            from .surface import run_job
            summary=run_job(model,job,target)
        elif job['kind']=='qualify':
            from .qualify import run_job
            summary=run_job(model,job,target)
        elif job['kind']=='quality':
            from .quality import run_job
            summary=run_job(model,job,target)
        elif job['kind']=='mlp_replay':
            from .mlp_replay import run_job
            summary=run_job(model,job,target)
        elif job['kind']=='measurement_controls':
            from .measurement_controls import run_job
            summary=run_job(model,job,target)
        else:
            raise ValueError(job['kind'])
        artifacts=[dict(path=str(p.relative_to(target)),bytes=p.stat().st_size,sha256=digest(p))
            for p in sorted(target.glob('*.json')) if p.name not in ('READY.json','completed.json')]
        write(target/'completed.json',dict(**who,summary=summary,artifacts=artifacts,end=time.time()))
    except BaseException as error:
        write(target/'failure.json',dict(**who,error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        raise

if __name__=='__main__':
    main()
