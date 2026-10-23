"""One GPU, one immutable attempt, explicit source and device identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import torch
from .common import ROOT,PRIOR,REPO,FORK,MODEL,UUIDS,write,digest,guard

def source_snapshot(target):
    source={p.name:dict(sha256=digest(p),text=p.read_text()) for p in Path(__file__).parent.glob('*.py')}
    write(target/'harness_sources.json',source)
    diff=subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    return dict(runtime_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=FORK,text=True).strip(),
        runtime_diff_sha256=hashlib.sha256(diff).hexdigest(),runtime_diff=diff.decode(),
        harness_sha256=digest(target/'harness_sources.json'),
        repository_SHA=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();job=json.loads(Path(args.job).read_text())
    target=ROOT/'raw'/job['id'];target.mkdir(parents=True,exist_ok=True)
    assert not (target/'completed.json').exists()
    gpu=guard()
    import vllm_rlt
    assert Path(vllm_rlt.__file__).resolve().parent==FORK/'vllm_rlt'
    assert torch.cuda.device_count()==1 and not torch.distributed.is_initialized()
    torch.set_num_threads(1)
    actual=str(torch.cuda.get_device_properties(0).uuid)
    assert actual.removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
    who=dict(physical_GPU=gpu,UUID=UUIDS[gpu],pid=os.getpid(),uid=os.getuid(),CVD=os.environ['CUDA_VISIBLE_DEVICES'],
        affinity=sorted(os.sched_getaffinity(0)),runtime_path=str(FORK),job=job,job_sha256=digest(args.job),
        scope='INDEPENDENT_SINGLE_GPU',start_wall_s=time.time(),**source_snapshot(target))
    write(target/'identity.json',who)
    try:
        from vllm_rlt.models import OuroForCausalLM
        model=OuroForCausalLM.from_pretrained(MODEL,device='cuda:0',dtype=torch.bfloat16)
        assert len(model.model.layers)==24 and model.config.total_ut_steps==4
        assert all(p.dtype==torch.bfloat16 for p in model.parameters())
        if job['kind']=='calibration':
            from ..ouro_adaptive_chunk_loop import worker as previous
            previous.ROOT=ROOT;previous.FORK=FORK
            summary=previous.calibration(model,job,target)
        else:
            from .benchmark import run_job
            summary=run_job(model,job,target)
        artifacts=[dict(path=str(p.relative_to(target)),bytes=p.stat().st_size,sha256=digest(p))
            for p in sorted(target.glob('*.json')) if p.name not in ('READY.json','completed.json')]
        write(target/'completed.json',dict(physical_GPU=gpu,UUID=UUIDS[gpu],summary=summary,artifacts=artifacts,end_wall_s=time.time()))
    except BaseException as error:
        write(target/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),wall_s=time.time()))
        raise

if __name__=='__main__':
    main()
