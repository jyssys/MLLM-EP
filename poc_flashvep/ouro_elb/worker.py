"""One model process per physical GPU. Failed attempts remain immutable."""
import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import torch
from .common import ROOT,FORK,REPO,MODEL,UUIDS,UPSTREAM,guard,write,digest

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job=json.loads(Path(a.job).read_text());target=ROOT/'raw'/job['id']
    target.mkdir(parents=True,exist_ok=True)
    assert not (target/'completed.json').exists()
    gpu=guard();torch.set_num_threads(1)
    import vllm_rlt
    assert Path(vllm_rlt.__file__).resolve().parent==FORK/'vllm_rlt'
    assert torch.cuda.device_count()==1 and not torch.distributed.is_initialized()
    actual=str(torch.cuda.get_device_properties(0).uuid)
    assert actual.removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
    diff=subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=FORK,text=True).strip()==UPSTREAM
    write(target/'sources.json',{p.name:dict(sha256=digest(p),text=p.read_text()) for p in Path(__file__).parent.glob('*.py')})
    write(target/'identity.json',dict(job=job,gpu=gpu,UUID=UUIDS[gpu],pid=os.getpid(),uid=os.getuid(),
        affinity=sorted(os.sched_getaffinity(0)),CVD=os.environ['CUDA_VISIBLE_DEVICES'],start_wall_s=time.time(),
        runtime_pin=UPSTREAM,runtime_diff=diff.decode(),runtime_diff_sha256=hashlib.sha256(diff).hexdigest(),
        source_sha256=digest(target/'sources.json'),repo_SHA=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
    try:
        from vllm_rlt.models import OuroForCausalLM
        model=OuroForCausalLM.from_pretrained(MODEL,device='cuda:0',dtype=torch.bfloat16)
        assert len(model.model.layers)==24 and model.config.total_ut_steps==4
        assert all(p.dtype==torch.bfloat16 for p in model.parameters())
        storages={p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}
        write(target/'model_facts.json',dict(model_revision=MODEL.name,physical_layers=24,recurrent_depth=4,
            unique_weight_storage_bytes=sum(storages.values()),parameters=sum(p.numel() for p in model.parameters()),
            precision='BF16',config={k:getattr(model.config,k,None) for k in ('hidden_size','num_attention_heads','num_key_value_heads','max_position_embeddings')}))
        from .experiments import run_job
        result=run_job(model,job,target)
        files=[dict(path=str(p.relative_to(target)),bytes=p.stat().st_size,sha256=digest(p))
            for p in sorted(target.iterdir()) if p.is_file() and p.name not in ('READY.json','completed.json','telemetry.jsonl')]
        write(target/'completed.json',dict(gpu=gpu,summary=result,artifacts=files,end_wall_s=time.time()))
    except BaseException as error:
        write(target/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),wall_s=time.time()))
        raise

if __name__=='__main__':main()
