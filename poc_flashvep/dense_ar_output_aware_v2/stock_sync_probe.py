"""Test a completion fence without copying or correcting model tensors."""
import atexit
import time
from .common import ROOT,write

def install(cfg,role):
    if role not in cfg.get('stock_sync_after_forward_roles',[]):return
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original=GPUModelRunner.execute_model;rows=[]
    def execute(runner,*args,**kwargs):
        result=original(runner,*args,**kwargs)
        frame=runner.execute_model_state
        if frame is not None and frame.hidden_states is not None and not torch.cuda.is_current_stream_capturing():
            left=time.monotonic();torch.cuda.synchronize();right=time.monotonic()
            rows.append(dict(start=left,end=right,requests=list(frame.input_batch.req_ids),
                scope='Post-forward device completion fence. No tensor copy or correction. Diagnostic timing only.'))
        return result
    GPUModelRunner.execute_model=execute
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/f'diagnostic_post_forward_fences_{role}.json',dict(rows=rows,
        evidence='DIAGNOSTIC',scope='A synchronization intervention,not a production optimization or an exclusive GPU timer.')))
