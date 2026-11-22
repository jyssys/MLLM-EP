"""Test a pre-forward device fence without reading or correcting tensors."""
import atexit
import time
from poc_flashvep.dense_ar_longbench400_pd.common import ROOT,write

def install(cfg,role):
    if role!='decode':return
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original=GPUModelRunner.execute_model;rows=[]
    def execute(runner,*args,**kwargs):
        if cfg.get('diagnostic_D_pre_forward_fence') and not kwargs.get('dummy_run') and not torch.cuda.is_current_stream_capturing():
            begin=time.monotonic();torch.cuda.synchronize();end=time.monotonic()
            rows.append(dict(start=begin,end=end,scope='Pre-forward device fence. No tensor read or correction.'))
        return original(runner,*args,**kwargs)
    GPUModelRunner.execute_model=execute
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'diagnostic_D_pre_forward_fences_v14.json',dict(rows=rows,
        enabled=bool(cfg.get('diagnostic_D_pre_forward_fence')),evidence='DIAGNOSTIC',
        scope='A device-ordering counterfactual. Host duration is not exclusive GPU time. '
            'This does not prove external NIXL DMA readiness or identify one race. Not a production optimization.')))
