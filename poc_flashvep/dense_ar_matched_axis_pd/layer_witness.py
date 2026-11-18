"""Observe DecoderLayer calls even when the engine bypasses the LM wrapper."""
import atexit
import time
from .common import ROOT,write


def install(cfg,role):
    if not cfg.get('independent_model_layer_witness'):
        return
    import torch
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    state=dict(batch=None,layers=[])
    calls=[]
    definitions=[]
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        batch=prepare(self,*a,**kw)
        state['batch']=batch
        return batch
    GPUModelRunner.prepare_inputs=prepared
    forward=Qwen3DecoderLayer.forward
    def observed(self,*a,**kw):
        number=int(self.self_attn.attn.layer_name.split('.')[2])
        capturing=torch.cuda.is_current_stream_capturing()
        if number==0:
            state['layers']=[]
        state['layers'].append(number)
        result=forward(self,*a,**kw)
        if number==35:
            actual=list(state['layers'])
            assert actual==list(range(36)),actual
            if capturing:
                definitions.append(dict(actual=actual,passed=True,time=time.monotonic()))
            else:
                batch=state['batch']
                if batch is not None and any(rid.split('-')[0].startswith('q') for rid in batch.req_ids):
                    calls.append(dict(time=time.monotonic(),actual=actual,passed=True,
                        requests=[rid.split('-')[0] for rid in batch.req_ids],
                        query_tokens=batch.num_tokens,
                        prefilling=bool(batch.is_prefilling_np[:batch.num_reqs].any())))
        return result
    Qwen3DecoderLayer.forward=observed
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/f'independent_layer_witness_{role}.json',dict(
        actual_calls=calls,captured_definitions=definitions,evidence='CORRECTNESS_ONLY',
        scope='Observe actual Qwen3DecoderLayer forward invocations. Captured definitions include all36 original layers. Runtime graph replay calls are recorded separately. No extra forward and no GPU duration measurement.')))
