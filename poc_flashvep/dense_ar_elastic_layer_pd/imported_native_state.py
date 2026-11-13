"""Hash imported Native prompt KV without an additional model execution."""
import atexit
from .common import *


def install(cfg,role):
    if not cfg.get('imported_native_state_witness') or role!='decode':return
    import torch
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    state={};complete={};prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        batch=prepare(self,*a,**kw);state.update(batch=batch,runner=self);return batch
    GPUModelRunner.prepare_inputs=prepared
    def observe():
        batch=state.get('batch');runner=state.get('runner')
        if batch is None:return
        for i,rid in enumerate(batch.req_ids):
            external=rid.split('-')[0]
            if not (external.startswith('q') and external[1:].isdigit()) or external in complete:continue
            if int(batch.num_computed_tokens_np[i])<cfg['length']:continue
            positions=torch.arange(cfg['length'],device='cuda');table=runner.block_tables.input_block_tables[0][i];planes=[]
            for layer in range(36):
                attn=runner.vllm_config.compilation_config.static_forward_context[f'model.layers.{layer}.self_attn.attn']
                cache=attn.kv_cache
                value=cache[table[positions//16].long(),:,positions%16,:].detach().contiguous().view(torch.uint8).cpu()
                planes.append(dict(layer=layer,SHA256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),bytes=value.numel()))
            complete[external]=dict(KV=planes)
    forward=Qwen3ForCausalLM.forward
    def observed(self,*a,**kw):
        value=forward(self,*a,**kw);observe();return value
    Qwen3ForCausalLM.forward=observed
    replay=CudaGraphManager.run_fullgraph
    def replayed(self,*a,**kw):
        value=replay(self,*a,**kw);observe();return value
    CudaGraphManager.run_fullgraph=replayed
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'imported_native_prompt_KV.json',dict(requests=complete,
        evidence='CORRECTNESS_ONLY',scope='All36 valid original prompt KV planes after a real new-token query. CPU copies perturb timing. No prompt query or model forward is added.')))
