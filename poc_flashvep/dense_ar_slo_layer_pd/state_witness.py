"""Hash original prompt state without a second model forward."""
import atexit
from .common import *


def install(cfg,role):
    if not cfg.get('full_prompt_state_witness'):return
    if role not in cfg.get('full_prompt_state_witness_roles',('prefill','decode','replica0','replica1')):return
    if cfg.get('post_graph_state_witness'):
        from .post_graph_state import install as post_graph
        return post_graph(cfg,role)
    import torch
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    active={};pending={};completed={};folder=ROOT/'raw'/cfg['name']
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        value=prepare(self,*a,**kw);active.update(batch=value,runner=self);return value
    GPUModelRunner.prepare_inputs=prepared
    forward=Qwen3ForCausalLM.forward
    def observed(self,*a,**kw):
        value=forward(self,*a,**kw);batch=active.get('batch')
        if batch is None:return value
        for i,rid in enumerate(batch.req_ids):
            external=rid.split('-')[0]
            if not (external.startswith('q') and external[1:].isdigit()):continue
            if role=='prefill' and cfg.get('dynamic_layer_extension'):
                from .dynamic_extension import REQUEST_FRONTIERS
                if REQUEST_FRONTIERS.get(rid,36)<36:continue
            from .phase_stable_norm import PREFIX_LENGTHS
            length=PREFIX_LENGTHS.get(rid,cfg['length']) if cfg.get('variable_input') else cfg['length']
            start=int(batch.num_computed_tokens_np[i]);size=min(int(batch.num_scheduled_tokens[i]),length-start)
            if size<=0:continue
            row=int(batch.query_start_loc_np[i]);state=pending.setdefault(rid,dict(cursor=0,hidden=hashlib.sha256()))
            assert state['cursor']==start,'Duplicate or missing prompt query'
            state['hidden'].update(value[row:row+size].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            state['cursor']+=size
            if state['cursor']!=length:continue
            runner=active['runner'];table=runner.block_tables.input_block_tables[0][i]
            positions=torch.arange(length,device=value.device);planes=[]
            for layer in range(36):
                attn=runner.vllm_config.compilation_config.static_forward_context[f'model.layers.{layer}.self_attn.attn']
                valid=attn.kv_cache[table[positions//16].long(),:,positions%16,:].detach().contiguous().view(torch.uint8).cpu()
                planes.append(dict(layer=layer,SHA256=hashlib.sha256(valid.numpy().tobytes()).hexdigest(),bytes=valid.numel()))
            completed[external]=dict(KV=planes,final_hidden_SHA256=state['hidden'].hexdigest(),queries=state['cursor'])
        return value
    Qwen3ForCausalLM.forward=observed
    atexit.register(lambda:write(folder/f'full_prompt_state_{role}.json',dict(requests=completed,
        evidence='CORRECTNESS_ONLY',scope='All36 valid prompt KV planes and all8000 final Hidden rows. SHA256 over actual query order. CPU copies perturb timing. No second model forward.')))

