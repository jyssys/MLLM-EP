"""Read the runner's actual hidden output after eager or graph execution."""
import atexit
from .common import *


def install(cfg,role):
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original=GPUModelRunner.execute_model;pending={};completed={}
    def observed(self,*a,**kw):
        value=original(self,*a,**kw)
        if kw.get('dummy_run') or torch.cuda.is_current_stream_capturing():return value
        frame=self.execute_model_state
        if frame is None or frame.hidden_states is None:return value
        batch=frame.input_batch;hidden=frame.hidden_states
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
            assert state['cursor']==start,'Duplicate or missing post-graph prompt query'
            state['hidden'].update(hidden[row:row+size].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            state['cursor']+=size
            if state['cursor']!=length:continue
            table=self.block_tables.input_block_tables[0][i];positions=torch.arange(length,device=hidden.device);planes=[]
            for layer in range(36):
                attn=self.vllm_config.compilation_config.static_forward_context[f'model.layers.{layer}.self_attn.attn']
                valid=attn.kv_cache[table[positions//16].long(),:,positions%16,:].detach().contiguous().view(torch.uint8).cpu()
                planes.append(dict(layer=layer,SHA256=hashlib.sha256(valid.numpy().tobytes()).hexdigest(),bytes=valid.numel()))
            completed[external]=dict(KV=planes,final_hidden_SHA256=state['hidden'].hexdigest(),queries=state['cursor'])
        return value
    GPUModelRunner.execute_model=observed
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/f'full_prompt_state_{role}.json',dict(requests=completed,
        evidence='CORRECTNESS_ONLY',scope='Read the actual runner output after its model call or graph replay. Hash all valid prompt KV and final Hidden rows. CPU copies perturb timing. No second forward.')))
