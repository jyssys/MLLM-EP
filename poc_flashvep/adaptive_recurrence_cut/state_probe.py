"""Diagnostic-only byte hashes after full prompt computation, before first CODA."""
import hashlib
import os


def install(cfg):
    if not cfg.get('state_probe'):return
    import torch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.request import Operation
    from .common import ROOT,append
    if getattr(ModelRunner,'_arc_state_probe',False):return
    ModelRunner._arc_state_probe=True
    original=ModelRunner._execute_batch

    def sha(tensor):
        raw=tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def execute(self,batch,prepared):
        if batch.operation==Operation.CODA:
            hidden=prepared.forwards[0].forward.hidden
            offset=0
            for span in batch.spans:
                request=span.request;n=len(request.prompt_token_ids)
                if span.token_start==n-1:
                    cache=self.cache_manager;allocation=cache._get_allocation(request.request_id)
                    recurrent=set(getattr(self.model,'recurrent_kv_layers',range(cache.num_layers)))
                    coda=set(getattr(self.model,'coda_kv_layers',()))
                    planes=[]
                    for depth,table in enumerate(allocation.block_tables):
                        blocks=torch.tensor(table[:(n+cache.block_size-1)//cache.block_size],device=cache.device)
                        for layer in range(cache.num_layers):
                            if depth and layer not in recurrent:continue
                            valid=n-1 if layer in coda else n
                            for kind,tensor in (('K',cache.key_cache),('V',cache.value_cache)):
                                value=tensor.index_select(0,blocks)[:,layer].flatten(0,1)[:valid]
                                planes.append(dict(depth=depth,layer=layer,kind=kind,valid_tokens=valid,sha256=sha(value)))
                    append(ROOT/'raw'/os.environ['ARC_RUN']/f'prompt_states_{os.getpid()}.jsonl',
                           dict(seed=request.sampling_params.seed,cut=getattr(request,'arc_received_cut',None),
                                prompt_tokens=n,hidden_sha256=sha(hidden[offset]),planes=planes,
                                evidence='MEASURED_HEAVY_CORRECTNESS',scope='Before first CODA. Last prompt coda KV is not valid yet.'))
                offset+=span.token_count
        return original(self,batch,prepared)

    ModelRunner._execute_batch=execute
