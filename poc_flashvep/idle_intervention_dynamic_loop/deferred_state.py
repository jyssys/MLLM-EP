"""Hash immutable prompt planes before release. Preserve the first Coda hidden."""
import hashlib
import os
from .common import ROOT


def install(cfg):
    import torch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.request import Operation
    from ..looped_pd_headroom_recovery.state_probe import plane
    from ..looped_pd_abc.common import append
    initialize,execute,free=StageEngine.__init__,ModelRunner._execute_batch,KVCacheManager.free

    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        engine.cache_manager._idle_probe_runner=engine.model_runner
        engine.model_runner._idle_prompt_snapshots={}
        return result

    def captured(runner,batch,prepared):
        if batch.operation==Operation.CODA:
            hidden=prepared.forwards[0].forward.hidden;offset=0
            for span in batch.spans:
                request=span.request;n=len(request.prompt_token_ids)
                if span.token_start==n-1 and not request.generated_token_ids:
                    assert request.request_id not in runner._idle_prompt_snapshots
                    value=hidden[offset].detach().clone()
                    event=torch.cuda.Event();event.record(torch.cuda.current_stream(runner.device))
                    runner._idle_prompt_snapshots[request.request_id]=(request,value,event)
                offset+=span.token_count
        return execute(runner,batch,prepared)

    def sha(tensor):
        return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def released(cache,request_id):
        runner=getattr(cache,'_idle_probe_runner',None)
        pending=runner._idle_prompt_snapshots if runner is not None else {}
        allocation=cache._allocations.get(request_id)
        if request_id in pending and allocation is not None and not allocation.transfer_leases:
            request,hidden,event=pending.pop(request_id);event.synchronize()
            n=len(request.prompt_token_ids)
            recurrent=set(getattr(runner.model,'recurrent_kv_layers',range(cache.num_layers)))
            coda=set(getattr(runner.model,'coda_kv_layers',()))
            planes=[]
            for depth,table in enumerate(allocation.block_tables):
                blocks=torch.tensor(table[:(n+cache.block_size-1)//cache.block_size],device=cache.device)
                for layer in range(cache.num_layers):
                    if depth and layer not in recurrent:continue
                    valid=n-1 if layer in coda else n
                    assert allocation.written[depth][layer].prefix>=valid
                    for kind,tensor in (('K',cache.key_cache),('V',cache.value_cache)):
                        value=plane(tensor,blocks,layer).flatten(0,1)[:valid]
                        planes.append(dict(depth=depth,layer=layer,kind=kind,valid_tokens=valid,sha256=sha(value)))
            append(ROOT/'raw'/cfg['name']/f'prompt_states_{os.getpid()}.jsonl',
                dict(seed=request.sampling_params.seed,cut=getattr(request,'arc_received_cut',None),
                    prompt_tokens=n,hidden_sha256=sha(hidden),planes=planes,
                    evidence='MEASURED_HEAVY_DEFERRED_CORRECTNESS',
                    scope='First-Coda hidden clone. Read-only prompt KV hashes immediately before native free, after Decode. Same valid prompt bytes. No allocation releases before hashing. This run is not clean timing. Handoff fingerprint is covered separately.'))
        return free(cache,request_id)
    StageEngine.__init__,ModelRunner._execute_batch,KVCacheManager.free=initialized,captured,released
