"""Reuse the exact static extension until dynamic qualification is enabled."""
from .common import ROOT
from ..dense_ar_layerwise_pd.extension import tensor_hash


def install(cfg,role,notify=None):
    import vllm,torch
    from .common import write
    from vllm.v1.attention.backends.utils import get_kv_cache_layout
    original_llm=vllm.LLM
    def audited_llm(*args,**kwargs):
        llm=original_llm(*args,**kwargs)
        runner=llm.llm_engine.model_executor.driver_worker.worker.model_runner
        write(ROOT/'raw'/cfg['name']/f'effective_runtime_{role}.json',dict(
            KV_layout=get_kv_cache_layout(),KV_tensors=[dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype)) for t in runner.kv_caches],
            actual_scheduler_max_num_batched_tokens=runner.vllm_config.scheduler_config.max_num_batched_tokens,
            actual_scheduler_max_num_seqs=runner.vllm_config.scheduler_config.max_num_seqs,
            hidden_transport=cfg.get('hidden_transport'),torch_allocated_bytes_before_optional_hidden_pool=torch.cuda.memory_allocated(),
            scope='Actual constructed model and KV storage. This startup observation is outside the Serving measurement.'))
        return llm
    vllm.LLM=audited_llm
    if cfg.get('dynamic_layer_extension'):
        from .dynamic_extension import install as dynamic_install
        result=dynamic_install(cfg,role,notify)
        from .boundaries import install as boundaries
        boundaries(cfg,role)
        return result
    from ..dense_ar_layerwise_pd import extension
    extension.ROOT=ROOT
    result=extension.install(cfg,role,notify)
    from .boundaries import install as boundaries
    boundaries(cfg,role)
    return result
