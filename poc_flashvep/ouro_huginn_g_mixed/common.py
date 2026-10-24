"""Immutable model inputs and evidence paths."""
from pathlib import Path
from ..ouro_adaptive_chunk_loop.common import REPO, PYTHON, UUIDS, UPSTREAM, write, append, digest

ROOT = REPO / 'poc_flashvep/reports/ouro_huginn_adaptive_g_mixed_prefill_discovery'
PRIOR = REPO / 'poc_flashvep/reports/ouro_loop_aware_kv_residency_distributed_attention_discovery'
FORK = Path('/home/esjung/external/ouro-huginn-g-mixed-rlt')
SPEC = REPO / 'poc_flashvep/reports/ouro_huginn_adaptive_g_mixed_prefill_poc_spec.md'
MODELS = {
    'ouro': Path('/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1'),
    'huginn': Path('/home/esjung/.cache/huggingface/hub/models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c'),
}
DEPTHS = {'ouro': 4, 'huginn': 16}
GROUPS = {'ouro': (1, 2, 3, 4), 'huginn': (1, 2, 4, 8, 16)}

def configuration(model, arm, group, chunk, n, graphs=True, asynchronous=None):
    from vllm_rlt import CacheConfig, SchedulerConfig, ExecutionConfig, ExitConfig
    async_mode = model == 'ouro' if asynchronous is None else asynchronous
    assert model != 'huginn' or not async_mode
    return dict(cache_config=CacheConfig(num_blocks=12288 if model == 'ouro' else 12288,
                    block_size=16, layout='last_exited'),
        scheduler_config=SchedulerConfig(max_num_seqs=max(n, 2),
            max_num_batched_tokens=2048, prefill_chunk_size=chunk,
            prefill_loop_group_size=group,
            mode='mixed' if arm in ('mixed', 'fair', 'hybrid', 'event') else 'refill'),
        execution_config=ExecutionConfig(atomic_prefill=arm == 'atomic',
            async_scheduling=async_mode, static_buffers=True, multi_stream=True,
            cuda_graphs=graphs, cuda_graph_max_batch_size=max(n, 2), cuda_graph_max_graphs=32),
        exit_config=ExitConfig(mode='ouro_delayed' if async_mode else 'ouro'),
        attention_backend='flash_attn_4')
