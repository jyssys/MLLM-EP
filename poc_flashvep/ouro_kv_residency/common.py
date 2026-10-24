"""Pinned inputs and append-only evidence helpers."""
from pathlib import Path
import os
from ..ouro_adaptive_chunk_loop.common import REPO, MODEL, PYTHON, UUIDS, UPSTREAM, write, append, digest

ROOT = Path(os.environ.get('OURO_KV_OUTPUT',str(REPO/'poc_flashvep/reports/ouro_loop_aware_kv_residency_distributed_attention_discovery')))
PRIOR = REPO/'poc_flashvep/reports/ouro_8gpu_elastic_loop_batching_decode_fusion_discovery'
FORK = Path('/home/esjung/external/ouro-kv-residency-rlt')
FIXTURES = Path(os.environ.get('OURO_KV_FIXTURES','/home/esjung/.cache/ouro-kv-residency-fixtures'))

def config(n, blocks=18432, graphs=True):
    from vllm_rlt import CacheConfig, SchedulerConfig, ExecutionConfig, ExitConfig
    return dict(cache_config=CacheConfig(num_blocks=blocks,block_size=16,layout='last_exited'),
        scheduler_config=SchedulerConfig(max_num_seqs=n,max_num_batched_tokens=2048,
            prefill_chunk_size=512,mode='mixed'),
        execution_config=ExecutionConfig(atomic_prefill=False,async_scheduling=True,
            static_buffers=True,multi_stream=True,cuda_graphs=graphs,
            cuda_graph_max_batch_size=n,cuda_graph_max_graphs=32),
        exit_config=ExitConfig(mode='ouro_delayed'),attention_backend='flash_attn_4')
