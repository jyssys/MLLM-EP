"""Pinned paths and immutable evidence helpers."""
import os
from pathlib import Path
from ..ouro_adaptive_chunk_loop.common import REPO, MODEL, PYTHON, UUIDS, UPSTREAM, write, append, digest

ROOT = REPO/'poc_flashvep/reports/ouro_8gpu_elastic_loop_batching_decode_fusion_discovery'
PRIOR = REPO/'poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_discovery'
FORK = Path('/home/esjung/external/ouro-elb-rlt')
GPUS = tuple(range(8))

def guard():
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert value in tuple(map(str, GPUS)), value
    return int(value)

def config(n=8, group=1, graphs=True, asynchronous=True, mode='refill', blocks=18432):
    from vllm_rlt import CacheConfig, SchedulerConfig, ExecutionConfig, ExitConfig
    return dict(cache_config=CacheConfig(num_blocks=blocks,block_size=16,layout='last_exited'),
        scheduler_config=SchedulerConfig(max_num_seqs=n,max_num_batched_tokens=2048,
            prefill_chunk_size=512,mode=mode,decode_loop_group_size=group),
        execution_config=ExecutionConfig(atomic_prefill=False,async_scheduling=asynchronous,
            static_buffers=True,multi_stream=True,cuda_graphs=graphs,
            cuda_graph_max_batch_size=n,cuda_graph_max_graphs=32),
        exit_config=ExitConfig(mode='ouro_delayed' if asynchronous else 'ouro'),attention_backend='flash_attn_4')
