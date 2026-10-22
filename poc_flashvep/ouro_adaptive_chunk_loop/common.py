"""Explicit pins and append-only evidence helpers."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'poc_flashvep/reports/ouro_8gpu_deadline_aware_joint_chunk_loop_discovery'
LEGACY = REPO / 'poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery'
MODEL = Path('/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1')
SELECTION = ROOT/'active_runtime.json'
FORK = Path(os.environ.get('OURO_CG_RUNTIME',
    json.loads(SELECTION.read_text())['path'] if SELECTION.exists()
    else '/home/esjung/external/ouro-deadline-adaptive-rlt'))
PYTHON = '/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python'
SPEC = REPO / 'poc_flashvep/reports/ouro_8gpu_deadline_aware_joint_chunk_loop_poc_spec.md'
UPSTREAM = '4a350344027960a48384626aa646f1765fda6d4f'
UUIDS = dict(enumerate([
 'GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d',
 'GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb',
 'GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e',
 'GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8',
 'GPU-6076e2f2-5b63-3761-5586-56ceb7df8139',
 'GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730',
 'GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28',
 'GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b']))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

def guard():
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert mask in tuple(map(str, range(8))), mask
    return int(mask)

def config(mode, chunk, width=8, blocks=18432, budget=2048, graphs=True, streams=True):
    from vllm_rlt import CacheConfig, SchedulerConfig, ExecutionConfig
    group = {'split': 1, 'mixed': 1, 'grouped': 2, 'grouped3': 3, 'staged4': 4}.get(mode, 1)
    return dict(cache_config=CacheConfig(num_blocks=blocks, block_size=16, layout='last_exited'),
        scheduler_config=SchedulerConfig(max_num_seqs=width, max_num_batched_tokens=budget,
            prefill_chunk_size=chunk, mode='mixed' if mode == 'mixed' else 'refill',
            prefill_loop_group_size=group, max_prefill_batches_before_decode=1),
        execution_config=ExecutionConfig(atomic_prefill=mode=='atomic', cuda_graphs=graphs,
            multi_stream=streams, static_buffers=True, cuda_graph_max_batch_size=width,
            cuda_graph_max_graphs=32), attention_backend='flash_attn_4')
