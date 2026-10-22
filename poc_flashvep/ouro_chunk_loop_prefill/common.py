"""Pinned configuration and append-only experiment records."""

import json
import os
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/ouro_chunk_loop_prefill_joint_discovery"
)
MODEL = Path(
    "/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
)
UPSTREAM = "4a350344027960a48384626aa646f1765fda6d4f"
FORK = Path("/home/esjung/external/ouro-chunk-loop-prefill-rlt")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, default=str) + "\n")


def guard():
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert mask in ("4", "5", "6", "7"), mask
    return mask


def configuration(mode, chunk, concurrency=8, graphs=True, streams=True, blocks=4096):
    from vllm_rlt import CacheConfig, SchedulerConfig, ExecutionConfig

    return dict(
        cache_config=CacheConfig(
            num_blocks=blocks, block_size=16, layout="last_exited"
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency,
            max_num_batched_tokens=1024,
            prefill_chunk_size=chunk,
            mode="mixed" if mode == "mixed" else "refill",
            prefill_loop_group_size=2
            if mode == "grouped"
            else 4
            if mode == "staged4"
            else 1,
            max_prefill_batches_before_decode=1,
        ),
        execution_config=ExecutionConfig(
            atomic_prefill=mode == "atomic",
            cuda_graphs=graphs,
            multi_stream=streams,
            static_buffers=True,
            cuda_graph_max_batch_size=concurrency,
            cuda_graph_max_graphs=32,
        ),
        attention_backend=os.environ.get("OURO_PREFILL_BACKEND", "triton"),
    )
