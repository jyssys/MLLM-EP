"""Exact Qwen3-VL attention context-parallel replay.

This is a forward-only layer diagnostic, not a serving implementation.  It
uses a real Qwen3-VL hidden-state capture and the checkpoint's BF16 attention
weights.  CP is Ulysses-style: sequence-sharded QKV is transposed to
head-sharded full-sequence attention with an All-to-All, followed by the
inverse All-to-All.  Only ranks ``[0, cp_size)`` own the one measured request;
the remaining EP-world ranks stay idle so that the request's token work is
constant across CP1/2/4.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from vllm.vllm_flash_attn import flash_attn_varlen_func


def load_weight(model: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + 1e-6)).to(x.dtype) * weight


def apply_text_rope(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float = 5_000_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL text RoPE for controlled monotonically indexed tokens.

    With identical temporal/height/width position ids, Qwen3-VL's interleaved
    MRoPE reduces exactly to this ordinary rotary embedding.  The controlled
    long-context replay deliberately uses this text-position contract rather
    than inventing image-grid coordinates for repeated captured activations.
    """
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, q.shape[-1], 2, device=q.device, dtype=torch.float32)
            / q.shape[-1]
        )
    )
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(q.dtype).unsqueeze(1)
    sin = emb.sin().to(q.dtype).unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_cu = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
    k_cu = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=k.device)
    return flash_attn_varlen_func(
        q,
        k,
        v,
        q.shape[0],
        q_cu,
        k.shape[0],
        k_cu,
        causal=True,
        softmax_scale=128**-0.5,
        fa_version=3,
    )


def a2a_sequence_to_heads(x: torch.Tensor, group, cp_size: int) -> torch.Tensor:
    # [S/P, heads, d] -> [S, heads/P, d]
    local_s, heads, dim = x.shape
    assert heads % cp_size == 0
    per_rank_heads = heads // cp_size
    send = (
        x.view(local_s, cp_size, per_rank_heads, dim)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(-1)
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.view(cp_size, local_s, per_rank_heads, dim).reshape(
        cp_size * local_s, per_rank_heads, dim
    )


def a2a_heads_to_sequence(x: torch.Tensor, group, cp_size: int) -> torch.Tensor:
    # [S, heads/P, d] -> [S/P, heads, d]
    total_s, per_rank_heads, dim = x.shape
    assert total_s % cp_size == 0
    local_s = total_s // cp_size
    send = x.view(cp_size, local_s, per_rank_heads, dim).contiguous().view(-1)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.view(cp_size, local_s, per_rank_heads, dim)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(local_s, cp_size * per_rank_heads, dim)
    )


def exact_block_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block: int
) -> torch.Tensor:
    """Exact causal query blocking used by the streaming candidate."""
    pieces = []
    for start in range(0, q.shape[0], block):
        end = min(start + block, q.shape[0])
        # Bottom-right causal alignment maps these queries to positions
        # [start, end), so K/V prefix [:end] is exact.
        pieces.append(flash_attention(q[start:end], k[:end], v[:end]))
    return torch.cat(pieces, dim=0)


def build_local_hidden(base: torch.Tensor, length: int, cp_rank: int, cp: int) -> torch.Tensor:
    assert length % cp == 0
    local = length // cp
    indexes = torch.arange(cp_rank * local, (cp_rank + 1) * local)
    # Preserve the empirical activation distribution while constructing a
    # controlled long-context shape.  No random tensor replaces the capture.
    return base[indexes.remainder(base.shape[0])].contiguous()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--lengths", default="4096,8192,16384,32768,65536")
    parser.add_argument("--cp", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--block-correctness", type=int, default=256)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    args.out.mkdir(parents=True, exist_ok=True)

    cp_groups = {}
    for degree in (1, 2, 4):
        cp_groups[degree] = dist.new_group(list(range(degree)), backend="nccl")

    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{args.layer}.self_attn."
    q_weight = load_weight(args.model, index, prefix + "q_proj.weight").cuda()
    k_weight = load_weight(args.model, index, prefix + "k_proj.weight").cuda()
    v_weight = load_weight(args.model, index, prefix + "v_proj.weight").cuda()
    o_weight = load_weight(args.model, index, prefix + "o_proj.weight").cuda()
    q_norm_weight = load_weight(args.model, index, prefix + "q_norm.weight").cuda()
    k_norm_weight = load_weight(args.model, index, prefix + "k_norm.weight").cuda()
    input_norm = load_weight(
        args.model,
        index,
        f"model.language_model.layers.{args.layer}.input_layernorm.weight",
    ).cuda()
    capture = torch.load(args.hidden, map_location="cpu", weights_only=True)
    base = capture[f"layer{args.layer}_input"].to(torch.bfloat16).contiguous().cuda()

    lengths = [int(item) for item in args.lengths.split(",")]
    degrees = [int(item) for item in args.cp.split(",")]
    cases = [(length, degree) for length in lengths for degree in degrees]
    rows = []
    outputs = {}

    def run_case(length: int, cp: int, measured: bool, repeat: int) -> None:
        active = rank < cp
        dist.barrier()
        if active:
            local_hidden = build_local_hidden(base, length, rank, cp)
            ev = [event() for _ in range(8)]
            ev[0].record()
            normalized = rms_norm(local_hidden, input_norm)
            q = rms_norm(
                F.linear(normalized, q_weight).view(-1, 32, 128), q_norm_weight
            )
            k = rms_norm(
                F.linear(normalized, k_weight).view(-1, 4, 128), k_norm_weight
            )
            v = F.linear(normalized, v_weight).view(-1, 4, 128)
            local_tokens = length // cp
            positions = torch.arange(
                rank * local_tokens,
                (rank + 1) * local_tokens,
                device=q.device,
            )
            q, k = apply_text_rope(q, k, positions)
            ev[1].record()
            if cp > 1:
                q = a2a_sequence_to_heads(q, cp_groups[cp], cp)
                k = a2a_sequence_to_heads(k, cp_groups[cp], cp)
                v = a2a_sequence_to_heads(v, cp_groups[cp], cp)
            ev[2].record()
            attention = flash_attention(q, k, v)
            ev[3].record()
            if cp > 1:
                attention = a2a_heads_to_sequence(attention, cp_groups[cp], cp)
            ev[4].record()
            projected = F.linear(attention.reshape(-1, 4096), o_weight)
            ev[5].record()
            # Validate that query-blocking creates exact-ready pieces rather
            # than an approximate attention result.  This check is run at a
            # bounded length to avoid perturbing the timing grid.
            if length == min(lengths) and repeat == -args.warmup and cp == 1:
                blocked = exact_block_attention(q, k, v, args.block_correctness)
                ev[6].record()
                delta = blocked.float() - attention.float()
                outputs["block_correctness"] = {
                    "cosine": float(
                        F.cosine_similarity(blocked.flatten().float(), attention.flatten().float(), dim=0)
                    ),
                    "relative_l2": float(delta.norm() / attention.float().norm().clamp_min(1e-12)),
                    "max_abs": float(delta.abs().max()),
                }
            else:
                ev[6].record()
            ev[7].record()
            ev[7].synchronize()
            if measured:
                rows.append(
                    {
                        "rank": rank,
                        "physical_gpu": [4, 5, 6, 7][rank],
                        "layer": args.layer,
                        "length": length,
                        "cp": cp,
                        "repeat": repeat,
                        "local_tokens": local_hidden.shape[0],
                        "qkv_norm_ms": ev[0].elapsed_time(ev[1]),
                        "qkv_a2a_ms": ev[1].elapsed_time(ev[2]),
                        "attention_ms": ev[2].elapsed_time(ev[3]),
                        "return_a2a_ms": ev[3].elapsed_time(ev[4]),
                        "o_proj_ms": ev[4].elapsed_time(ev[5]),
                        "attention_path_ms": ev[0].elapsed_time(ev[5]),
                        "output_norm": float(projected.float().norm()),
                    }
                )
        torch.cuda.synchronize()
        dist.barrier()

    # Global and per-shape warmup.
    for length, cp in cases:
        for warmup_index in range(args.warmup):
            run_case(length, cp, False, -args.warmup + warmup_index)

    schedule = [(length, cp, rep) for rep in range(args.reps) for length, cp in cases]
    rng = random.Random(91731)
    rng.shuffle(schedule)
    schedule_object = [schedule if rank == 0 else None]
    dist.broadcast_object_list(schedule_object, src=0)
    schedule = schedule_object[0]
    started = time.time()
    for length, cp, rep in schedule:
        run_case(length, cp, True, rep)

    result = {
        "scope": "EXACT_QWEN3_VL_LAYER_REPLAY_NOT_SERVING",
        "rank": rank,
        "physical_gpu": [4, 5, 6, 7][rank],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "layer": args.layer,
        "warmup": args.warmup,
        "reps": args.reps,
        "duration_sec": time.time() - started,
        "checkpoint": str(args.model),
        "hidden_capture": str(args.hidden),
        "block_correctness": outputs.get("block_correctness"),
        "rows": rows,
    }
    (args.out / f"cp_attention_rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
