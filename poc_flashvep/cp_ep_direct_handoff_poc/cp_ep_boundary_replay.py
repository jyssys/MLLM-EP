"""Exact Qwen3-VL CP4-attention to DeepEP-EP4 layer boundary replay.

This forward-only diagnostic uses real BF16 checkpoint weights and a captured
Qwen3-VL activation distribution.  It is intentionally stage-serial so each
same-device CUDA interval has an unambiguous meaning.  It is not reported as
native serving latency.
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
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from cp_attention_replay import (
    a2a_heads_to_sequence,
    a2a_sequence_to_heads,
    apply_text_rope,
    build_local_hidden,
    event,
    flash_attention,
    rms_norm,
)


def load_weight(
    model: Path, index: dict[str, str], key: str, start: int | None = None, end: int | None = None
) -> torch.Tensor:
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as handle:
        if start is None:
            return handle.get_tensor(key).contiguous()
        return handle.get_slice(key)[start:end].contiguous()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--vl-capture", type=Path)
    parser.add_argument("--modality-base", choices=("all", "vision", "text"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--lengths", default="4096,8192,16384,32768")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--buffer-mib", type=int, default=1024)
    args = parser.parse_args()

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4
    args.out.mkdir(parents=True, exist_ok=True)

    import deep_ep

    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        args.buffer_mib * 1024 * 1024,
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )
    comm_stream = buffer.get_comm_stream()

    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    attn = f"model.language_model.layers.{args.layer}.self_attn."
    q_weight = load_weight(args.model, index, attn + "q_proj.weight").cuda()
    k_weight = load_weight(args.model, index, attn + "k_proj.weight").cuda()
    v_weight = load_weight(args.model, index, attn + "v_proj.weight").cuda()
    o_weight = load_weight(args.model, index, attn + "o_proj.weight").cuda()
    q_norm_weight = load_weight(args.model, index, attn + "q_norm.weight").cuda()
    k_norm_weight = load_weight(args.model, index, attn + "k_norm.weight").cuda()
    input_norm = load_weight(
        args.model, index, f"model.language_model.layers.{args.layer}.input_layernorm.weight"
    ).cuda()
    post_norm = load_weight(
        args.model,
        index,
        f"model.language_model.layers.{args.layer}.post_attention_layernorm.weight",
    ).cuda()
    gate = load_weight(
        args.model, index, f"model.language_model.layers.{args.layer}.mlp.gate.weight"
    ).cuda()
    expert = f"model.language_model.layers.{args.layer}.mlp.experts."
    first = rank * 32
    # The Hugging Face checkpoint stores the packed experts as [E, K, N],
    # while vLLM's fused_experts operator consumes [E, N, K].
    w1 = (
        load_weight(args.model, index, expert + "gate_up_proj", first, first + 32)
        .transpose(1, 2)
        .contiguous()
        .cuda()
    )
    w2 = (
        load_weight(args.model, index, expert + "down_proj", first, first + 32)
        .transpose(1, 2)
        .contiguous()
        .cuda()
    )
    expert_map = torch.full((128,), -1, dtype=torch.int32, device="cuda")
    expert_map[first : first + 32] = torch.arange(32, dtype=torch.int32, device="cuda")
    if args.vl_capture is not None:
        capture = torch.load(args.vl_capture, map_location="cpu", weights_only=False)
        base = capture["hidden_diagnostic"][f"layer{args.layer}_input"]
        if args.modality_base != "all":
            mask = capture["vision_mask"].bool()
            base = base[mask if args.modality_base == "vision" else ~mask]
    else:
        capture = torch.load(args.hidden, map_location="cpu", weights_only=True)
        base = capture[f"layer{args.layer}_input"]
    base = base.to(torch.bfloat16).contiguous().cuda()
    if base.shape[0] == 0:
        raise ValueError(f"empty {args.modality_base} activation base")

    rows: list[dict[str, float | int | str]] = []
    route_summaries: dict[int, dict[str, object]] = {}

    def synchronize_world() -> None:
        torch.cuda.synchronize()
        dist.barrier()

    def run_case(length: int, measured: bool, repeat: int) -> torch.Tensor:
        local_tokens = length // world
        local_hidden = build_local_hidden(base, length, rank, world)
        positions = torch.arange(
            rank * local_tokens, (rank + 1) * local_tokens, device="cuda"
        )
        synchronize_world()
        ev = [event() for _ in range(18)]
        ev[0].record()
        normalized = rms_norm(local_hidden, input_norm)
        q = rms_norm(F.linear(normalized, q_weight).view(-1, 32, 128), q_norm_weight)
        k = rms_norm(F.linear(normalized, k_weight).view(-1, 4, 128), k_norm_weight)
        v = F.linear(normalized, v_weight).view(-1, 4, 128)
        q, k = apply_text_rope(q, k, positions)
        ev[1].record()
        q = a2a_sequence_to_heads(q, dist.group.WORLD, world)
        k = a2a_sequence_to_heads(k, dist.group.WORLD, world)
        v = a2a_sequence_to_heads(v, dist.group.WORLD, world)
        ev[2].record()
        output = flash_attention(q, k, v)
        ev[3].record()
        output = a2a_heads_to_sequence(output, dist.group.WORLD, world)
        ev[4].record()
        projected = F.linear(output.reshape(local_tokens, 4096), o_weight)
        ev[5].record()
        moe_input = rms_norm(projected + local_hidden, post_norm)
        ev[6].record()
        logits = F.linear(moe_input, gate)
        probabilities = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probabilities, 8, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()
        ids = ids.to(deep_ep.topk_idx_t).contiguous()
        ev[7].record()
        layout = buffer.get_dispatch_layout(
            ids,
            128,
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        ev[8].record()
        (num_rank, num_rdma, num_expert, in_rank, layout_event) = layout
        (recv_hidden, recv_ids, recv_weights, recv_counts, handle, dispatch_event) = buffer.dispatch(
            x=moe_input,
            handle=None,
            num_tokens_per_rank=num_rank,
            num_tokens_per_rdma_rank=num_rdma,
            is_token_in_rank=in_rank,
            num_tokens_per_expert=num_expert,
            topk_idx=ids,
            topk_weights=weights,
            expert_alignment=1,
            config=deep_ep.Buffer.get_dispatch_config(world),
            previous_event=layout_event,
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        ev[9].record(comm_stream)
        dispatch_event.current_stream_wait()
        ev[10].record()
        # DeepEP encodes received expert ids in this rank's local 0..31
        # namespace.  vLLM's raw fused-expert operator expects logical global
        # ids when an EP map is supplied.
        global_recv_ids = torch.where(
            recv_ids == -1,
            127 if first == 0 else 0,
            recv_ids.to(torch.int64) + first,
        )
        local_output = fused_experts(
            recv_hidden,
            w1,
            w2,
            recv_weights,
            global_recv_ids,
            global_num_experts=128,
            expert_map=expert_map,
        )
        ev[11].record()
        combined, _, combine_event = buffer.combine(
            x=local_output,
            handle=handle,
            topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world),
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        ev[12].record(comm_stream)
        combine_event.current_stream_wait()
        ev[13].record()
        ev[13].synchronize()

        if measured:
            rank_load = torch.bincount((ids.to(torch.int64) // 32).flatten(), minlength=4)
            expert_load = torch.bincount(ids.to(torch.int64).flatten(), minlength=128)
            if repeat == 0:
                destination = ids.to(torch.int64) // 32
                token_fanout = torch.stack(
                    [torch.unique(row).numel() * torch.ones((), device="cuda") for row in destination]
                )
                remote_destinations = torch.stack(
                    [
                        (torch.unique(row) != rank).sum()
                        for row in destination
                    ]
                )
                route_summaries[length] = {
                    "rank_load": rank_load.cpu().tolist(),
                    "active_experts": int((expert_load > 0).sum()),
                    "expert_max": int(expert_load.max()),
                    "expert_mean": float(expert_load.float().mean()),
                    "mean_rank_fanout": float(token_fanout.float().mean()),
                    "mean_remote_destinations": float(remote_destinations.float().mean()),
                    "fraction_all_four_ranks": float((token_fanout == 4).float().mean()),
                }
            rows.append(
                {
                    "rank": rank,
                    "physical_gpu": [4, 5, 6, 7][rank],
                    "layer": args.layer,
                    "length": length,
                    "local_tokens": local_tokens,
                    "repeat": repeat,
                    "qkv_norm_ms": ev[0].elapsed_time(ev[1]),
                    "qkv_a2a_ms": ev[1].elapsed_time(ev[2]),
                    "attention_ms": ev[2].elapsed_time(ev[3]),
                    "return_a2a_ms": ev[3].elapsed_time(ev[4]),
                    "o_proj_ms": ev[4].elapsed_time(ev[5]),
                    "residual_norm_ms": ev[5].elapsed_time(ev[6]),
                    "router_topk_ms": ev[6].elapsed_time(ev[7]),
                    "route_layout_enqueue_ms": ev[7].elapsed_time(ev[8]),
                    "dispatch_critical_ms": ev[8].elapsed_time(ev[10]),
                    "expert_ms": ev[10].elapsed_time(ev[11]),
                    "combine_critical_ms": ev[11].elapsed_time(ev[13]),
                    "attention_path_ms": ev[0].elapsed_time(ev[5]),
                    "cp_to_first_expert_ms": ev[3].elapsed_time(ev[10]),
                    "moe_path_ms": ev[5].elapsed_time(ev[13]),
                    "layer_path_ms": ev[0].elapsed_time(ev[13]),
                    "output_norm": float(combined.float().norm()),
                }
            )
        return combined

    lengths = [int(item) for item in args.lengths.split(",")]
    for length in lengths:
        for warmup_index in range(args.warmup):
            run_case(length, False, -args.warmup + warmup_index)

    schedule = [(length, repeat) for repeat in range(args.reps) for length in lengths]
    random.Random(13091).shuffle(schedule)
    object_list = [schedule if rank == 0 else None]
    dist.broadcast_object_list(object_list, src=0)
    schedule = object_list[0]
    started = time.time()
    for length, repeat in schedule:
        run_case(length, True, repeat)

    result = {
        "scope": "EXACT_QWEN3_VL_CP4_DEEPEP_EP4_LAYER_REPLAY_NOT_SERVING",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rank": rank,
        "physical_gpu": [4, 5, 6, 7][rank],
        "layer": args.layer,
        "warmup": args.warmup,
        "reps": args.reps,
        "duration_sec": time.time() - started,
        "checkpoint": str(args.model),
        "hidden_capture": str(args.hidden),
        "vl_capture": str(args.vl_capture) if args.vl_capture else None,
        "modality_base": args.modality_base,
        "base_tokens": int(base.shape[0]),
        "deep_ep": str(Path(deep_ep.__file__).resolve()),
        "deep_ep_sm90": bool(deep_ep.Buffer.is_sm90_compiled()),
        "route_summaries": route_summaries,
        "rows": rows,
    }
    (args.out / f"cp_ep_boundary_rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    synchronize_world()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
