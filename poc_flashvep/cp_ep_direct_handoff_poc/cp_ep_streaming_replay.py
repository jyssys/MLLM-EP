"""Exact block-streaming CP4-attention to DeepEP-EP4 layer replay.

Each streaming step computes the next exact causal query block for every CP
sequence owner, returns the four head shards with an exact NCCL All-to-All,
then routes that block through real DeepEP and real Qwen3-VL expert weights.
The candidate overlaps only already-exact next-block attention with current
block EP work; it never changes routes, drops tokens, or approximates values.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
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


@dataclass
class ReadyBlock:
    offset: int
    count: int
    sent_heads: torch.Tensor
    received_heads: torch.Tensor
    ready: torch.cuda.Event
    work: object


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--lengths", default="8192,16384,32768")
    parser.add_argument("--blocks", default="64,128,256,512,1024")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--buffer-mib", type=int, default=1024)
    parser.add_argument(
        "--clean-timing",
        action="store_true",
        help=(
            "Record only invocation-level timing. This removes per-block stage "
            "events and host synchronizations so the streaming result is not an "
            "observer-tax artifact."
        ),
    )
    args = parser.parse_args()

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4
    cp_group = dist.new_group(list(range(world)), backend="nccl")
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
    attention_stream = torch.cuda.Stream()
    cp_stream = torch.cuda.Stream()

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
    w1 = (
        load_weight(args.model, index, expert + "gate_up_proj", first, first + 32)
        .transpose(1, 2).contiguous().cuda()
    )
    w2 = (
        load_weight(args.model, index, expert + "down_proj", first, first + 32)
        .transpose(1, 2).contiguous().cuda()
    )
    expert_map = torch.full((128,), -1, dtype=torch.int32, device="cuda")
    expert_map[first : first + 32] = torch.arange(32, dtype=torch.int32, device="cuda")
    capture = torch.load(args.hidden, map_location="cpu", weights_only=True)
    base = capture[f"layer{args.layer}_input"].to(torch.bfloat16).contiguous().cuda()

    rows: list[dict[str, object]] = []
    correctness: list[dict[str, object]] = []
    references: dict[int, dict[str, torch.Tensor]] = {}

    def world_sync() -> None:
        torch.cuda.synchronize()
        dist.barrier()

    def make_qkv(length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]:
        local_tokens = length // world
        local_hidden = build_local_hidden(base, length, rank, world)
        positions = torch.arange(rank * local_tokens, (rank + 1) * local_tokens, device="cuda")
        e0, e1, e2 = event(), event(), event()
        e0.record()
        normalized = rms_norm(local_hidden, input_norm)
        q = rms_norm(F.linear(normalized, q_weight).view(-1, 32, 128), q_norm_weight)
        k = rms_norm(F.linear(normalized, k_weight).view(-1, 4, 128), k_norm_weight)
        v = F.linear(normalized, v_weight).view(-1, 4, 128)
        q, k = apply_text_rope(q, k, positions)
        e1.record()
        q = a2a_sequence_to_heads(q, cp_group, world)
        k = a2a_sequence_to_heads(k, cp_group, world)
        v = a2a_sequence_to_heads(v, cp_group, world)
        e2.record()
        return local_hidden, q, k, v, e0, e1, e2

    def route_and_ep(moe_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.cuda.Event]]:
        ev = [] if args.clean_timing else [event() for _ in range(8)]
        if ev:
            ev[0].record()
        logits = F.linear(moe_input, gate)
        probabilities = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probabilities, 8, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()
        ids = ids.to(deep_ep.topk_idx_t).contiguous()
        if ev:
            ev[1].record()
        layout = buffer.get_dispatch_layout(
            ids, 128, async_finish=True, allocate_on_comm_stream=False
        )
        if ev:
            ev[2].record()
        num_rank, num_rdma, num_expert, in_rank, layout_event = layout
        recv_hidden, recv_ids, recv_weights, _, handle, dispatch_event = buffer.dispatch(
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
        dispatch_event.current_stream_wait()
        if ev:
            ev[3].record()
        global_recv_ids = torch.where(
            recv_ids == -1, 127 if first == 0 else 0, recv_ids.to(torch.int64) + first
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
        if ev:
            ev[4].record()
        combined, _, combine_event = buffer.combine(
            x=local_output,
            handle=handle,
            topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world),
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        combine_event.current_stream_wait()
        if ev:
            ev[5].record()
        return combined, ids, weights, ev

    def launch_ready_block(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        local_tokens: int,
        offset: int,
        block: int,
    ) -> ReadyBlock:
        count = min(block, local_tokens - offset)
        compute_done = event()
        with torch.cuda.stream(attention_stream):
            pieces = []
            for owner in range(world):
                start = owner * local_tokens + offset
                end = start + count
                pieces.append(flash_attention(q[start:end], k[:end], v[:end]))
            send = torch.stack(pieces, dim=0).contiguous()
            compute_done.record(attention_stream)
        received = torch.empty_like(send)
        ready = event()
        with torch.cuda.stream(cp_stream):
            cp_stream.wait_event(compute_done)
            work = dist.all_to_all_single(received, send, group=cp_group, async_op=True)
            ready.record(cp_stream)
        return ReadyBlock(offset, count, send, received, ready, work)

    def unpack_block(block: ReadyBlock) -> torch.Tensor:
        # The CUDA event is kept for same-device timing/order, while Work.wait
        # is the NCCL completion contract.  Both are required when another
        # communicator (DeepEP) is concurrently active.
        block.work.wait()
        torch.cuda.current_stream().wait_event(block.ready)
        # [source head-rank, tokens, heads/source, d] -> [tokens, all heads, d]
        return block.received_heads.permute(1, 0, 2, 3).contiguous().view(block.count, 32, 128)

    def run_serial(length: int) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        local_tokens = length // world
        local_hidden, q, k, v, e0, e1, e2 = make_qkv(length)
        ea, er, eo, en, end = event(), event(), event(), event(), event()
        ea.record()
        attention = flash_attention(q, k, v)
        er.record()
        attention = a2a_heads_to_sequence(attention, cp_group, world)
        eo.record()
        projected = F.linear(attention.reshape(local_tokens, 4096), o_weight)
        en.record()
        moe_input = rms_norm(projected + local_hidden, post_norm)
        combined, ids, _, mev = route_and_ep(moe_input)
        end.record(); end.synchronize()
        timing = {
            "wall_ms": e0.elapsed_time(end),
            "qkv_norm_ms": e0.elapsed_time(e1),
            "qkv_a2a_ms": e1.elapsed_time(e2),
        }
        if mev:
            timing.update({
                "attention_ms": ea.elapsed_time(er),
                "return_a2a_ms": er.elapsed_time(eo),
                "o_proj_norm_ms": eo.elapsed_time(mev[0]),
                "router_ms": mev[0].elapsed_time(mev[1]),
                "dispatch_ms": mev[1].elapsed_time(mev[3]),
                "expert_ms": mev[3].elapsed_time(mev[4]),
                "combine_ms": mev[4].elapsed_time(mev[5]),
            })
        return timing, combined, ids

    def run_chunked(length: int, block: int, overlap: bool) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        local_tokens = length // world
        local_hidden, q, k, v, e0, e1, e2 = make_qkv(length)
        # Q/K/V are produced and redistributed on the default stream.  The
        # speculative attention stream must not read them before that exact
        # CP input A2A completes.
        attention_stream.wait_event(e2)
        offsets = list(range(0, local_tokens, block))
        outputs: list[torch.Tensor] = []
        routes: list[torch.Tensor] = []
        end = event()
        stage = {"router_ms": 0.0, "dispatch_ms": 0.0, "expert_ms": 0.0, "combine_ms": 0.0}
        if overlap:
            current = launch_ready_block(q, k, v, local_tokens, offsets[0], block)
            for index, offset in enumerate(offsets):
                attention = unpack_block(current)
                projected = F.linear(attention.reshape(current.count, 4096), o_weight)
                moe_input = rms_norm(
                    projected + local_hidden[offset : offset + current.count], post_norm
                )
                # Maintain exactly one attention block of lookahead.  Its
                # attention and CP return can overlap current DeepEP work.
                following = (
                    launch_ready_block(q, k, v, local_tokens, offsets[index + 1], block)
                    if index + 1 < len(offsets)
                    else None
                )
                combined, ids, _, mev = route_and_ep(moe_input)
                if mev:
                    mev[5].synchronize()
                    for key, a, b in (
                        ("router_ms", 0, 1), ("dispatch_ms", 1, 3),
                        ("expert_ms", 3, 4), ("combine_ms", 4, 5),
                    ):
                        stage[key] += mev[a].elapsed_time(mev[b])
                # DeepEP/vLLM may recycle communication and fused-MoE
                # workspaces on the next micro-invocation.  Preserve each
                # exact token block before launching the following one.
                outputs.append(combined.clone()); routes.append(ids.clone())
                current = following
        else:
            for offset in offsets:
                current = launch_ready_block(q, k, v, local_tokens, offset, block)
                attention = unpack_block(current)
                projected = F.linear(attention.reshape(current.count, 4096), o_weight)
                moe_input = rms_norm(
                    projected + local_hidden[offset : offset + current.count], post_norm
                )
                combined, ids, _, mev = route_and_ep(moe_input)
                if mev:
                    mev[5].synchronize()
                    for key, a, b in (
                        ("router_ms", 0, 1), ("dispatch_ms", 1, 3),
                        ("expert_ms", 3, 4), ("combine_ms", 4, 5),
                    ):
                        stage[key] += mev[a].elapsed_time(mev[b])
                outputs.append(combined.clone()); routes.append(ids.clone())
                if args.clean_timing:
                    # Preserve the no-overlap control without synchronizing the
                    # host or recording every internal stage.
                    block_done = event()
                    block_done.record()
                    attention_stream.wait_event(block_done)
        end.record(); end.synchronize()
        return {
            "wall_ms": e0.elapsed_time(end),
            "qkv_norm_ms": e0.elapsed_time(e1),
            "qkv_a2a_ms": e1.elapsed_time(e2),
            **stage,
        }, torch.cat(outputs), torch.cat(routes)

    lengths = [int(x) for x in args.lengths.split(",")]
    blocks = [int(x) for x in args.blocks.split(",")]
    cases = [(n, "serial", 0) for n in lengths]
    cases += [(n, variant, b) for n in lengths for b in blocks for variant in ("chunk_no_overlap", "streaming") if n // world >= b]

    # One correctness/reference pass per length before timed randomization.
    for length in lengths:
        world_sync()
        _, output, ids = run_serial(length)
        references[length] = {"output": output.detach().clone(), "ids": ids.detach().clone()}
        for block in blocks:
            if length // world < block:
                continue
            for variant, overlap in (("chunk_no_overlap", False), ("streaming", True)):
                world_sync()
                _, candidate, candidate_ids = run_chunked(length, block, overlap)
                reference = references[length]
                delta = candidate.float() - reference["output"].float()
                correctness.append({
                    "rank": rank,
                    "length": length,
                    "variant": variant,
                    "block": block,
                    "cosine": float(F.cosine_similarity(candidate.flatten().float(), reference["output"].flatten().float(), dim=0)),
                    "relative_l2": float(delta.norm() / reference["output"].float().norm().clamp_min(1e-12)),
                    "max_abs": float(delta.abs().max()),
                    "route_agreement": float((candidate_ids == reference["ids"]).float().mean()),
                })

    # Per-shape warmup.
    for length, variant, block in cases:
        for _ in range(args.warmup):
            world_sync()
            if variant == "serial":
                run_serial(length)
            else:
                run_chunked(length, block, variant == "streaming")

    schedule = [(case, rep) for rep in range(args.reps) for case in cases]
    random.Random(22117).shuffle(schedule)
    obj = [schedule if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    schedule = obj[0]
    started = time.time()
    for (length, variant, block), repeat in schedule:
        world_sync()
        if variant == "serial":
            timing, _, _ = run_serial(length)
        else:
            timing, _, _ = run_chunked(length, block, variant == "streaming")
        rows.append({
            "rank": rank,
            "physical_gpu": [4, 5, 6, 7][rank],
            "layer": args.layer,
            "length": length,
            "local_tokens": length // world,
            "variant": variant,
            "block": block,
            "repeat": repeat,
            **timing,
        })

    result = {
        "scope": "EXACT_QWEN3_VL_CP4_DEEPEP_EP4_STREAMING_LAYER_REPLAY_NOT_SERVING",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rank": rank,
        "physical_gpu": [4, 5, 6, 7][rank],
        "layer": args.layer,
        "warmup": args.warmup,
        "reps": args.reps,
        "duration_sec": time.time() - started,
        "checkpoint": str(args.model),
        "hidden_capture": str(args.hidden),
        "deep_ep": str(Path(deep_ep.__file__).resolve()),
        "deep_ep_sm90": bool(deep_ep.Buffer.is_sm90_compiled()),
        "clean_timing": args.clean_timing,
        "correctness": correctness,
        "rows": rows,
    }
    (args.out / f"streaming_rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    world_sync()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
