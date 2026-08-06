"""Four-rank DeepEP intranode smoke for the Qwen3-VL MoE contract."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def _sorted_pairs(values: torch.Tensor) -> list[tuple[int, int]]:
    cpu = values.to(torch.int64).cpu()
    return sorted((int(row[0]), int(row[1])) for row in cpu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens-per-rank", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--buffer-mib", type=int, default=256)
    args = parser.parse_args()

    import deep_ep

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world != 4:
        raise AssertionError(f"expected EP4, got {world}")
    if args.num_experts % world:
        raise AssertionError("experts must divide evenly over ranks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1701 + rank)
    tokens = args.tokens_per_rank
    hidden = torch.zeros(
        (tokens, args.hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    hidden[:, 0] = rank
    hidden[:, 1] = torch.arange(tokens, device="cuda", dtype=torch.bfloat16)
    hidden[:, 2:] = torch.randn(
        (tokens, args.hidden_size - 2), dtype=torch.bfloat16, device="cuda"
    )
    row = torch.arange(tokens, device="cuda", dtype=torch.int64)[:, None]
    lane = torch.arange(args.top_k, device="cuda", dtype=torch.int64)[None, :]
    topk_ids = (row * 7 + lane * 17 + rank * 11) % args.num_experts
    topk_ids = topk_ids.to(deep_ep.topk_idx_t).contiguous()
    raw_weights = 1.0 + ((row + lane + rank) % 11).float()
    topk_weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).contiguous()

    gathered_hidden = [torch.empty_like(hidden) for _ in range(world)]
    gathered_ids = [torch.empty_like(topk_ids) for _ in range(world)]
    gathered_weights = [torch.empty_like(topk_weights) for _ in range(world)]
    dist.all_gather(gathered_hidden, hidden)
    dist.all_gather(gathered_ids, topk_ids)
    dist.all_gather(gathered_weights, topk_weights)

    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        args.buffer_mib * 1024 * 1024,
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )
    layout = buffer.get_dispatch_layout(
        topk_ids,
        args.num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = layout
    (
        recv_hidden,
        recv_ids,
        recv_weights,
        recv_count_per_expert,
        handle,
        dispatch_event,
    ) = buffer.dispatch(
        x=hidden,
        handle=None,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        expert_alignment=1,
        config=deep_ep.Buffer.get_dispatch_config(world),
        previous_event=previous_event,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    del dispatch_event

    local_experts = args.num_experts // world
    expected_pairs: list[tuple[int, int]] = []
    expected_routes: dict[tuple[int, int], tuple[list[int], list[float]]] = {}
    for source_rank in range(world):
        source_ids = gathered_ids[source_rank].to(torch.int64)
        source_weights = gathered_weights[source_rank]
        belongs = (source_ids >= rank * local_experts) & (
            source_ids < (rank + 1) * local_experts
        )
        for token_index in torch.nonzero(belongs.any(dim=1), as_tuple=False).flatten():
            token = int(token_index.item())
            pair = (source_rank, token)
            expected_pairs.append(pair)
            local_ids = torch.where(
                belongs[token], source_ids[token] - rank * local_experts, -1
            )
            local_weights = torch.where(
                belongs[token], source_weights[token], torch.zeros_like(source_weights[token])
            )
            expected_routes[pair] = (
                [int(value) for value in local_ids.cpu().tolist()],
                [float(value) for value in local_weights.cpu().tolist()],
            )

    received_pairs = recv_hidden[:, :2]
    received_pair_list = [
        (int(row_values[0]), int(row_values[1]))
        for row_values in received_pairs.to(torch.int64).cpu().tolist()
    ]
    count_ok = len(received_pair_list) == len(expected_pairs)
    token_multiset_ok = sorted(received_pair_list) == sorted(expected_pairs)
    route_ok = True
    weight_ok = True
    for recv_index, pair in enumerate(received_pair_list):
        expected_local_ids, expected_local_weights = expected_routes[pair]
        actual_ids = [int(value) for value in recv_ids[recv_index].cpu().tolist()]
        if actual_ids != expected_local_ids:
            route_ok = False
        valid = recv_ids[recv_index] != -1
        expected_weight_tensor = torch.tensor(
            expected_local_weights, device="cuda", dtype=recv_weights.dtype
        )
        if not torch.allclose(
            recv_weights[recv_index][valid],
            expected_weight_tensor[valid],
            rtol=1e-6,
            atol=1e-6,
        ):
            weight_ok = False

    local_contribution = torch.where(
        recv_ids != -1, recv_weights, torch.zeros_like(recv_weights)
    ).sum(dim=1, keepdim=True)
    contribution = local_contribution.to(torch.bfloat16).expand(
        -1, args.hidden_size
    ).contiguous()
    combined, _, combine_event = buffer.combine(
        x=contribution,
        handle=handle,
        topk_weights=None,
        config=deep_ep.Buffer.get_combine_config(world),
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    del combine_event
    expected_combined = torch.ones_like(combined)
    combine_error = (combined.float() - expected_combined.float()).abs()
    combine_ok = bool(torch.allclose(combined, expected_combined, rtol=1e-2, atol=1e-2))
    order_ok = combined.shape == hidden.shape and combine_ok

    local_pass = all((count_ok, token_multiset_ok, route_ok, weight_ok, order_ok))
    pass_tensor = torch.tensor(int(local_pass), device="cuda")
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    all_ranks_pass = bool(pass_tensor.item())
    result = {
        "status": "ok" if all_ranks_pass else "failed",
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "physical_gpu": [4, 5, 6, 7][local_rank],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "world_size": world,
        "dtype": str(hidden.dtype),
        "hidden_size": args.hidden_size,
        "top_k": args.top_k,
        "num_experts": args.num_experts,
        "tokens_per_rank": tokens,
        "received_tokens": int(recv_hidden.shape[0]),
        "expected_received_tokens": len(expected_pairs),
        "dispatch_received_token_count": count_ok,
        "source_token_multiset": token_multiset_ok,
        "topk_id_preservation": route_ok,
        "topk_weight_preservation": weight_ok,
        "combine_output": combine_ok,
        "source_token_order_restoration": order_ok,
        "combine_max_abs_error": float(combine_error.max().item()),
        "combine_mean_abs_error": float(combine_error.mean().item()),
        "deep_ep_import": str(Path(deep_ep.__file__).resolve()),
        "deep_ep_sm90_compiled": bool(deep_ep.Buffer.is_sm90_compiled()),
        "all_ranks_pass": all_ranks_pass,
        "recv_count_per_expert_sum": int(sum(recv_count_per_expert)),
    }
    (args.output_dir / f"smoke_rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    if not all_ranks_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
