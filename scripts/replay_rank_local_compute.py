#!/usr/bin/env python3
"""Replay virtual-rank expert histograms with the target BF16 grouped MLP.

The replay measures owner-local routed expert compute only.  It contains no
dispatch, combine, TP collective, or replicated-state bridge cost and never
uses EP2/P arithmetic scaling for EP4/EP8 predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from virtual_ep.mapping import ExpertOwnership, SourcePartition
from virtual_ep.schema import TraceBundle, invocation_slices
from virtual_ep.traffic import build_traffic


def grouped_mlp(
    x: torch.Tensor,
    counts: np.ndarray,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.as_tensor(counts, device=x.device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    gate_up = torch._grouped_mm(x, gate_up_weights, offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    return torch._grouped_mm(activated, down_weights, offs=offsets)


def owner_local_expert_stage(
    recv_hidden: torch.Tensor,
    recv_ids: torch.Tensor,
    recv_weights: torch.Tensor,
    local_experts: int,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    """Replay the post-DeepEP packing, grouped MLP, and weighted reduction."""

    row_index, slot_index = torch.where(recv_ids >= 0)
    expert_index = recv_ids[row_index, slot_index].long()
    order = torch.argsort(expert_index, stable=True)
    counts = torch.bincount(
        expert_index[order], minlength=local_experts
    ).to(torch.int32)
    branch_output = grouped_mlp(
        recv_hidden[row_index[order]], counts,
        gate_up_weights, down_weights,
    )
    branches = torch.zeros(
        recv_hidden.shape[0], recv_ids.shape[1], recv_hidden.shape[1],
        dtype=recv_hidden.dtype, device=recv_hidden.device,
    )
    branches[row_index[order], slot_index[order]] = branch_output
    return branches.float().mul_(recv_weights.unsqueeze(-1)).sum(dim=1).bfloat16()


def _invocations(trace: TraceBundle, ep_size: int) -> list[dict]:
    ownership = ExpertOwnership(int(trace.metadata["num_routed_experts"]), ep_size)
    partition = SourcePartition(ep_size)
    cases: list[dict] = []
    for key, selected in invocation_slices(trace.rows):
        source_keys = trace.rows["source_partition_key"][selected]
        invocation_ids = trace.rows["expert_ids"][selected]
        invocation_weights = trace.rows["router_weights"][selected]
        traffic = build_traffic(
            invocation_ids,
            source_keys,
            ownership,
            partition,
            int(trace.metadata["hidden_size"]),
        )
        for rank in range(ep_size):
            begin = rank * ownership.experts_per_rank
            end = begin + ownership.experts_per_rank
            counts = traffic.expert_assignments[begin:end]
            owners = ownership.owner(invocation_ids)
            received = np.any(owners == rank, axis=1)
            recv_ids = np.where(
                owners[received] == rank,
                invocation_ids[received] - begin,
                -1,
            ).astype(np.int16)
            active = counts[counts > 0]
            cases.append(
                {
                    "request_id": key[0],
                    "block_id": key[1],
                    "iteration_id": key[2],
                    "nfe": key[3],
                    "layer_id": key[4],
                    "rank": rank,
                    "counts": counts,
                    "recv_ids": recv_ids,
                    "recv_weights": invocation_weights[received].astype(np.float32),
                    "assignments": int(counts.sum()),
                    "active_experts": int(len(active)),
                    "max_rows_per_expert": int(active.max(initial=0)),
                    "cv_rows_per_expert": float(active.std() / active.mean()) if len(active) else 0.0,
                }
            )
    return cases


def _select_diverse(cases: list[dict], maximum: int) -> list[dict]:
    """Select measured-shape coverage without random row-level train leakage."""
    if len(cases) <= maximum:
        return cases
    # Keep extrema/quantiles along each principal structural dimension, then
    # fill in deterministic request/layer order. Heldout status remains at the
    # request level and is not used by selection.
    indices: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index not in seen and len(indices) < maximum:
            seen.add(index)
            indices.append(index)

    for field in (
        "assignments", "active_experts", "max_rows_per_expert", "cv_rows_per_expert"
    ):
        ordered = sorted(range(len(cases)), key=lambda index: cases[index][field])
        for quantile in np.linspace(0.0, 1.0, min(maximum // 4, 32)):
            add(ordered[round(quantile * (len(ordered) - 1))])
    ordered_all = sorted(
        range(len(cases)),
        key=lambda index: (
            cases[index]["request_id"], cases[index]["nfe"],
            cases[index]["layer_id"], cases[index]["rank"],
        ),
    )
    for index in ordered_all:
        if len(indices) >= maximum:
            break
        add(index)
    return [cases[index] for index in indices]


def _select_train_heldout(cases: list[dict], maximum: int) -> list[dict]:
    train = [case for case in cases if case["request_id"] % 5 != 0]
    heldout = [case for case in cases if case["request_id"] % 5 == 0]
    train_target = min(len(train), max(1, maximum * 2 // 3))
    heldout_target = min(len(heldout), maximum - train_target)
    selected = _select_diverse(train, train_target)
    selected.extend(_select_diverse(heldout, heldout_target))
    remaining = maximum - len(selected)
    if remaining:
        selected_ids = {id(case) for case in selected}
        selected.extend(
            case for case in cases if id(case) not in selected_ids
        )
    return selected[:maximum]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ep", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    trace = TraceBundle.load(args.trace)
    if int(trace.metadata["hidden_size"]) != 2048:
        raise RuntimeError("this replay is bounded to LLaDA2.0-mini hidden=2048")
    local_experts = int(trace.metadata["num_routed_experts"]) // args.ep
    hidden = int(trace.metadata["hidden_size"])
    intermediate = 512
    generator = torch.Generator(device=device).manual_seed(20260916 + args.ep)
    gate_up_weights = torch.randn(
        local_experts, hidden, 2 * intermediate,
        dtype=torch.bfloat16, device=device, generator=generator,
    ) * 0.01
    down_weights = torch.randn(
        local_experts, intermediate, hidden,
        dtype=torch.bfloat16, device=device, generator=generator,
    ) * 0.01
    cases = _select_train_heldout(_invocations(trace, args.ep), args.maximum_cases)
    rows = []
    for case_index, case in enumerate(cases):
        counts = case["counts"]
        assignments = int(counts.sum())
        if assignments == 0:
            continue
        recv_ids = torch.as_tensor(case["recv_ids"], device=device).long()
        recv_weights = torch.as_tensor(
            case["recv_weights"], device=device, dtype=torch.float32
        )
        x = torch.randn(
            recv_ids.shape[0], hidden, dtype=torch.bfloat16, device=device,
            generator=generator,
        ) * 0.02
        for _ in range(args.warmup):
            owner_local_expert_stage(
                x, recv_ids, recv_weights, local_experts,
                gate_up_weights, down_weights,
            )
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output = owner_local_expert_stage(
                x, recv_ids, recv_weights, local_experts,
                gate_up_weights, down_weights,
            )
            end.record()
            end.synchronize()
            if not torch.isfinite(output.flatten()[0]):
                raise RuntimeError("non-finite grouped MLP replay output")
            samples.append(begin.elapsed_time(end))
        samples.sort()
        split = "heldout" if case["request_id"] % 5 == 0 else "train"
        rows.append(
            {
                "ep_size": args.ep,
                "sample_id": (
                    f"request{case['request_id']}_nfe{case['nfe']}_"
                    f"layer{case['layer_id']}_rank{case['rank']}"
                ),
                "request_id": case["request_id"],
                "block_id": case["block_id"],
                "iteration_id": case["iteration_id"],
                "nfe": case["nfe"],
                "layer_id": case["layer_id"],
                "virtual_rank": case["rank"],
                "active_experts": case["active_experts"],
                "assignments": assignments,
                "recv_unique_activation_rows": int(recv_ids.shape[0]),
                "max_rows_per_expert": case["max_rows_per_expert"],
                "cv_rows_per_expert": case["cv_rows_per_expert"],
                "latency_ms": float(np.median(samples)),
                "p10_ms": float(np.percentile(samples, 10)),
                "p90_ms": float(np.percentile(samples, 90)),
                "split": split,
                "backend": "DeepEP_receive_format_plus_torch_grouped_mm_bf16_MLP",
            }
        )
        del x, recv_ids, recv_weights, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "scope": "rank-local routed expert compute only",
                "excluded": [
                    "EP dispatch", "EP combine", "replicated-state bridge all-gather",
                ],
                "ep_size": args.ep,
                "backend": str(torch._grouped_mm),
                "dtype": "bfloat16",
                "hidden": hidden,
                "intermediate": intermediate,
                "local_experts": local_experts,
                "measured_cases": len(rows),
                "selection": "deterministic diverse actual-trace shapes",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
