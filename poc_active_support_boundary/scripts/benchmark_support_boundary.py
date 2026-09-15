#!/usr/bin/env python3
"""Measure active-expert support and MoE MLP-boundary taxes on H100 BF16.

This is an owner-local, prepacked expert benchmark.  It preserves the exact
LLaDA2.0-Flash local expert contract (E=64, H=4096, I=1024) while separating:

* full-64 descriptors versus active-only descriptors;
* prebuilt offsets versus offsets rebuilt from GPU counts;
* gate/up grouped GEMM, SwiGLU, down grouped GEMM, and route-weight epilogue.

The active-only variant uses copied active weights prepared outside the timed
region.  It is therefore an optimistic descriptor oracle, not a deployable
kernel or a measured end-to-end implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


VISIBLE = "4,5,6,7"
EXPECTED_UUIDS = [
    "6076e2f2-5b63-3761-5586-56ceb7df8139",
    "a1a1cfcf-93a1-3544-9a5e-e58144b68730",
    "e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28",
    "4cc26b88-19fc-1988-f9e0-17858aa7a99b",
]
TOTAL_ROWS = [64, 128, 256, 512, 1024]
ACTIVE_EXPERTS = [1, 2, 4, 8, 16, 32, 64]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def event_samples(fn: Callable[[], torch.Tensor], repetitions: int) -> list[float]:
    values: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        if output.numel() and not torch.isfinite(output.reshape(-1)[0]):
            raise RuntimeError("non-finite output")
        values.append(float(start.elapsed_time(end)))
    return values


def wall_samples_us(fn: Callable[[], object], repetitions: int) -> list[float]:
    values: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        output = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter_ns() - begin
        if isinstance(output, torch.Tensor) and output.numel():
            _ = output.reshape(-1)[0]
        values.append(elapsed / 1e3)
    return values


def counts_for(total: int, active: int, geometry: str) -> list[int]:
    if total < active:
        raise ValueError((total, active))
    if geometry == "uniform" or active == 1:
        base, rem = divmod(total, active)
        return [base + (index < rem) for index in range(active)]
    if geometry != "high_variance":
        raise ValueError(geometry)
    tiny = active // 2
    counts = [1] * tiny
    remaining = total - tiny
    broad = active - tiny
    base, rem = divmod(remaining, broad)
    counts.extend(base + (index < rem) for index in range(broad))
    assert len(counts) == active and sum(counts) == total and min(counts) > 0
    return counts


def make_inputs(
    counts: list[int], hidden: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((sum(counts), hidden), generator=generator,
                    device=device, dtype=torch.bfloat16) * 0.02
    ids = torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.long),
        torch.tensor(counts, device=device, dtype=torch.long),
    )[:, None]
    route_weights = torch.rand((sum(counts), 1), generator=generator,
                               device=device, dtype=torch.bfloat16)
    return x, ids, route_weights


def grouped_components(
    x: torch.Tensor,
    offsets: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    route_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_up = torch._grouped_mm(x, gate_up_weights, offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    down = torch._grouped_mm(activated, down_weights, offs=offsets)
    return gate_up, activated, down * route_weights


def benchmark_case(
    *,
    total: int,
    active: int,
    geometry: str,
    full_counts_override: list[int] | None = None,
    case_id: str = "synthetic",
    task: str = "control",
    phase: str = "control",
    scope: str = "synthetic",
    hidden: int,
    intermediate: int,
    device: torch.device,
    w1: torch.Tensor,
    w2: torch.Tensor,
    fused_experts: Callable,
    warmup: int,
    repetitions: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    full_counts = (counts_for(total, active, geometry) + [0] * (64 - active)
                   if full_counts_override is None else full_counts_override)
    assert len(full_counts) == 64 and sum(full_counts) == total
    active_ids = [index for index, count in enumerate(full_counts) if count]
    assert len(active_ids) == active
    active_counts = [full_counts[index] for index in active_ids]
    x, ids, route_weights = make_inputs(full_counts, hidden, device, seed)
    full_offsets = torch.tensor(full_counts, device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32)
    active_offsets = torch.tensor(active_counts, device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32)
    full_counts_gpu = torch.tensor(full_counts, device=device, dtype=torch.int32)

    # Real expert IDs are sparse. A static active weight gather is prepared
    # outside timing: this deliberately overestimates a deployable descriptor
    # table, which must refer to resident weights without actually copying.
    full_gate_up_weights = w1.transpose(1, 2)
    full_down_weights = w2.transpose(1, 2)
    if active_ids == list(range(active)):
        active_gate_up_weights = w1[:active].transpose(1, 2)
        active_down_weights = w2[:active].transpose(1, 2)
    else:
        idx = torch.tensor(active_ids, device=device)
        active_gate_up_weights = w1.index_select(0, idx).transpose(1, 2)
        active_down_weights = w2.index_select(0, idx).transpose(1, 2)

    gate_up = torch._grouped_mm(x, full_gate_up_weights, offs=full_offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    down = torch._grouped_mm(activated, full_down_weights, offs=full_offsets)
    torch.cuda.synchronize()

    component_ops: dict[str, Callable[[], torch.Tensor]] = {
        "gate_up_full64_prebuilt": lambda: torch._grouped_mm(
            x, full_gate_up_weights, offs=full_offsets),
        "gate_up_active_prebuilt": lambda: torch._grouped_mm(
            x, active_gate_up_weights, offs=active_offsets),
        "swiglu": lambda: torch.nn.functional.silu(gate) * up,
        "down_full64_prebuilt": lambda: torch._grouped_mm(
            activated, full_down_weights, offs=full_offsets),
        "down_active_prebuilt": lambda: torch._grouped_mm(
            activated, active_down_weights, offs=active_offsets),
        "route_weight": lambda: down * route_weights,
    }

    def full64_prebuilt() -> torch.Tensor:
        return grouped_components(
            x, full_offsets, full_gate_up_weights, full_down_weights,
            route_weights)[2]

    def active_prebuilt() -> torch.Tensor:
        return grouped_components(
            x, active_offsets, active_gate_up_weights, active_down_weights,
            route_weights)[2]

    def full64_gpu_cumsum() -> torch.Tensor:
        offsets = full_counts_gpu.cumsum(0, dtype=torch.int32)
        return grouped_components(
            x, offsets, full_gate_up_weights, full_down_weights,
            route_weights)[2]

    def full64_cpu_list_offsets() -> torch.Tensor:
        offsets = torch.tensor(full_counts, device=device,
                               dtype=torch.int32).cumsum(0, dtype=torch.int32)
        return grouped_components(
            x, offsets, full_gate_up_weights, full_down_weights,
            route_weights)[2]

    def vllm_fused() -> torch.Tensor:
        return fused_experts(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=route_weights,
            topk_ids=ids,
            inplace=False,
            activation="silu",
            is_act_and_mul=True,
        )

    stage_ops = {
        "torch_grouped_full64_prebuilt": full64_prebuilt,
        "torch_grouped_active_prebuilt_oracle": active_prebuilt,
        "torch_grouped_full64_gpu_cumsum": full64_gpu_cumsum,
        "torch_grouped_full64_cpu_list_offsets": full64_cpu_list_offsets,
        "vllm_fused": vllm_fused,
    }

    all_ops = {**component_ops, **stage_ops}
    for fn in all_ops.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    order = list(all_ops)
    random.Random(seed + 71).shuffle(order)
    common = {
        "case_id": case_id,
        "task": task,
        "phase": phase,
        "scope": scope,
        "total_rows": total,
        "active_experts": active,
        "inactive_experts": 64 - active,
        "geometry": geometry,
        "mean_m_e": total / active,
        "min_m_e": min(active_counts),
        "max_m_e": max(active_counts),
        "m_e_cv": statistics.pstdev(active_counts) / statistics.mean(active_counts),
    }
    component_rows: list[dict] = []
    stage_rows: list[dict] = []
    for name in order:
        samples = event_samples(all_ops[name], repetitions)
        target = component_rows if name in component_ops else stage_rows
        for repetition, latency in enumerate(samples):
            target.append({**common, "operation": name, "repetition": repetition,
                           "latency_ms": latency})

    reference = vllm_fused()
    grouped = full64_prebuilt()
    active_grouped = active_prebuilt()
    torch.cuda.synchronize()
    correctness_rows = []
    for name, candidate in [("torch_grouped_full64_prebuilt", grouped),
                            ("torch_grouped_active_prebuilt_oracle", active_grouped)]:
        delta = candidate.float().flatten() - reference.float().flatten()
        correctness_rows.append({
            **common,
            "candidate": name,
            "max_abs": float(delta.abs().max()),
            "mean_abs": float(delta.abs().mean()),
            "relative_l2": float(torch.linalg.vector_norm(delta) /
                                 torch.linalg.vector_norm(reference.float().flatten()).clamp_min(1e-12)),
            "cosine": float(torch.nn.functional.cosine_similarity(
                candidate.float().flatten(), reference.float().flatten(), dim=0)),
        })
    return stage_rows, component_rows, correctness_rows


def load_real_cases(path: Path, max_per_task_phase: int) -> list[dict]:
    picked: list[dict] = []
    seen: dict[tuple[str, str], int] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("geometry") != "real" or not sum(row["m_e_vector"]):
            continue
        key = row["task"], row["phase"]
        if seen.get(key, 0) >= max_per_task_phase:
            continue
        seen[key] = seen.get(key, 0) + 1
        picked.append(row)
    return picked


def benchmark_metadata(
    device: torch.device, warmup: int, repetitions: int
) -> list[dict]:
    rows: list[dict] = []
    from vllm.model_executor.layers.fused_moe.modular_kernel import ExpertTokensMetadata

    for active in ACTIVE_EXPERTS:
        counts = counts_for(256, active, "uniform") + [0] * (64 - active)
        counts_gpu = torch.tensor(counts, device=device, dtype=torch.int32)
        operations: dict[str, Callable[[], object]] = {
            "gpu_counts_cumsum": lambda: counts_gpu.cumsum(0, dtype=torch.int32),
            "cpu_list_to_cuda_offsets": lambda: torch.tensor(
                counts, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32),
            "vllm_make_from_list": lambda: ExpertTokensMetadata.make_from_list(
                counts, device=str(device)),
        }
        for fn in operations.values():
            for _ in range(warmup):
                fn()
        for name, fn in operations.items():
            for repetition, latency_us in enumerate(wall_samples_us(fn, repetitions)):
                rows.append({
                    "active_experts": active,
                    "inactive_experts": 64 - active,
                    "operation": name,
                    "repetition": repetition,
                    "host_visible_latency_us": latency_us,
                })
    return rows


def benchmark_copy_bandwidth(
    device: torch.device, warmup: int, repetitions: int
) -> list[dict]:
    rows: list[dict] = []
    for total in TOTAL_ROWS:
        # 12 KiB/row is the gate/up + activated intermediate round trip.
        elements = total * 6144
        source = torch.empty(elements, device=device, dtype=torch.bfloat16)
        destination = torch.empty_like(source)
        fn = lambda: destination.copy_(source)
        for _ in range(warmup):
            fn()
        for repetition, latency in enumerate(event_samples(fn, repetitions)):
            rows.append({
                "total_rows": total,
                "bytes": elements * 2,
                "repetition": repetition,
                "latency_ms": latency,
                "effective_gbps": elements * 2 / (latency * 1e6),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-gpu", type=int, required=True, choices=range(4))
    parser.add_argument("--restart", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--real-replay", type=Path)
    parser.add_argument("--max-per-task-phase", type=int, default=2)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE}")
    if torch.cuda.device_count() != 4:
        raise SystemExit("expected four visible GPUs")
    device = torch.device(f"cuda:{args.logical_gpu}")
    torch.cuda.set_device(device)
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    if uuid != EXPECTED_UUIDS[args.logical_gpu]:
        raise SystemExit(f"logical GPU UUID mismatch: {uuid}")

    from vllm.model_executor.layers.fused_moe import fused_experts

    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(20260915)
    w1 = torch.randn((64, 2 * args.intermediate, args.hidden), generator=generator,
                     device=device, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn((64, args.hidden, args.intermediate), generator=generator,
                     device=device, dtype=torch.bfloat16) * 0.01
    torch.cuda.synchronize()

    stage_rows: list[dict] = []
    component_rows: list[dict] = []
    correctness_rows: list[dict] = []
    started = time.time()
    case_index = 0
    cases: list[dict] = []
    if args.mode == "synthetic":
        for total in TOTAL_ROWS:
            for active in ACTIVE_EXPERTS:
                if active > total:
                    continue
                for geometry in ("uniform", "high_variance"):
                    if geometry == "high_variance" and active == 1:
                        continue
                    cases.append({"total": total, "active": active,
                                  "geometry": geometry})
    else:
        if args.real_replay is None:
            raise SystemExit("--real-replay required in real mode")
        for row in load_real_cases(args.real_replay, args.max_per_task_phase):
            vector = list(map(int, row["m_e_vector"]))
            cases.append({
                "total": sum(vector), "active": sum(count > 0 for count in vector),
                "geometry": "real", "full_counts_override": vector,
                "case_id": row["case_id"], "task": row["task"],
                "phase": row["phase"],
                "scope": row.get("worklist_scope", "real_route_replay"),
            })
    for case in cases:
        case_index += 1
        stage, component, correctness = benchmark_case(
            **case,
            hidden=args.hidden,
            intermediate=args.intermediate,
            device=device,
            w1=w1,
            w2=w2,
            fused_experts=fused_experts,
            warmup=args.warmup,
            repetitions=args.repetitions,
            seed=20260915 + args.restart * 10000 + case_index,
        )
        stage_rows.extend(stage)
        component_rows.extend(component)
        correctness_rows.extend(correctness)
    metadata_rows = benchmark_metadata(device, args.warmup, args.repetitions)
    copy_rows = benchmark_copy_bandwidth(device, args.warmup, args.repetitions)

    write_csv(args.output / "ACTIVE_SUPPORT_RAW.csv", stage_rows)
    write_csv(args.output / "MLP_COMPONENT_RAW.csv", component_rows)
    write_csv(args.output / "METADATA_PREP_RAW.csv", metadata_rows)
    write_csv(args.output / "COPY_BANDWIDTH_RAW.csv", copy_rows)
    write_csv(args.output / "CORRECTNESS.csv", correctness_rows)
    audit = {
        "cuda_visible_devices": VISIBLE,
        "logical_gpu": args.logical_gpu,
        "physical_gpu": args.logical_gpu + 4,
        "physical_uuid": uuid,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm_fused_experts": str(fused_experts),
        "torch_grouped_mm": str(torch._grouped_mm),
        "restart": args.restart,
        "mode": args.mode,
        "case_count": len(cases),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "wall_seconds": time.time() - started,
        "scope": "owner-local prepacked exact BF16 expert execution; active-only is an optimistic contiguous-weight descriptor oracle",
    }
    (args.output / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
