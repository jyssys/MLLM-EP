#!/usr/bin/env python3
"""H100 BF16 expert-kernel envelope for RefineGEMM.

This is an owner-local expert execution benchmark.  It intentionally excludes
router, DeepEP dispatch, and combine.  Real M_e vectors come from measured
LLaDA2 EP4 routes after a future-known live-row filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

import torch


VISIBLE = "4,5,6,7"
PHYSICAL_UUID = "6076e2f2-5b63-3761-5586-56ceb7df8139"
M_VALUES = [1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256]


def event_ms(fn: Callable[[], torch.Tensor], repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        if not torch.isfinite(out.flatten()[0]):
            raise RuntimeError("non-finite output")
        values.append(float(start.elapsed_time(end)))
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def grouped_full_mlp(
    x: torch.Tensor,
    counts: list[int],
    w1: torch.Tensor,
    w2: torch.Tensor,
    route_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    # Equal adjacent offsets are accepted for inactive experts.  Keeping all
    # weights avoids an invalid timed multi-GB advanced-index copy.
    offsets = torch.tensor(counts, device=x.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    # torch._grouped_mm expects A packed by group and B=[G,K,N].
    gate_up_weights = w1.transpose(1, 2)
    down_weights = w2.transpose(1, 2)
    gate_up = torch._grouped_mm(x, gate_up_weights, offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    output = torch._grouped_mm(activated, down_weights, offs=offsets)
    return output if route_weights is None else output * route_weights


def make_case(vector: list[int], hidden: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.repeat_interleave(
        torch.arange(len(vector), device=device, dtype=torch.long),
        torch.tensor(vector, device=device, dtype=torch.long),
    )
    generator = torch.Generator(device=device).manual_seed(20260915 + sum(vector))
    x = torch.randn((sum(vector), hidden), generator=generator, device=device, dtype=torch.bfloat16) * 0.02
    weights = torch.ones((sum(vector), 1), device=device, dtype=torch.bfloat16)
    return x, ids[:, None], weights


def fused_call(fused_experts, x, ids, route_weights, w1, w2):
    if x.shape[0] == 0:
        return torch.empty_like(x)
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


def single_sweep(args, fused_experts, device, w1_one, w2_one) -> list[dict]:
    rows: list[dict] = []
    for m_e in M_VALUES:
        x = torch.randn((m_e, args.hidden), device=device, dtype=torch.bfloat16) * 0.02
        gate_up_ref = lambda: torch.nn.functional.linear(x, w1_one[0])
        down_input = torch.randn((m_e, args.intermediate), device=device, dtype=torch.bfloat16) * 0.02
        down_ref = lambda: torch.nn.functional.linear(down_input, w2_one[0])
        ids = torch.zeros((m_e, 1), device=device, dtype=torch.long)
        route_weights = torch.ones((m_e, 1), device=device, dtype=torch.bfloat16)
        fused = lambda: fused_call(fused_experts, x, ids, route_weights, w1_one, w2_one)
        grouped = lambda: grouped_full_mlp(x, [m_e], w1_one, w2_one, route_weights)
        ops = {
            "torch_mm_gate_up": (gate_up_ref, 4 * m_e * args.hidden * args.intermediate),
            "torch_mm_down": (down_ref, 2 * m_e * args.hidden * args.intermediate),
            "vllm_fused_full_mlp": (fused, 6 * m_e * args.hidden * args.intermediate),
            "torch_grouped_full_mlp": (grouped, 6 * m_e * args.hidden * args.intermediate),
        }
        for fn, _ in ops.values():
            for _ in range(args.warmup):
                fn()
        torch.cuda.synchronize()
        order = list(ops)
        random.Random(9000 + m_e).shuffle(order)
        for backend in order:
            fn, flops = ops[backend]
            for repetition, latency in enumerate(event_ms(fn, args.repetitions)):
                rows.append(
                    {
                        "m_e": m_e,
                        "backend": backend,
                        "repetition": repetition,
                        "latency_ms": latency,
                        "effective_tflops": flops / (latency * 1e9),
                        "dtype": "bfloat16",
                        "hidden": args.hidden,
                        "intermediate": args.intermediate,
                        "device": "physical_gpu4",
                    }
                )
    return rows


def load_replays(path: Path, max_real_per_phase: int) -> list[dict]:
    all_rows = [json.loads(line) for line in path.read_text().splitlines()]
    picked: list[dict] = []
    real_counts: dict[tuple[str, str], int] = {}
    selected_groups: set[str] = set()
    for row in all_rows:
        if row["geometry"] != "real":
            continue
        key = (row["task"], row["phase"])
        if real_counts.get(key, 0) < max_real_per_phase:
            real_counts[key] = real_counts.get(key, 0) + 1
            selected_groups.add(row["same_total_assignment_control_group"])
    for row in all_rows:
        if row["same_total_assignment_control_group"] in selected_groups:
            picked.append(row)
    for m_e in M_VALUES:
        vector = [m_e] * 64
        picked.append(
            {
                "case_id": f"homogeneous_me{m_e}",
                "geometry": "homogeneous_control",
                "task": "control",
                "wave": -1,
                "layer": -1,
                "rank": 0,
                "phase": "control",
                "fresh_m": 8 * m_e,
                "m_e_vector": vector,
                "same_total_assignment_control_group": f"homogeneous_me{m_e}",
            }
        )
    return picked


def bench_replays(args, fused_experts, device, w1, w2) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    correctness: list[dict] = []
    cases = load_replays(args.replay, args.max_real_per_phase)
    for case_index, case in enumerate(cases):
        counts = [int(v) for v in case["m_e_vector"]]
        if sum(counts) == 0:
            continue
        x, ids, route_weights = make_case(counts, args.hidden, device)
        whole_ops: dict[str, Callable] = {
            "vllm_fused": lambda: fused_call(fused_experts, x, ids, route_weights, w1, w2),
            "torch_grouped": lambda: grouped_full_mlp(x, counts, w1, w2, route_weights),
        }
        reference = whole_ops["vllm_fused"]()
        candidate = whole_ops["torch_grouped"]()
        torch.cuda.synchronize()
        diff = (candidate.float() - reference.float()).flatten()
        correctness.append(
            {
                "case_id": case["case_id"],
                "rows": sum(counts),
                "max_abs": float(diff.abs().max()),
                "mean_abs": float(diff.abs().mean()),
                "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference.float().flatten()).clamp_min(1e-12)),
                "cosine": float(torch.nn.functional.cosine_similarity(candidate.float().flatten(), reference.float().flatten(), dim=0)),
                "note": "independent BF16 grouped reduction-order comparison",
            }
        )

        # Real measured subgroup executions. Partition/scatter is timed and the
        # complete resident weight tensors are reused; no multi-GB weight copy
        # is hidden outside the timing boundary.
        for threshold in (1, 2, 4, 8, 16):
            tiny_experts = [i for i, count in enumerate(counts) if 0 < count <= threshold]
            large_experts = [i for i, count in enumerate(counts) if count > threshold]
            row_experts = ids[:, 0]
            tiny_mask = torch.zeros(64, device=device, dtype=torch.bool)
            if tiny_experts:
                tiny_mask[torch.tensor(tiny_experts, device=device)] = True
            tiny_idx = torch.nonzero(tiny_mask[row_experts], as_tuple=False).flatten()
            large_idx = torch.nonzero(~tiny_mask[row_experts], as_tuple=False).flatten()
            tiny_counts = [count if i in tiny_experts else 0 for i, count in enumerate(counts)]

            def hybrid(
                tiny_idx=tiny_idx,
                large_idx=large_idx,
                tiny_counts=tiny_counts,
            ):
                output = torch.empty_like(x)
                if tiny_idx.numel():
                    xt = x.index_select(0, tiny_idx)
                    # xt is still expert-major because the source is expert-major.
                    yt = grouped_full_mlp(xt, tiny_counts, w1, w2, route_weights.index_select(0, tiny_idx))
                    output.index_copy_(0, tiny_idx, yt)
                if large_idx.numel():
                    xl = x.index_select(0, large_idx)
                    il = ids.index_select(0, large_idx)
                    wl = route_weights.index_select(0, large_idx)
                    yl = fused_call(fused_experts, xl, il, wl, w1, w2)
                    output.index_copy_(0, large_idx, yl)
                return output

            whole_ops[f"hybrid_grouped_tiny_le{threshold}_fused_large"] = hybrid

            large_counts = [count if count > threshold else 0 for count in counts]

            def reverse_hybrid(
                tiny_idx=tiny_idx,
                large_idx=large_idx,
                large_counts=large_counts,
            ):
                output = torch.empty_like(x)
                if tiny_idx.numel():
                    xt = x.index_select(0, tiny_idx)
                    it = ids.index_select(0, tiny_idx)
                    wt = route_weights.index_select(0, tiny_idx)
                    yt = fused_call(fused_experts, xt, it, wt, w1, w2)
                    output.index_copy_(0, tiny_idx, yt)
                if large_idx.numel():
                    xl = x.index_select(0, large_idx)
                    wl = route_weights.index_select(0, large_idx)
                    yl = grouped_full_mlp(xl, large_counts, w1, w2, wl)
                    output.index_copy_(0, large_idx, yl)
                return output

            whole_ops[f"hybrid_fused_tiny_le{threshold}_grouped_large"] = reverse_hybrid

        for fn in whole_ops.values():
            for _ in range(args.warmup):
                fn()
        torch.cuda.synchronize()
        names = list(whole_ops)
        random.Random(20260915 + case_index).shuffle(names)
        for name in names:
            for repetition, latency in enumerate(event_ms(whole_ops[name], args.repetitions)):
                values = [v for v in counts if v]
                results.append(
                    {
                        "case_id": case["case_id"],
                        "control_group": case["same_total_assignment_control_group"],
                        "geometry": case["geometry"],
                        "task": case["task"],
                        "phase": case["phase"],
                        "wave": case["wave"],
                        "layer": case["layer"],
                        "rank": case["rank"],
                        "backend": name,
                        "repetition": repetition,
                        "total_assignments": sum(counts),
                        "active_experts": len(values),
                        "median_m_e": float(torch.tensor(values, dtype=torch.float32).median()) if values else 0.0,
                        "max_m_e": max(values, default=0),
                        "tiny_le4_fraction": sum(v <= 4 for v in values) / len(values) if values else 0.0,
                        "m_e_cv": float(torch.tensor(values, dtype=torch.float32).std(unbiased=False) / torch.tensor(values, dtype=torch.float32).mean()) if values else 0.0,
                        "latency_ms": latency,
                        "evidence_boundary": "owner-local prepacked replay; hybrid includes row partition/scatter and reuses complete resident weight tensors",
                    }
                )
        del x, ids, route_weights, reference, candidate
        torch.cuda.empty_cache()
    return results, correctness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--max-real-per-phase", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE}")
    if torch.cuda.device_count() != 4:
        raise SystemExit("expected four visible GPUs")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if uuid != PHYSICAL_UUID:
        raise SystemExit(f"logical cuda:0 UUID mismatch: {uuid}")
    args.output.mkdir(parents=True, exist_ok=True)
    from vllm.model_executor.layers.fused_moe import fused_experts

    torch.manual_seed(20260915)
    device = torch.device("cuda:0")
    w1 = torch.randn((64, 2 * args.intermediate, args.hidden), device=device, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn((64, args.hidden, args.intermediate), device=device, dtype=torch.bfloat16) * 0.01
    torch.cuda.synchronize()

    started = time.time()
    single = single_sweep(args, fused_experts, device, w1[:1], w2[:1])
    write_csv(args.output / "SINGLE_EXPERT_SWEEP.csv", single)
    replay, correctness = bench_replays(args, fused_experts, device, w1, w2)
    write_csv(args.output / "EXISTING_EXPERT_KERNELS.csv", replay)
    write_csv(args.output / "KERNEL_CORRECTNESS.csv", correctness)
    audit = {
        "cuda_visible_devices": VISIBLE,
        "logical_cuda0_physical_uuid": uuid,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm_fused_experts": str(fused_experts),
        "torch_grouped_mm": str(torch._grouped_mm),
        "dtype": "bfloat16",
        "shape": {"hidden": args.hidden, "intermediate": args.intermediate, "local_experts": 64},
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "wall_seconds": time.time() - started,
        "scope": "owner-local expert execution only; no DeepEP communication",
    }
    (args.output / "KERNEL_BENCHMARK_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
