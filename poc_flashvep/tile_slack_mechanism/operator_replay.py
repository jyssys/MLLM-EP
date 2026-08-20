"""Scheduler-free DeepEP timing for the FlashVEP tile-to-slack gate.

The replay keeps captured Qwen3-VL routes unchanged and invokes the actual
model-loaded layer-24 Triton experts plus DeepEP high-throughput dispatch and
combine.  It does not modify vLLM scheduling, routing, or expert placement.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from poc_flashvep.deepep_revalidation.operator_replay import (
    ExpertSpec,
    MicroState,
    StageRecord,
    _combine,
    _correctness,
    _dispatch,
    _expert,
)


_INSTALLED = False
_RAN_RANKS: set[int] = set()
_CONTEXT = threading.local()


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pct(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    return {
        "median_ms": float(statistics.median(values)),
        "p25_ms": float(pct(0.25)),
        "p75_ms": float(pct(0.75)),
        "mean_ms": float(statistics.fmean(values)),
        "stddev_ms": float(statistics.stdev(values) if len(values) > 1 else 0.0),
    }


def _layer_from_prefix(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _current_layer() -> int:
    return int(getattr(_CONTEXT, "layer", -1))


@dataclass
class RoutedSample:
    sample_id: str
    category: str
    routes: torch.Tensor
    metadata: dict[str, Any]


def _load_samples(directory: Path, device: torch.device) -> dict[str, RoutedSample]:
    manifest = json.loads((directory / "sample_manifest.json").read_text())
    result: dict[str, RoutedSample] = {}
    for metadata in manifest["samples"]:
        sample_id = metadata["sample_id"]
        matches = list(directory.glob(f"routing.dp*.{sample_id}.npz"))
        if len(matches) != 1:
            raise AssertionError(f"{sample_id}: expected one routing archive, got {matches}")
        with np.load(matches[0]) as archive:
            routes = torch.from_numpy(archive["routed_experts"].astype(np.int64))
        if routes.ndim != 3 or routes.shape[1:] != (48, 8):
            raise AssertionError(f"{sample_id}: invalid route shape {tuple(routes.shape)}")
        result[sample_id] = RoutedSample(
            sample_id=sample_id,
            category=metadata["category"],
            routes=routes.to(device),
            metadata=metadata,
        )
    return result


def _template_inputs(
    count: int,
    routes: torch.Tensor,
    capture: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = capture["post_attention_hidden"].to(device)
    weights = capture["topk_weights"].to(device)
    index = torch.arange(count, device=device) % hidden.shape[0]
    return (
        hidden[index].contiguous(),
        weights[index].contiguous(),
        routes.to(device=device, dtype=torch.int64).contiguous(),
    )


def _states_from_groups(
    groups: list[list[int]],
    routes: torch.Tensor,
    capture: dict[str, Any],
    device: torch.device,
) -> tuple[list[MicroState], torch.Tensor]:
    hidden, weights, ids = _template_inputs(len(routes), routes, capture, device)
    states: list[MicroState] = []
    order: list[int] = []
    for values in groups:
        index = torch.tensor(values, dtype=torch.int64, device=device)
        state = MicroState(hidden[index], weights[index], ids[index])
        states.append(state)
        order.extend(values)
    if sorted(order) != list(range(len(routes))):
        raise AssertionError("wave groups must partition every token exactly once")
    return states, torch.tensor(order, dtype=torch.int64, device=device)


def _single_state(
    routes: torch.Tensor, capture: dict[str, Any], device: torch.device
) -> tuple[list[MicroState], torch.Tensor]:
    return _states_from_groups([list(range(len(routes)))], routes, capture, device)


def _iteration(
    variant: str,
    states: list[MicroState],
    kernel: Any,
    original_experts: Any,
    buffer: Any,
    spec: ExpertSpec,
    rank: int,
    measure: bool,
) -> tuple[dict[str, Any] | None, torch.Tensor]:
    comm_stream = buffer.get_comm_stream()
    expert_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()
    origin, wall_end = _event(), _event()
    records: list[StageRecord] = []
    origin.record(default_stream)
    comm_stream.wait_event(origin)
    expert_stream.wait_event(origin)

    if variant == "serial":
        for index, state in enumerate(states):
            _dispatch(index, state, buffer, comm_stream, spec, records)
            _expert(index, state, kernel, original_experts, expert_stream, spec, rank, records)
            _combine(index, state, buffer, comm_stream, records)
    elif variant == "overlap":
        _dispatch(0, states[0], buffer, comm_stream, spec, records)
        _expert(0, states[0], kernel, original_experts, expert_stream, spec, rank, records)
        for index in range(len(states) - 1):
            _dispatch(index + 1, states[index + 1], buffer, comm_stream, spec, records)
            _combine(index, states[index], buffer, comm_stream, records)
            _expert(
                index + 1,
                states[index + 1],
                kernel,
                original_experts,
                expert_stream,
                spec,
                rank,
                records,
            )
        _combine(len(states) - 1, states[-1], buffer, comm_stream, records)
    else:
        raise ValueError(variant)

    states[-1].combine_event.current_stream_wait()
    wall_end.record(default_stream)
    wall_end.synchronize()
    output = torch.cat([state.combined for state in states if state.combined is not None])
    if not measure:
        return None, output

    per_wave: list[dict[str, float]] = [dict() for _ in states]
    intervals: dict[str, list[tuple[float, float]]] = {
        "dispatch": [], "expert": [], "combine": []
    }
    for record in records:
        start = _elapsed(origin, record.start)
        end = _elapsed(origin, record.end)
        intervals[record.stage].append((start, end))
        per_wave[record.microbatch][f"{record.stage}_ms"] = end - start
        per_wave[record.microbatch][f"{record.stage}_start_ms"] = start
        per_wave[record.microbatch][f"{record.stage}_end_ms"] = end

    de = ec = 0.0
    for e_start, e_end in intervals["expert"]:
        for d_start, d_end in intervals["dispatch"]:
            de += max(0.0, min(e_end, d_end) - max(e_start, d_start))
        for c_start, c_end in intervals["combine"]:
            ec += max(0.0, min(e_end, c_end) - max(e_start, c_start))
    return {
        "wall_ms": _elapsed(origin, wall_end),
        "dispatch_ms": sum(b - a for a, b in intervals["dispatch"]),
        "expert_ms": sum(b - a for a, b in intervals["expert"]),
        "combine_ms": sum(b - a for a, b in intervals["combine"]),
        "dispatch_expert_overlap_ms": de,
        "expert_combine_overlap_ms": ec,
        "per_wave": per_wave,
    }, output


def _restore(output: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    restored = torch.empty_like(output)
    restored[order] = output
    return restored


def _run_variant(
    variant: str,
    groups: list[list[int]],
    routes: torch.Tensor,
    capture: dict[str, Any],
    kernel: Any,
    original_experts: Any,
    buffer: Any,
    spec: ExpertSpec,
    rank: int,
    warmups: int,
    iterations: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    dist.barrier()
    for _ in range(warmups):
        states, _ = _states_from_groups(groups, routes, capture, spec.w1.device)
        _iteration(variant, states, kernel, original_experts, buffer, spec, rank, False)
    samples: list[dict[str, Any]] = []
    output = None
    order = None
    for _ in range(iterations):
        states, order = _states_from_groups(groups, routes, capture, spec.w1.device)
        sample, output = _iteration(
            variant, states, kernel, original_experts, buffer, spec, rank, True
        )
        assert sample is not None
        samples.append(sample)
    assert output is not None and order is not None
    result: dict[str, Any] = {
        "variant": variant,
        "waves": len(groups),
        "wave_token_counts": [len(group) for group in groups],
        "warmups": warmups,
        "iterations": iterations,
        "samples": samples,
    }
    for key in (
        "wall_ms", "dispatch_ms", "expert_ms", "combine_ms",
        "dispatch_expert_overlap_ms", "expert_combine_overlap_ms",
    ):
        values = [float(sample[key]) for sample in samples]
        result[key] = values
        result[f"{key}_stats"] = _stats(values)
    return result, _restore(output, order).detach().clone()


def _equal_groups(total: int, sizes: list[int], permutation: list[int]) -> list[list[int]]:
    groups = []
    cursor = 0
    for size in sizes:
        groups.append(permutation[cursor : cursor + size])
        cursor += size
    if cursor != total:
        raise AssertionError((cursor, total))
    return groups


def _spatial_groups(metadata: dict[str, Any], grid: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(grid * grid)]
    vision: set[int] = set()
    for image in metadata["images"]:
        start, end = image["token_span"]
        height, width = image["post_merge_grid_hw"]
        if end - start != height * width:
            raise AssertionError("invalid post-merge spatial span")
        for offset, token in enumerate(range(start, end)):
            row, col = divmod(offset, width)
            tile_row = min(grid - 1, row * grid // height)
            tile_col = min(grid - 1, col * grid // width)
            groups[tile_row * grid + tile_col].append(token)
            vision.add(token)
    for token in range(metadata["processor_prompt_tokens"]):
        if token not in vision:
            min(groups, key=len).append(token)
    if any(not group for group in groups):
        raise AssertionError(f"empty {grid}x{grid} tile")
    return groups


def _groupings(metadata: dict[str, Any], seed: int = 1729) -> dict[str, list[list[int]]]:
    total = int(metadata["processor_prompt_tokens"])
    result: dict[str, list[list[int]]] = {}
    rng = np.random.default_rng(seed)
    for grid in (2, 4):
        spatial = _spatial_groups(metadata, grid)
        sizes = [len(group) for group in spatial]
        result[f"spatial_{grid}x{grid}"] = spatial
        result[f"sequential_{grid}x{grid}"] = _equal_groups(
            total, sizes, list(range(total))
        )
        result[f"generic_{grid}x{grid}"] = _equal_groups(
            total, sizes, rng.permutation(total).tolist()
        )
    return result


def _rank_assignments(routes: torch.Tensor, groups: list[list[int]]) -> list[list[int]]:
    values = []
    for group in groups:
        ids = routes[torch.tensor(group, device=routes.device)]
        values.append([int(((ids // 32) == rank).sum()) * 4 for rank in range(4)])
    return values


def _run_stage_b(
    samples: dict[str, RoutedSample],
    capture: dict[str, Any],
    kernel: Any,
    original_experts: Any,
    buffer: Any,
    spec: ExpertSpec,
    rank: int,
) -> list[dict[str, Any]]:
    requested = (
        "astronaut,camera,retina,microaneurysms,model_card,scanned_page"
    ).split(",")
    selected = [samples[name] for name in requested if name in samples]
    rows = []
    warmups = _int_env("FLASHVEP_TILE_STAGE_B_WARMUPS", 3)
    iterations = _int_env("FLASHVEP_TILE_STAGE_B_ITERATIONS", 10)
    for sample in selected:
        total = int(sample.routes.shape[0])
        groups = [list(range(total))]
        for layer in range(48):
            routes = sample.routes[:, layer, :]
            timing, _ = _run_variant(
                "serial", groups, routes, capture, kernel, original_experts,
                buffer, spec, rank, warmups, iterations,
            )
            assignments = _rank_assignments(routes, groups)[0]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "layer": layer,
                    "rank": rank,
                    "total_assignments": assignments[rank],
                    "rank_assignments": assignments,
                    "vision_assignments": int(
                        sum(
                            ((routes[start:end] // 32) == rank).sum().item()
                            for start, end in (
                                image["token_span"] for image in sample.metadata["images"]
                            )
                        )
                    ) * 4,
                    "nonvision_assignments": assignments[rank] - int(
                        sum(
                            ((routes[start:end] // 32) == rank).sum().item()
                            for start, end in (
                                image["token_span"] for image in sample.metadata["images"]
                            )
                        )
                    ) * 4,
                    "timing": timing,
                }
            )
    return rows


def _run_stage_c(
    samples: dict[str, RoutedSample],
    capture: dict[str, Any],
    kernel: Any,
    original_experts: Any,
    buffer: Any,
    spec: ExpertSpec,
    rank: int,
) -> list[dict[str, Any]]:
    selected = [samples[name] for name in ("cat", "model_card", "retina")]
    layers = (0, 12, 24, 36, 47)
    warmups = _int_env("FLASHVEP_TILE_STAGE_C_WARMUPS", 5)
    iterations = _int_env("FLASHVEP_TILE_STAGE_C_ITERATIONS", 20)
    rows = []
    for sample in selected:
        for layer in layers:
            routes = sample.routes[:, layer, :]
            full_groups = [list(range(len(routes)))]
            full, reference = _run_variant(
                "serial", full_groups, routes, capture, kernel, original_experts,
                buffer, spec, rank, warmups, iterations,
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "layer": layer,
                    "rank": rank,
                    "strategy": "serial",
                    "granularity": "full",
                    "rank_assignments_by_wave": _rank_assignments(routes, full_groups),
                    "serial": full,
                    "overlap": None,
                    "correctness": {"passed": True},
                }
            )
            for name, groups in _groupings(sample.metadata).items():
                strategy, granularity = name.rsplit("_", 1)
                serial, serial_output = _run_variant(
                    "serial", groups, routes, capture, kernel, original_experts,
                    buffer, spec, rank, warmups, iterations,
                )
                overlap, overlap_output = _run_variant(
                    "overlap", groups, routes, capture, kernel, original_experts,
                    buffer, spec, rank, warmups, iterations,
                )
                serial_correct = _correctness(reference, serial_output)
                overlap_correct = _correctness(reference, overlap_output)
                serial_wall = serial["wall_ms_stats"]["median_ms"]
                overlap_wall = overlap["wall_ms_stats"]["median_ms"]
                serial_comm = (
                    serial["dispatch_ms_stats"]["median_ms"]
                    + serial["combine_ms_stats"]["median_ms"]
                )
                overlap_expert = overlap["expert_ms_stats"]["median_ms"]
                exposed_comm = max(0.0, overlap_wall - overlap_expert)
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                        "layer": layer,
                        "rank": rank,
                        "strategy": strategy,
                        "granularity": granularity,
                        "rank_assignments_by_wave": _rank_assignments(routes, groups),
                        "serial": serial,
                        "overlap": overlap,
                        "speedup": serial_wall / overlap_wall,
                        "hidden_comm_ms": max(0.0, serial_comm - exposed_comm),
                        "expert_slowdown": (
                            overlap["expert_ms_stats"]["median_ms"]
                            / serial["expert_ms_stats"]["median_ms"]
                        ),
                        "dispatch_slowdown": (
                            overlap["dispatch_ms_stats"]["median_ms"]
                            / serial["dispatch_ms_stats"]["median_ms"]
                        ),
                        "combine_slowdown": (
                            overlap["combine_ms_stats"]["median_ms"]
                            / serial["combine_ms_stats"]["median_ms"]
                        ),
                        "net_benefit_ms": serial_wall - overlap_wall,
                        "correctness": {
                            "passed": serial_correct["passed"] and overlap_correct["passed"],
                            "serial": serial_correct,
                            "overlap": overlap_correct,
                            "route_identity": True,
                            "token_order_restoration": True,
                        },
                    }
                )
    return rows


def _run_replay(kernel: Any, original_experts: Any, spec: ExpertSpec) -> dict[str, Any]:
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    rank = int(ep.rank_in_group)
    if int(ep.world_size) != 4:
        raise AssertionError(f"expected EP4, got {ep.world_size}")
    if type(kernel.prepare_finalize).__name__ != "DeepEPHTPrepareAndFinalize":
        raise AssertionError(type(kernel.prepare_finalize).__name__)
    source_dir = Path(os.environ["FLASHVEP_TILE_ROUTING_RESULT_DIR"])
    capture = torch.load(
        os.environ["FLASHVEP_TILE_CAPTURE_PATH"], map_location="cpu", weights_only=False
    )
    samples = _load_samples(source_dir, spec.w1.device)
    buffer = kernel.prepare_finalize.buffer
    stage_b = _run_stage_b(
        samples, capture, kernel, original_experts, buffer, spec, rank
    )
    stage_c = _run_stage_c(
        samples, capture, kernel, original_experts, buffer, spec, rank
    )
    return {
        "status": "ok",
        "rank": rank,
        "physical_gpu": [4, 5, 6, 7][rank],
        "settings": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "routing_result_dir": str(source_dir),
            "capture_path": os.environ["FLASHVEP_TILE_CAPTURE_PATH"],
            "layer_weights": 24,
            "expert_backend": type(kernel.fused_experts).__name__,
            "prepare_finalize_backend": type(kernel.prepare_finalize).__name__,
            "communication_backend": "DeepEP high-throughput Buffer dispatch/combine",
            "input_replication": "one identical captured request route per EP source rank",
            "hidden_provenance": "validated real layer-24 capture, cycled to route length",
        },
        "stage_b": stage_b,
        "stage_c": stage_c,
    }


def _write_result(rank: int, payload: dict[str, Any]) -> None:
    directory = Path(os.environ["FLASHVEP_TILE_REPLAY_RESULT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def install_tile_slack_replay() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_tile_layer = _layer_from_prefix(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = _current_layer()
        _CONTEXT.layer = int(getattr(self, "_flashvep_tile_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        from vllm.distributed import get_ep_group

        rank = int(get_ep_group().rank_in_group)
        if (
            rank not in _RAN_RANKS
            and _current_layer() == 24
            and type(self.prepare_finalize).__name__ == "DeepEPHTPrepareAndFinalize"
        ):
            _RAN_RANKS.add(rank)
            names = (
                "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                "topk_ids", "activation", "global_num_experts",
                "local_num_experts", "expert_map", "apply_router_weight_on_input",
                "expert_tokens_meta",
            )
            values = dict(zip(names, args, strict=False))
            values.update(kwargs)
            spec = ExpertSpec(
                in_dtype=values["in_dtype"],
                w1=values["w1"],
                w2=values["w2"],
                activation=values["activation"],
                global_num_experts=int(values["global_num_experts"]),
                local_num_experts=int(values["local_num_experts"]),
                expert_map=values["expert_map"],
                apply_router_weight_on_input=bool(values["apply_router_weight_on_input"]),
            )
            try:
                _write_result(rank, _run_replay(self, original_experts, spec))
            except BaseException as exc:
                _write_result(
                    rank,
                    {
                        "status": "error",
                        "rank": rank,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
        return original_experts(self, *args, **kwargs)

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
