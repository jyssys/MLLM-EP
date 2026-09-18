#!/usr/bin/env python3
"""Bounded GPU screen of pack/sort/count/offset overhead on B1 shapes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_frontier_delta import load_shapes
from scripts.analyze_h1_straggler_deepdive import BatchComputeModel
from virtual_ep.comm_model import CommunicationScenario, matrix_endpoint_bytes


def time_worklist(experts: torch.Tensor, local_experts: int, repeats: int = 30) -> float:
    def operation():
        order = torch.argsort(experts, stable=True)
        counts = torch.bincount(experts[order], minlength=local_experts).to(torch.int32)
        return counts.cumsum(0, dtype=torch.int32)

    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record(); value = operation(); end.record(); end.synchronize()
        if value.numel() != local_experts:
            raise RuntimeError("invalid metadata output")
        samples.append(begin.elapsed_time(end))
    return float(np.median(samples))


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    torch.cuda.set_device(0)
    root = Path("artifacts/frontier_delta_ep")
    comm = CommunicationScenario.load(
        Path("artifacts/virtual_ep/20260916_215010/communication/model.json"),
        "ep2_calibrated_base",
    )
    model_paths = {
        4: Path("artifacts/virtual_ep/20260917_four_hypothesis_discovery/compute_model/ep4_grouped_mm_discovery.csv"),
        8: Path("artifacts/virtual_ep/20260916_215010/compute_model/ep8_grouped_mm.csv"),
    }
    result = {"scope": "critical-rank argsort+bincount+cumsum lower-bound screen",
              "excluded": ["expert GEMMs", "dispatch", "combine", "branch scatter/reduction"]}
    all_overhead, all_stage = 0.0, 0.0
    for ep in (4, 8):
        model = BatchComputeModel(model_paths[ep])
        hist, unique, _requests = load_shapes(root / "block_cached_gsm8_v2", ep)
        width = 256 // ep
        rank_times = np.column_stack([
            model.predict_batch(hist[:, rank * width:(rank + 1) * width])
            for rank in range(ep)
        ])
        expert = rank_times.max(axis=1)
        dispatch, combine = [], []
        for matrix in unique:
            outgoing, incoming = matrix_endpoint_bytes(matrix, 2048)
            dispatch.append(comm.dispatch.latency(outgoing, incoming))
            combine.append(comm.combine.latency(incoming, outgoing))
        stage = np.asarray(dispatch) + expert + np.asarray(combine)
        # Deterministic coverage across stage-cost quantiles.
        order = np.argsort(stage)
        chosen = np.unique(np.linspace(0, len(order) - 1, 96).round().astype(int))
        overhead, selected_stage, selected_expert = [], [], []
        generator = torch.Generator(device="cuda:0").manual_seed(20260918 + ep)
        for position in chosen:
            index = int(order[position])
            critical = int(np.argmax(rank_times[index]))
            counts = hist[index, critical * width:(critical + 1) * width].astype(np.int64)
            ids = np.repeat(np.arange(width, dtype=np.int64), counts)
            if not len(ids):
                continue
            ids_tensor = torch.as_tensor(ids, device="cuda:0")
            permutation = torch.randperm(len(ids), generator=generator, device="cuda:0")
            ids_tensor = ids_tensor[permutation]
            overhead.append(time_worklist(ids_tensor, width))
            selected_stage.append(float(stage[index]))
            selected_expert.append(float(expert[index]))
        overhead = np.asarray(overhead); selected_stage = np.asarray(selected_stage)
        selected_expert = np.asarray(selected_expert)
        entry = {
            "cases": int(len(overhead)),
            "overhead_ms_p50": float(np.percentile(overhead, 50)),
            "overhead_ms_p90": float(np.percentile(overhead, 90)),
            "overhead_share_of_expert_percent": float(100 * overhead.sum() / selected_expert.sum()),
            "overhead_share_of_routed_stage_percent": float(100 * overhead.sum() / selected_stage.sum()),
        }
        result[f"ep{ep}"] = entry
        all_overhead += overhead.sum(); all_stage += selected_stage.sum()
    result["overhead_share_percent"] = float(100 * all_overhead / all_stage)
    result["status"] = "FOLLOWUP_CANDIDATE" if result["overhead_share_percent"] >= 5 else "RETIRE"
    path = root / "worklist_overhead.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
