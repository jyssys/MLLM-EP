"""Measure Qwen3-VL EP8-local (16 expert) Triton MoE shapes on one H100.

This is a rank-local kernel calibration only.  It does not create eight
distributed ranks and must never be labelled physical EP8 communication or
full-latency validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"median_ms": float(np.median(array)),
            "p10_ms": float(np.quantile(array, 0.1)),
            "p90_ms": float(np.quantile(array, 0.9))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpus", default="4")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cases", type=int, default=16)
    args = parser.parse_args()
    gpus = [int(item) for item in args.physical_gpus.split(",")]
    if len(gpus) != 1 or gpus[0] not in (4, 5, 6, 7):
        raise ValueError("rank-local calibration requires one allowed GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])

    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    replay = json.loads(args.replay.read_text())
    if int(replay["target_ep"]) != 8:
        raise ValueError("expected an EP8 structural replay")
    candidates = []
    for row in replay["rows"]:
        histogram = np.asarray(row["expert_histogram"], dtype=np.int64)
        if histogram.shape != (128,):
            raise ValueError("expected the Qwen3-VL 128-expert histogram")
        for rank in range(8):
            local = histogram[rank * 16:(rank + 1) * 16]
            if local.sum():
                candidates.append((int(local.sum()), row, rank, local))
    candidates.sort(key=lambda item: item[0])
    indexes = np.unique(np.linspace(0, len(candidates) - 1,
                                    min(args.cases, len(candidates))).round().astype(int))

    torch.manual_seed(3407)
    device = torch.device("cuda:0")
    # Actual Qwen3-VL expert tensor layout: gate/up=[E,1536,2048],
    # down=[E,2048,768].  Random values are sufficient for kernel timing;
    # shapes, BF16 math, routing histogram and installed Triton implementation
    # match the target contract.
    w1 = torch.empty((16, 1536, 2048), device=device, dtype=torch.bfloat16).normal_(0, 0.01)
    w2 = torch.empty((16, 2048, 768), device=device, dtype=torch.bfloat16).normal_(0, 0.01)
    observations = []
    for candidate_index in indexes:
        assignments, replay_row, rank, counts = candidates[int(candidate_index)]
        ids = np.repeat(np.arange(16, dtype=np.int32), counts).reshape(-1, 1)
        topk_ids = torch.from_numpy(ids).to(device=device)
        topk_weights = torch.ones((assignments, 1), device=device, dtype=torch.float32)
        hidden = torch.empty((assignments, 2048), device=device,
                             dtype=torch.bfloat16).normal_(0, 0.02)
        for _ in range(args.warmup):
            fused_experts(hidden, w1, w2, topk_weights, topk_ids, inplace=False)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            output = fused_experts(hidden, w1, w2, topk_weights, topk_ids, inplace=False)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
            del output
        observations.append({
            "request_ids": replay_row["request_ids"], "wave": replay_row["wave"],
            "layer": replay_row["layer"], "virtual_rank": rank,
            "assignments": assignments, "active_local_experts": int(np.count_nonzero(counts)),
            "max_expert_rows": int(counts.max(initial=0)),
            "local_expert_histogram": counts.astype(int).tolist(),
            **_percentiles(samples), "samples_ms": samples,
        })
    payload = {
        "schema_version": "mllm_ep8_local_kernel.v1",
        "status": "LOCAL_KERNEL_CALIBRATED",
        "evidence": "ACTUAL_SINGLE_H100_TRITON_16_LOCAL_EXPERT_BF16",
        "not_evidence_for": ["physical EP8 communication", "physical EP8 E2E latency"],
        "physical_gpu": gpus[0], "gpu_name": torch.cuda.get_device_name(0),
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "shape": {"hidden_size": 2048, "intermediate_size": 768,
                  "local_experts": 16, "global_experts": 128,
                  "expanded_assignment_top_k": 1, "dtype": "bfloat16"},
        "warmup": args.warmup, "repeats": args.repeats,
        "replay": str(args.replay.resolve()), "observations": observations,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "observations"} | {"cases": len(observations)}, indent=2))


if __name__ == "__main__":
    main()
