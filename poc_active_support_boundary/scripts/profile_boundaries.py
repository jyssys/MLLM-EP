#!/usr/bin/env python3
"""NVTX-only observer-heavy reference trace; never use timings as clean E2E."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmark_support_boundary import EXPECTED_UUIDS, VISIBLE, load_real_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE:
        raise SystemExit("wrong CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() != 4:
        raise SystemExit("unexpected logical GPU count")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    if uuid != EXPECTED_UUIDS[0]:
        raise SystemExit("wrong physical GPU UUID")
    case = load_real_cases(args.replay, 2)[args.case]
    counts = list(map(int, case["m_e_vector"]))
    if len(counts) != 64 or not sum(counts):
        raise SystemExit("invalid replay")
    total = sum(counts)
    torch.manual_seed(6)
    x = torch.randn((total, 4096), device=device, dtype=torch.bfloat16) * 0.02
    w1 = torch.randn((64, 4096, 2048), device=device, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn((64, 1024, 4096), device=device, dtype=torch.bfloat16) * 0.01
    weights = torch.rand((total, 1), device=device, dtype=torch.bfloat16)
    offsets = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32)

    def run() -> torch.Tensor:
        with torch.cuda.nvtx.range("gate_up_grouped_mm"):
            intermediate = torch._grouped_mm(x, w1, offs=offsets)
        with torch.cuda.nvtx.range("swiglu_and_materialize"):
            gate, up = intermediate.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
        with torch.cuda.nvtx.range("down_grouped_mm"):
            out = torch._grouped_mm(activated, w2, offs=offsets)
        with torch.cuda.nvtx.range("route_weight_multiply"):
            return out * weights

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("profiling_scope_observer_heavy"):
        for _ in range(args.repeats):
            run()
    torch.cuda.synchronize()
    print(json.dumps({"scope": "observer_heavy_operator_only", "uuid": uuid,
                      "case_id": case["case_id"], "task": case["task"],
                      "phase": case["phase"], "rows": total,
                      "active_experts": sum(bool(n) for n in counts)}))


if __name__ == "__main__":
    main()
