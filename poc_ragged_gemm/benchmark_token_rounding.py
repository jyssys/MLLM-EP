#!/usr/bin/env python3
"""Bounded SonicMoE token-rounding counterfactual on fixed random router scores."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from triton.testing import do_bench


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("sonic_token_rounding_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def assignment_set(tokens: torch.Tensor, experts: torch.Tensor) -> set[tuple[int, int]]:
    return set(zip(tokens.cpu().tolist(), experts.cpu().tolist(), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonic-root", type=Path, default=Path("/home/esjung/external/sonic-moe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 2048])
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"4", "5", "6", "7"}:
        raise RuntimeError("requires one physical GPU 4--7")
    ref = load_reference(args.sonic_root / "benchmarks" / "moe-token-rounding.py")
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs
    torch.manual_seed(1100); torch.cuda.manual_seed_all(1100)
    model = MoE(128, 8, 2048, 768, ActivationType.SWIGLU, False, 0.02).cuda().to(torch.bfloat16)
    w1 = model.c_fc.weight.permute(1, 2, 0); w2 = model.c_proj.weight.permute(1, 2, 0)
    router = model.router.weight.detach()
    rows = []
    for tokens in args.tokens:
        x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
        routes = {}
        for method in ("top_k", "nr", "up", "down"):
            if method == "top_k":
                scores, token_ids, expert_ids = ref.forward_topk(x, router, 128, 8)
            else:
                scores, token_ids, expert_ids = ref.forward_token_choice_rounding(x, router, 128, 8, 128, method)
            routes[method] = (scores, token_ids, expert_ids)
            call = lambda: moe_general_routing_inputs(
                x, scores, token_ids, expert_ids, w1, None, w2, None, 128, None,
                ActivationType.SWIGLU, True,
            )
            call(); torch.cuda.synchronize()
            latency = float(do_bench(call, warmup=20, rep=100))
            counts = torch.bincount(expert_ids, minlength=128).cpu().tolist()
            rows.append({
                "tokens": tokens, "method": method, "latency_ms": latency,
                "assignments": len(expert_ids), "counts": counts,
                "effective_tiles": int(sum((x + 127) // 128 for x in counts if x)),
            })
        original = assignment_set(routes["top_k"][1], routes["top_k"][2])
        for row in rows:
            if row["tokens"] != tokens:
                continue
            current = assignment_set(routes[row["method"]][1], routes[row["method"]][2])
            changed = len(original.symmetric_difference(current)) / 2
            row["routing_edit_fraction_of_original"] = changed / len(original)
    top = {(r["tokens"]): r for r in rows if r["method"] == "top_k"}
    for row in rows:
        row["speedup_vs_topk"] = top[row["tokens"]]["latency_ms"] / row["latency_ms"]
        row["tile_reduction_fraction"] = 1 - row["effective_tiles"] / top[row["tokens"]]["effective_tiles"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": str(args.sonic_root), "rows": rows}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
