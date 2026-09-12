#!/usr/bin/env python3
"""Compute an independent checkpoint-level reference for layer-1 routed MoE."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path("/home/esjung/models/LLaDA2.0-flash-744c3f8")
RESULTS = Path(
    "/home/esjung/MLLM-EP-llada2-flash-100b/poc_llada2_flash_ep/results"
)


def main() -> None:
    route = torch.load(
        RESULTS / "moe_debug_ep_input/ep_layer_01_routes.pt", map_location="cpu"
    )
    x = route["moe_input"].to("cuda", dtype=torch.bfloat16)
    ids = route["ids"].long()
    weights = route["weights"].float()
    out = torch.zeros_like(x, dtype=torch.float32)
    out_by_rank = torch.zeros((4, *x.shape), device=x.device, dtype=torch.float32)

    weight_map = json.loads((ROOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    needed_keys = []
    for expert in sorted(set(ids.flatten().tolist())):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        needed_keys.extend(
            [
                prefix + ".gate_proj.weight",
                prefix + ".up_proj.weight",
                prefix + ".down_proj.weight",
            ]
        )
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(ROOT / shard, framework="pt", device="cpu")
            )
            for shard in sorted({weight_map[key] for key in needed_keys})
        }

        def get_tensor(key: str) -> torch.Tensor:
            return handles[weight_map[key]].get_tensor(key)

        for expert in sorted(set(ids.flatten().tolist())):
            token, slot = torch.where(ids == expert)
            xe = x[token.to("cuda")]
            prefix = f"model.layers.1.mlp.experts.{expert}"
            gate = get_tensor(prefix + ".gate_proj.weight").to("cuda")
            up = get_tensor(prefix + ".up_proj.weight").to("cuda")
            w2 = get_tensor(prefix + ".down_proj.weight").to("cuda")
            branch = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(xe, gate))
                * torch.nn.functional.linear(xe, up),
                w2,
            )
            scaled = branch.float() * weights[token, slot].to("cuda")[:, None]
            out.index_add_(0, token.to("cuda"), scaled)
            out_by_rank[expert // 64].index_add_(0, token.to("cuda"), scaled)
            del gate, up, w2, branch, scaled, xe

    torch.save(out.cpu(), RESULTS / "moe_debug_ep_input/layer_01_reference_routed.pt")
    torch.save(
        out_by_rank.cpu(),
        RESULTS / "moe_debug_ep_input/layer_01_reference_routed_by_rank.pt",
    )
    for name, path in (
        ("tp", RESULTS / "moe_debug_tp/tp_layer_01_moe.pt"),
        ("ep", RESULTS / "moe_debug_ep_input/ep_layer_01_moe.pt"),
    ):
        got = torch.load(path, map_location="cpu")["routed"].float()
        ref = out.cpu()
        rel = (got - ref).norm() / ref.norm()
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
        print(
            f"{name}: rel_l2={rel.item():.8f} cosine={cos.item():.8f} "
            f"max_abs={(got-ref).abs().max().item():.8f} "
            f"norm={got.norm().item():.8f} ref_norm={ref.norm().item():.8f}"
        )


if __name__ == "__main__":
    main()
