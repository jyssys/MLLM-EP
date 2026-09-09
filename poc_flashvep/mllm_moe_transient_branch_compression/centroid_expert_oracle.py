"""Exact expert replay for centroid and router-weighted-centroid representatives.

This is an observer-heavy oracle.  It loads the original Qwen checkpoint's
expert weights and evaluates one representative input per spatial/route group.
It does not claim a deployable timing result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    cosine_rows, load_layer, modality_and_coords, write_csv, write_json,
)


def load_expert_weights(model: Path, layer: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    names = [f"{prefix}.gate_up_proj", f"{prefix}.down_proj"]
    tensors = []
    for name in names:
        with safe_open(model / index[name], framework="pt", device="cpu") as handle:
            tensors.append(handle.get_tensor(name).to(device=device, dtype=torch.bfloat16))
    return tensors[0], tensors[1]


@torch.inference_mode()
def expert_forward(inputs: torch.Tensor, experts: torch.Tensor,
                   gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    # The released checkpoint stores grouped-MM weights transposed relative to
    # the eager module declaration: gate_up [E,H,2I], down [E,I,H].  Retain a
    # fallback for an eager-style [E,2I,H]/[E,H,I] checkpoint.
    grouped_layout = gate_up.shape[1] == inputs.shape[1]
    hidden_size = down.shape[2] if grouped_layout else down.shape[1]
    output = torch.empty((len(inputs), hidden_size), device=inputs.device, dtype=torch.bfloat16)
    for expert in torch.unique(experts).tolist():
        idx = torch.nonzero(experts == expert, as_tuple=False).flatten()
        gu = inputs[idx] @ gate_up[expert] if grouped_layout else F.linear(inputs[idx], gate_up[expert])
        gate, up = gu.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        output[idx] = intermediate @ down[expert] if grouped_layout else F.linear(intermediate, down[expert])
    return output


def make_groups(data: dict[str, np.ndarray], modality: np.ndarray, coords: np.ndarray,
                kind: str, cap: int) -> list[tuple[int, list[tuple[int, int]]]]:
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    for expert in np.unique(data["ids"]):
        token, slot = np.where((data["ids"] == expert) & (modality[:, None] == "vision"))
        if len(token) < 2:
            continue
        buckets: dict[Any, list[int]] = defaultdict(list)
        if kind == "window_2x2":
            for i, coordinate in enumerate(coords[token]):
                buckets[tuple((coordinate // 2).tolist())].append(i)
        elif kind == "window_2x4":
            for i, coordinate in enumerate(coords[token]):
                buckets[(int(coordinate[0] // 2), int(coordinate[1] // 4))].append(i)
        elif kind == "contiguous_1d":
            order = np.argsort(data["position"][token])
            current = [int(order[0])]
            for before, after in zip(order[:-1], order[1:], strict=True):
                if data["position"][token[after]] == data["position"][token[before]] + 1:
                    current.append(int(after))
                else:
                    buckets[len(buckets)] = current
                    current = [int(after)]
            buckets[len(buckets)] = current
        elif kind in {"hidden_nearest", "output_oracle_nearest"}:
            feature = data["hidden"][token] if kind == "hidden_nearest" else data["outputs"][token, slot]
            normalized = feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
            similarity = normalized @ normalized.T
            remaining = set(range(len(token)))
            while remaining:
                anchor = min(remaining)
                others = sorted(remaining - {anchor}, key=lambda j: -float(similarity[anchor, j]))
                local = [anchor, *others[:cap - 1]]
                buckets[len(buckets)] = local
                remaining.difference_update(local)
        else:
            raise ValueError(kind)
        for local in buckets.values():
            for start in range(0, len(local), cap):
                chunk = local[start:start + cap]
                if len(chunk) >= 2:
                    groups.append((int(expert), [(int(token[i]), int(slot[i])) for i in chunk]))
    return groups


def medoid(group: list[tuple[int, int]], data: dict[str, np.ndarray], feature: str) -> int:
    token = np.asarray([row[0] for row in group])
    slot = np.asarray([row[1] for row in group])
    values = data["hidden"][token] if feature == "hidden" else data["outputs"][token, slot]
    norm = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return int(np.argmax((norm @ norm.T).sum(axis=1)))


def evaluate(data: dict[str, np.ndarray], modality: np.ndarray,
             groups: list[tuple[int, list[tuple[int, int]]]], representative: str,
             gate_up: torch.Tensor, down: torch.Tensor, device: torch.device) -> dict[str, Any]:
    approx = data["outputs"].copy()
    rep_inputs, rep_experts, destinations = [], [], []
    replaced = set()
    saved_rows = 0
    for expert, group in groups:
        tokens = np.asarray([row[0] for row in group])
        # Every group of n inputs is evaluated once, irrespective of whether
        # that representative is an existing token or a newly created centroid.
        saved_rows += len(group) - 1
        if representative == "anchor":
            rep = 0
            output = data["outputs"][group[rep][0], group[rep][1]]
            for token, slot in group[1:]:
                approx[token, slot] = output
                replaced.add((token, slot))
        elif representative in {"hidden_medoid", "output_oracle_medoid"}:
            rep = medoid(group, data, "hidden" if representative == "hidden_medoid" else "output")
            output = data["outputs"][group[rep][0], group[rep][1]]
            for i, (token, slot) in enumerate(group):
                if i != rep:
                    approx[token, slot] = output
                    replaced.add((token, slot))
        else:
            h = data["hidden"][tokens]
            if representative == "router_weighted_centroid":
                weights = np.asarray([data["weights"][token, slot] for token, slot in group])
                h = (h * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
            elif representative == "centroid":
                h = h.mean(axis=0)
            else:
                raise ValueError(representative)
            rep_inputs.append(h)
            rep_experts.append(expert)
            destinations.append(group)
    if rep_inputs:
        h = torch.from_numpy(np.stack(rep_inputs)).to(device=device, dtype=torch.bfloat16)
        e = torch.tensor(rep_experts, device=device, dtype=torch.long)
        outputs = expert_forward(h, e, gate_up, down).float().cpu().numpy()
        for output, group in zip(outputs, destinations, strict=True):
            for token, slot in group:
                approx[token, slot] = output
                replaced.add((token, slot))
    exact = np.einsum("tk,tkh->th", data["weights"], data["outputs"])
    predicted = np.einsum("tk,tkh->th", data["weights"], approx)
    relative = np.linalg.norm(predicted - exact, axis=1) / np.maximum(np.linalg.norm(exact, axis=1), 1e-12)
    cosine = cosine_rows(predicted, exact)
    visual = modality == "vision"
    affected = np.zeros(len(modality), dtype=bool)
    for token, _ in replaced:
        affected[token] = True
    strict = (relative <= .01) & (cosine >= .9999)
    return {
        "groups": len(groups), "branch_reduction": saved_rows / max(int(visual.sum()) * data["ids"].shape[1], 1),
        "affected_token_fraction": float(affected[visual].mean()),
        "visual_rel_l2_median": float(np.median(relative[visual])),
        "visual_rel_l2_p90": float(np.quantile(relative[visual], .9)),
        "visual_cosine_median": float(np.median(cosine[visual])),
        "visual_strict_pass_fraction": float(strict[visual].mean()),
        "affected_rel_l2_median": float(np.median(relative[affected])) if affected.any() else 0.0,
        "affected_rel_l2_p90": float(np.quantile(relative[affected], .9)) if affected.any() else 0.0,
        "affected_cosine_median": float(np.median(cosine[affected])) if affected.any() else 1.0,
        "affected_strict_pass_fraction": float(strict[affected].mean()) if affected.any() else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="4,24,44")
    parser.add_argument("--samples", default="")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.result_dir / "manifest.json").read_text())
    model = Path(manifest["model"])
    wanted_samples = set(filter(None, args.samples.split(",")))
    samples = [row for row in manifest["samples"] if not wanted_samples or row["sample_id"] in wanted_samples]
    device = torch.device(args.device)
    rows, validation = [], []
    for layer in map(int, args.layers.split(",")):
        gate_up, down = load_expert_weights(model, layer, device)
        for meta in samples:
            data = load_layer(args.result_dir, manifest, meta, layer)
            modality, coords = modality_and_coords(meta, data["position"])
            # Check the checkpoint replay against actual vLLM branch outputs.
            visual_token = np.flatnonzero(modality == "vision")
            if len(visual_token):
                token = np.repeat(visual_token, data["ids"].shape[1])
                slot = np.tile(np.arange(data["ids"].shape[1]), len(visual_token))
                h = torch.from_numpy(data["hidden"][token]).to(device=device, dtype=torch.bfloat16)
                e = torch.from_numpy(data["ids"][token, slot].astype(np.int64)).to(device)
                replay = expert_forward(h, e, gate_up, down).float().cpu().numpy()
                captured = data["outputs"][token, slot]
                rel = np.linalg.norm(replay - captured, axis=1) / np.maximum(np.linalg.norm(captured, axis=1), 1e-12)
                cos = cosine_rows(replay, captured)
                validation.append({"sample_id": meta["sample_id"], "layer": layer,
                                   "branches": len(token), "median_rel_l2": float(np.median(rel)),
                                   "p99_rel_l2": float(np.quantile(rel, .99)), "min_cosine": float(cos.min()),
                                   "median_cosine": float(np.median(cos))})
            for kind in ("contiguous_1d", "window_2x2", "window_2x4", "hidden_nearest", "output_oracle_nearest"):
                for cap in (2, 4, 8):
                    groups = make_groups(data, modality, coords, kind, cap)
                    for representative in ("anchor", "hidden_medoid", "output_oracle_medoid", "centroid", "router_weighted_centroid"):
                        rows.append({"sample_id": meta["sample_id"], "category": meta["category"],
                                     "edge": meta["fixed_edge"], "layer": layer, "grouping": kind,
                                     "group_cap": cap, "representative": representative,
                                     **evaluate(data, modality, groups, representative, gate_up, down, device)})
            print(json.dumps({"sample": meta["sample_id"], "layer": layer}), flush=True)
        del gate_up, down
        torch.cuda.empty_cache()
    write_csv(args.output_dir / "centroid_oracle.csv", rows)
    write_csv(args.output_dir / "expert_replay_validation.csv", validation)
    write_json(args.output_dir / "summary.json", {"rows": len(rows), "validation_rows": len(validation),
                                                   "warning": "observer-heavy exact expert replay; no latency claim"})


if __name__ == "__main__":
    main()
