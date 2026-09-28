from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .contract import NUM_EXPERTS, NUM_LAYERS, WINDOWS, decode_window, read_jsonl, write_jsonl


def _flatten(values: list[list[Any]]) -> list[Any]:
    return [item for row in values for item in row]


def _sparse(vector: np.ndarray) -> dict[str, float]:
    return {str(index): float(value) for index, value in enumerate(vector) if value != 0}


def dense(value: dict[str, float], width: int = NUM_EXPERTS) -> np.ndarray:
    output = np.zeros(width, dtype=np.float64)
    for key, item in value.items():
        output[int(key)] = float(item)
    return output


def consolidate(trace_dir: Path, results_path: Path, topology: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = {row["request_id"]: row for row in read_jsonl(results_path)}
    raw = []
    for path in sorted(trace_dir.glob("route_summary.pid*.jsonl")):
        raw.extend(read_jsonl(path))
    raw = [row for row in raw if row.get("request_id") in results and 0 <= int(row.get("layer", -1)) < NUM_LAYERS]
    ep_size = 2 if topology == "ep2" else 4

    pref: dict[tuple[str, int], dict[str, np.ndarray]] = defaultdict(lambda: {
        "vision_count": np.zeros(NUM_EXPERTS),
        "vision_mass": np.zeros(NUM_EXPERTS),
        "text_count": np.zeros(NUM_EXPERTS),
        "text_mass": np.zeros(NUM_EXPERTS),
    })
    dec: dict[tuple[str, str, int], dict[str, np.ndarray]] = defaultdict(lambda: {
        "count": np.zeros(NUM_EXPERTS),
        "mass": np.zeros(NUM_EXPERTS),
        "source_expert": np.zeros((ep_size, NUM_EXPERTS)),
        "steps": np.zeros(257),
    })
    observed_ep_world_sizes = set()
    invalid_ids = 0
    sampled_routes: dict[tuple[str, int, int], dict[str, Any]] = {}

    for row in raw:
        request = str(row["request_id"])
        layer = int(row["layer"])
        observed_ep_world_sizes.add(int(row.get("ep_world_size", -1)))
        if row.get("record_type") == "prefill" or row.get("decode_step") is None:
            target = pref[(request, layer)]
            for target_key, source_key in (
                ("vision_count", "vision_expert_histogram"),
                ("vision_mass", "vision_expert_weight_sum"),
                ("text_count", "text_expert_histogram"),
                ("text_mass", "text_expert_weight_sum"),
            ):
                target[target_key] += np.asarray(row[source_key], dtype=np.float64)
            continue
        step = int(row["decode_step"])
        if not 1 <= step <= 256:
            continue
        ids = [int(value) for value in _flatten(row.get("route_ids", []))]
        weights = [float(value) for value in _flatten(row.get("route_weights", []))]
        valid_pairs = [(expert, weight) for expert, weight in zip(ids, weights) if 0 <= expert < NUM_EXPERTS]
        invalid_ids += sum(not 0 <= expert < NUM_EXPERTS for expert in ids)
        source = int(row.get("source_ep_rank", -1))
        if not 0 <= source < ep_size:
            raise ValueError(f"invalid source rank {source} for {topology}")
        if int(row.get("tp_rank", -1)) == 0:
            key = (request, step, layer)
            route_ids = [expert for expert, _ in valid_pairs]
            route_weights = [weight for _, weight in valid_pairs]
            prior = sampled_routes.get(key)
            value = {
                "step": step,
                "layer": layer,
                "route_ids": route_ids,
                "route_weights": route_weights,
                "source_ep_rank": source,
            }
            if prior is not None and prior != value:
                raise ValueError(f"non-deterministic duplicate sampled route {key}")
            sampled_routes[key] = value
        for window in (decode_window(step), "1-256"):
            target = dec[(request, window, layer)]
            target["steps"][step] = 1
            for expert, weight in valid_pairs:
                target["count"][expert] += 1
                target["mass"][expert] += weight
                target["source_expert"][source, expert] += 1

    routes_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(sampled_routes):
        routes_by_request[key[0]].append(sampled_routes[key])
    compact = []
    for request, meta in sorted(results.items()):
        layers = []
        for layer in range(NUM_LAYERS):
            value = pref.get((request, layer))
            if value is None:
                value = {key: np.zeros(NUM_EXPERTS) for key in ("vision_count", "vision_mass", "text_count", "text_mass")}
            layers.append({key: _sparse(vector) for key, vector in value.items()})
        windows = {}
        for window in WINDOWS:
            layer_rows = []
            observed_steps = set()
            for layer in range(NUM_LAYERS):
                value = dec.get((request, window, layer))
                if value is None:
                    value = {
                        "count": np.zeros(NUM_EXPERTS),
                        "mass": np.zeros(NUM_EXPERTS),
                        "source_expert": np.zeros((ep_size, NUM_EXPERTS)),
                        "steps": np.zeros(257),
                    }
                observed_steps.update(np.flatnonzero(value["steps"]).tolist())
                layer_rows.append({
                    "count": _sparse(value["count"]),
                    "mass": _sparse(value["mass"]),
                    "source_expert": [_sparse(vector) for vector in value["source_expert"]],
                })
            windows[window] = {
                "observed_steps": sorted(observed_steps),
                "layers": layer_rows,
            }
        compact.append({
            "schema_version": 1,
            "request_id": request,
            "domain": meta["domain"],
            "analysis_split": meta["analysis_split"],
            "topology": topology,
            "ep_size": ep_size,
            "output_token_count": meta["output_token_count"],
            "finish_reason": meta["finish_reason"],
            "vision_token_count": meta["vision_token_count"],
            "prompt_token_count": meta["prompt_token_count"],
            "prefill_layers": layers,
            "decode_windows": windows,
            "sampled_routes": routes_by_request.get(request, []),
        })
    audit = {
        "schema_version": 1,
        "topology": topology,
        "expected_ep_world_size": ep_size,
        "observed_ep_world_sizes": sorted(observed_ep_world_sizes),
        "raw_rows": len(raw),
        "requests": len(compact),
        "requests_with_48_prefill_layers": sum(
            all(bool(layer["vision_count"]) and bool(layer["text_count"]) for layer in row["prefill_layers"])
            for row in compact
        ),
        "invalid_expert_ids": invalid_ids,
        "silent_fallback": observed_ep_world_sizes != {ep_size},
    }
    return compact, audit


def save_compact(rows: list[dict[str, Any]], audit: dict[str, Any], output: Path, audit_path: Path) -> None:
    write_jsonl(output, rows)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
