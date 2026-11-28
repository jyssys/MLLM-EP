"""Audit Qwen2.5-VL local windows, join boundaries and causal LM cuts on CPU."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from .cohorts import require
from ..mllm_epd_prefill_heavy_modality_discovery.datasets import file_digest, freeze


def shape(grid, cfg):
    t, h, w = grid
    patch, merge, temporal = (cfg[key] for key in ("patch_size", "spatial_merge_size", "temporal_patch_size"))
    merger_side = cfg["window_size"] // merge // patch
    require(h % merge == w % merge == 0, f"Invalid spatial merge geometry {grid}")
    hm, wm = h // merge, w // merge
    window_sizes, windows, permutation = [], [], []
    packed = 0
    for frame in range(t):
        for i in range(0, hm, merger_side):
            for j in range(0, wm, merger_side):
                bottom, right = min(hm, i + merger_side), min(wm, j + merger_side)
                labels = [(frame * hm + row) * wm + col for row in range(i, bottom) for col in range(j, right)]
                patches = len(labels) * merge * merge
                windows.append({
                    "frame": frame, "merge_grid_rows": [i, bottom], "merge_grid_cols": [j, right],
                    "packed_patch_span": [packed, packed + patches], "patches": patches,
                })
                packed += patches
                permutation.extend(labels)
                window_sizes.append(patches)
    patches = t * h * w
    require(packed == patches and len(permutation) == t * hm * wm
            and sorted(permutation) == list(range(t * hm * wm)), f"Missing or repeated patch rows {grid}")
    projection_elements = cfg["in_chans"] * temporal * patch * patch
    boundary_row_bytes = 2 * cfg["hidden_size"]
    bf16_input_row_bytes = 2 * projection_elements
    byte_counts = {
        "processor_FP32_patch_bytes": patches * 4 * projection_elements,
        "vision_BF16_patch_input_bytes": patches * bf16_input_row_bytes,
        "vision_BF16_hidden_at_first_full_attention_bytes": patches * boundary_row_bytes,
        "LM_ready_BF16_embedding_bytes": len(permutation) * 2 * cfg["out_hidden_size"],
    }
    boundaries = [0] + [win["packed_patch_span"][1] for win in windows]
    partitions = []
    for ratio in (.25, .5, .75):
        index = min(range(len(boundaries)), key=lambda i: (abs(boundaries[i] - ratio * patches), i))
        selected = boundaries[index]
        partitions.append({
            "target_patch_fraction": ratio, "actual_patch_fraction": selected / patches,
            "remote_windows": [0, index], "remote_patch_span": [0, selected],
            "remote_input_BF16_bytes": selected * bf16_input_row_bytes,
            "remote_post_local_blocks_BF16_return_bytes": selected * boundary_row_bytes,
            "remote_local_attention_pairs_per_block": sum(n * n for n in window_sizes[:index]),
        })
    result = {
        "grid_thw": list(grid), "processed_wh": [w * patch, h * patch], "patches": patches,
        "merged_rows": len(permutation), "local_window_count": len(windows), "local_windows": windows,
        "local_attention_pairs_per_local_block": sum(n * n for n in window_sizes),
        "full_attention_pairs_per_full_block": t * (h * w) ** 2,
        "full_attention_sequences": [h * w] * t,
        "local_window_patch_histogram": Counter(window_sizes),
        "byte_counts": byte_counts, "complete_window_partition_geometry": partitions,
        "ordered_merged_indices_SHA256": hashlib.sha256(",".join(map(str, permutation)).encode()).hexdigest(),
    }
    return result, permutation, boundaries[1:]


def native_window_function(source, cfg):
    """Obtain the installed method and run it solely on Torch CPU tensors."""
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    module = ast.parse(source.read_text(), filename=str(source))
    methods = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == "get_window_index_thw"]
    method, = methods
    globals_ = {"torch": torch, "F": functional}
    compiled = compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec")
    exec(compiled, globals_)
    context = SimpleNamespace(**cfg, spatial_merge_unit=cfg["spatial_merge_size"] ** 2)
    function = globals_[method.name]
    return lambda grid: function(context, *grid)


def causal_cuts(row, tokenizer):
    import numpy as np

    with np.load(row["token_metadata_file"], allow_pickle=False) as data:
        native_grid = data["image_grid_thw"].reshape(-1, 3).tolist()
        token_ids = data["input_ids"].reshape(-1)
        require(native_grid == row["grid_thw"] and len(token_ids) == row["total_input_tokens"],
                f"Processor tensor or image metadata differs for {row['request_id']}")
        cuts, previous = [], 0
        for start, end in row["image_visual_spans"]:
            require(previous < start < end < len(token_ids), f"Invalid original image span {row['request_id']}")
            require(token_ids[start - 1] == tokenizer["vision_start_token_id"]
                    and token_ids[end] == tokenizer["vision_end_token_id"]
                    and bool(np.all(token_ids[start:end] == tokenizer["image_token_id"])),
                    f"Original image span is not an exact image-token interval {row['request_id']}")
            cuts.append(end + 1)
            previous = end
        return cuts


def upper_summary(values):
    sorted_values = sorted(values)
    return {"min": sorted_values[0], "p50": median(sorted_values),
            "p95_upper": sorted_values[((len(values) - 1) * 95 + 99) // 100],
            "max": sorted_values[-1]}


def audit(name, rows, shapes, model):
    geometries, associations, windows, patches = Counter(), [], [], []
    max_shape = None
    max_original_area = None
    byte_counts = defaultdict(list)
    cut_distributions = defaultdict(list)
    exact_common_cuts = None
    for row in rows:
        raw_cuts = causal_cuts(row, model)
        cohort_row = {
            "request_id": row["request_id"], "input_tokens": row["total_input_tokens"],
            "visual_tokens": row["visual_tokens"], "image_count": row["image_count"],
            "images": [],
        }
        exact_common_cuts = set(raw_cuts) if exact_common_cuts is None else exact_common_cuts & set(raw_cuts)
        cut_distributions["first"].append(raw_cuts[0])
        cut_distributions["last"].append(raw_cuts[-1])
        for target in (1024, 2048, 4096, 8192):
            right = next((end for end in raw_cuts if end >= target), None)
            left = next((end for end in reversed(raw_cuts) if end <= target), None)
            if right is not None:
                cut_distributions[f"{target}_next_delta"].append(right - target)
            if left is not None:
                cut_distributions[f"{target}_previous_delta"].append(target - left)
        request_bytes = Counter()
        for i, (image, prefix) in enumerate(zip(row["images"], raw_cuts, strict=True)):
            grid = tuple(image["grid_thw"])
            geometry = shapes[grid]
            require(row["grid_thw"][i] == list(grid) and image["processed_resolution"] == geometry["processed_wh"]
                    and image["visual_tokens"] == geometry["merged_rows"]
                    and image["visual_span"] == row["image_visual_spans"][i],
                    f"Per-image geometry differs for {row['request_id']} image {i}")
            geometries[grid] += 1
            windows.extend(window["patches"] for window in geometry["local_windows"])
            patches.append(geometry["patches"])
            request_bytes.update(geometry["byte_counts"])
            item = {
                "image_index": i, "SHA256": image["SHA256"],
                "grid_thw": list(grid), "original_wh": image["original_resolution"],
                "visual_span": image["visual_span"], "after_original_vision_end_prefix_tokens": prefix,
            }
            cohort_row["images"].append(item)
            candidate = item | {"request_id": row["request_id"]}
            if max_shape is None or geometry["patches"] > shapes[tuple(max_shape["grid_thw"])]["patches"]:
                max_shape = candidate
            w, h = image["original_resolution"]
            if max_original_area is None or w * h > max_original_area["original_pixel_area"]:
                max_original_area = candidate | {"original_pixel_area": w * h}
        for key, value in request_bytes.items():
            byte_counts[key].append(value)
        associations.append(cohort_row)
    return {
        "cohort": name, "requests": len(rows), "image_occurrences": len(patches),
        "distinct_grids": len(geometries), "local_window_occurrences": len(windows),
        "patches_per_image": upper_summary(patches), "patches_per_local_window": upper_summary(windows),
        "local_window_patch_histogram": Counter(windows),
        "largest_processed_image": max_shape, "largest_source_image_by_pixel_area": max_original_area,
        "causal_LM_prefix_counts": {key: upper_summary(values) | {"covered_requests": len(values)}
                                    for key, values in cut_distributions.items()},
        "exact_prefix_end_shared_by_every_request": sorted(exact_common_cuts),
        "per_request_bytes": {key: upper_summary(values) for key, values in byte_counts.items()},
        "request_to_grid_and_prefix": associations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--runtime-vision-source", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.model_config.read_bytes())
    cfg = model["vision_config"]
    boundary = min(cfg["fullatt_block_indexes"])
    require(boundary == 7, f"Pinned vision architecture changed to first full attention at {boundary}")
    rosters = {path.stem: json.loads(path.read_bytes())["requests"] for path in args.manifest}
    grids = sorted({tuple(grid) for rows in rosters.values() for row in rows for grid in row["grid_thw"]})
    actual_function = native_window_function(args.runtime_vision_source, cfg)
    shapes = {}
    for grid in grids:
        value, permutation, ends = shape(grid, cfg)
        actual, actual_ends = actual_function(grid)
        require(permutation == actual.tolist() and ends == actual_ends.tolist(), f"Runtime windows differ for {grid}")
        shapes[grid] = value
    freeze(args.out, {
        "evidence": "CPU_SOURCE_AND_PROCESSOR_VERIFIED",
        "model_config_SHA256": file_digest(args.model_config),
        "runtime_vision_source_SHA256": file_digest(args.runtime_vision_source),
        "vision_depth": cfg["depth"], "full_image_attention_blocks_zero_based": cfg["fullatt_block_indexes"],
        "separable_local_vision_blocks_before_first_global_boundary": boundary,
        "grid_geometry_witnesses_matching_installed_cpu_window_method": len(shapes),
        "scope": "Recorded original processor inputs and exact local-window packing. Byte counts are logical "
                 "tensor sizes. Cutting at block 7 requires all ordered original patch activations for "
                 "the first full-image attention. No GPU measurement, live scheduling or latency is inferred.",
        "cohorts": [audit(name, rows, shapes, model) for name, rows in rosters.items()],
        "unique_grid_window_assignments": list(shapes.values()),
    })


if __name__ == "__main__":
    main()
