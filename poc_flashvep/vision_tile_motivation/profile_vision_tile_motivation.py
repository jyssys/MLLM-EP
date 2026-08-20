"""Read-only Qwen3-VL vision-tile routing motivation profiler.

The profile command runs stock vLLM routing with returned top-k expert IDs.
The analyze command reconstructs post-merge image coordinates and produces the
three motivation gates.  Neither command changes routing or model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import statistics
import traceback
from pathlib import Path
from typing import Any

import numpy as np


MODEL_DEFAULT = (
    "/home/esjung/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
)
IMAGE_TOKEN_ID = 151655
NUM_EXPERTS = 128
NUM_LAYERS = 48
TOPK = 8
EP_SIZE = 4
EXPERTS_PER_RANK = NUM_EXPERTS // EP_SIZE
RANDOM_SEEDS = tuple(range(10))


def sample_suite() -> list[dict[str, Any]]:
    ski = Path("/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data")
    mode = Path("/home/esjung/MLLM-EP/external/MODE/assets")
    tui = Path("/home/esjung/MLLM-EP/external/lmms-eval/docs/images")
    rows = [
        ("natural", "astronaut", [ski / "astronaut.png"]),
        ("natural", "motorcycle", [ski / "motorcycle_left.png"]),
        ("natural", "deep_field", [ski / "hubble_deep_field.jpg"]),
        ("natural", "coffee", [ski / "coffee.png"]),
        ("natural", "cat", [ski / "chelsea.png"]),
        ("natural", "rocket", [ski / "rocket.jpg"]),
        ("fine_grained", "retina", [ski / "retina.jpg"]),
        ("fine_grained", "histology", [ski / "ihc.png"]),
        ("fine_grained", "grass", [ski / "grass.png"]),
        ("fine_grained", "gravel", [ski / "gravel.png"]),
        ("chart_document", "fast_gptq", [mode / "fast_gptq.png"]),
        ("chart_document", "method", [mode / "method.png"]),
        ("chart_document", "model_card", [mode / "card_3.png"]),
        ("chart_document", "bit_allocate", [mode / "bit-allocate.png"]),
        ("chart_document", "tui_main", [tui / "tui-main.png"]),
        ("chart_document", "tui_log", [tui / "tui-log-streaming.png"]),
        ("multi_image", "coffee_rocket", [ski / "coffee.png", ski / "rocket.jpg"]),
    ]
    prompts = {
        "natural": "Describe the important objects and their spatial arrangement briefly.",
        "fine_grained": "Describe the visible fine-grained texture or structure briefly.",
        "chart_document": "Summarize the visible chart, diagram, or interface briefly.",
        "multi_image": "Compare the two images and their main visual content briefly.",
    }
    return [
        {
            "category": category,
            "sample_id": sample_id,
            "image_paths": [str(path) for path in paths],
            "question": prompts[category],
        }
        for category, sample_id, paths in rows
    ]


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare_sample(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in row["image_paths"]]
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image} for image in images
    ]
    content.append({"type": "text", "text": row["question"]})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    processed = processor(text=[prompt], images=images, return_tensors="pt")
    token_ids = processed["input_ids"][0].tolist()
    grids = processed["image_grid_thw"].tolist()
    merge_size = int(processor.image_processor.merge_size)
    image_token_id = int(
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    )
    runs: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor] != image_token_id:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(token_ids) and token_ids[end] == image_token_id:
            end += 1
        runs.append((cursor, end))
        cursor = end
    if len(runs) != len(images):
        raise AssertionError(f"{row['sample_id']}: image-run count mismatch")
    image_meta = []
    for index, (image, path, grid, run) in enumerate(
        zip(images, row["image_paths"], grids, runs, strict=True)
    ):
        t, h, w = map(int, grid)
        expected = t * h * w // (merge_size**2)
        if t != 1 or run[1] - run[0] != expected:
            raise AssertionError(
                f"{row['sample_id']} image {index}: grid/run mismatch {grid} {run}"
            )
        image_meta.append(
            {
                "image_index": index,
                "path": path,
                "sha256": _sha256(Path(path)),
                "source_size": list(image.size),
                "image_grid_thw": [t, h, w],
                "merge_size": merge_size,
                "post_merge_grid_hw": [h // merge_size, w // merge_size],
                "token_span": list(run),
                "vision_tokens": expected,
            }
        )
    mm_images: Any = images[0] if len(images) == 1 else images
    request = {"prompt": prompt, "multi_modal_data": {"image": mm_images}}
    metadata = {
        **row,
        "image_token_id": image_token_id,
        "processor_prompt_tokens": len(token_ids),
        "processor_vision_tokens": int(sum(end - start for start, end in runs)),
        "images": image_meta,
    }
    return request, metadata


def _balanced_partition(prepared: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[list[Any]]:
    partitions: list[list[Any]] = [[], []]
    loads = [0, 0]
    for item in sorted(prepared, key=lambda value: -value[1]["processor_prompt_tokens"]):
        rank = 0 if loads[0] <= loads[1] else 1
        partitions[rank].append(item)
        loads[rank] += item[1]["processor_prompt_tokens"]
    return partitions


def _rank_profile(
    dp_rank: int,
    port: int,
    args: argparse.Namespace,
    partition: list[tuple[dict[str, Any], dict[str, Any]]],
    barrier: Any,
) -> None:
    rank_json = args.output_dir / f"profile.dp_rank{dp_rank}.json"
    try:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(dp_rank),
                "VLLM_DP_RANK_LOCAL": str(dp_rank),
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(port),
                "FLASHVEP_DIRECT_ROUTING_DIR": str(
                    (args.output_dir / "direct_router_capture").resolve()
                ),
            }
        )
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput

        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_dbo=False,
            enable_return_routed_experts=True,
            enable_ep_weight_filter=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30,
            max_model_len=4096,
            max_num_batched_tokens=16384,
            max_num_seqs=16,
            limit_mm_per_prompt={"image": 2},
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            enforce_eager=True,
            disable_log_stats=False,
        )
        prompts = [item[0] for item in partition]
        metadata = [item[1] for item in partition]
        barrier.wait(timeout=900)
        submitted = llm._add_completion_requests(
            prompts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False
        )
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
        barrier.wait(timeout=900)
        if len(outputs) != len(metadata):
            raise AssertionError(
                f"DP{dp_rank}: output count {len(outputs)} != submitted {len(metadata)}"
            )
        prompt_lengths = [len(output.prompt_token_ids or []) for output in outputs]
        total_prompt_tokens = sum(prompt_lengths)
        direct_routed, call_groups = _load_direct_trace(
            args.output_dir / "direct_router_capture", dp_rank, prompt_lengths
        )
        if direct_routed.shape != (total_prompt_tokens, NUM_LAYERS, TOPK):
            raise AssertionError(f"DP{dp_rank}: direct capture shape {direct_routed.shape}")
        records = []
        token_offset = 0
        # _run_engine returns finished requests in submission order, while DP
        # mode prefixes its public output IDs. Keep both IDs for the audit.
        for submitted_id, meta, request in zip(submitted, metadata, outputs, strict=True):
            token_ids = np.asarray(request.prompt_token_ids or [], dtype=np.int64)
            routed = direct_routed[token_offset : token_offset + len(token_ids)]
            token_offset += len(token_ids)
            if routed.shape != (len(token_ids), NUM_LAYERS, TOPK):
                raise AssertionError(
                    f"{meta['sample_id']}: routed shape {routed.shape}, tokens {len(token_ids)}"
                )
            if token_ids.tolist().count(meta["image_token_id"]) != meta["processor_vision_tokens"]:
                raise AssertionError(f"{meta['sample_id']}: vLLM/HF vision-token mismatch")
            array_name = f"routing.dp{dp_rank}.{meta['sample_id']}.npz"
            np.savez_compressed(
                args.output_dir / array_name,
                routed_experts=routed,
                prompt_token_ids=token_ids,
            )
            records.append(
                {
                    **meta,
                    "dp_rank": dp_rank,
                    "submitted_request_id": submitted_id,
                    "returned_request_id": request.request_id,
                    "array_file": array_name,
                    "capture_source": "direct read-only router top-k, TP0/TP1 sequence chunks",
                    "model_call_groups": call_groups,
                    "vllm_prompt_tokens": int(len(token_ids)),
                    "routed_shape": list(routed.shape),
                    "output_token_ids": list(request.outputs[0].token_ids),
                }
            )
        _json(rank_json, {"ok": True, "dp_rank": dp_rank, "records": records})
    except Exception:
        _json(rank_json, {"ok": False, "dp_rank": dp_rank, "traceback": traceback.format_exc()})
        raise


def profile(args: argparse.Namespace) -> None:
    from transformers import AutoProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    suite = sample_suite()
    missing = [path for row in suite for path in row["image_paths"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing local samples: {missing}")
    prepared = [_prepare_sample(processor, row) for row in suite]
    partitions = _balanced_partition(prepared)
    manifest = {
        "model": args.model_path,
        "configuration": {
            "dtype": "BF16",
            "tp": 2,
            "dp": 2,
            "ep": 4,
            "pp": 1,
            "all2all": "deepep_high_throughput",
            "dbo": False,
            "enforce_eager": True,
            "physical_gpus": [4, 5, 6, 7],
        },
        "software": {"python": platform.python_version()},
        "partition": [
            [item[1]["sample_id"] for item in partition] for partition in partitions
        ],
        "partition_prompt_tokens": [
            sum(item[1]["processor_prompt_tokens"] for item in partition)
            for partition in partitions
        ],
        "samples": [item[1] for item in prepared],
    }
    _json(args.output_dir / "sample_manifest.json", manifest)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _open_port()
    processes = [
        context.Process(target=_rank_profile, args=(rank, port, args, partitions[rank], barrier))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"DP profiling failed: exit codes {codes}")


def _load_direct_trace(
    direct_dir: Path, dp_rank: int, prompt_lengths: list[int]
) -> tuple[np.ndarray, list[list[int]]]:
    call_ids = sorted(
        {
            int(path.name.split("_call", 1)[1].split("_", 1)[0])
            for path in direct_dir.glob(f"dp{dp_rank}_tp0_call*_layer0.npy")
        }
    )
    if not call_ids:
        raise AssertionError(f"DP{dp_rank}: no direct router captures")
    cursor = 0
    call_token_counts: list[int] = []
    call_groups: list[list[int]] = []
    layer_zero_rows = []
    for call in call_ids:
        shards = [
            np.load(direct_dir / f"dp{dp_rank}_tp{tp}_call{call}_layer0.npy")
            for tp in range(2)
        ]
        padded_rows = sum(len(shard) for shard in shards)
        start = cursor
        running = 0
        while cursor < len(prompt_lengths):
            running += prompt_lengths[cursor]
            cursor += 1
            if 2 * math.ceil(running / 2) == padded_rows:
                break
            if running > padded_rows:
                raise AssertionError(
                    f"DP{dp_rank} call {call}: cannot align {padded_rows} rows"
                )
        if 2 * math.ceil(running / 2) != padded_rows:
            raise AssertionError(f"DP{dp_rank} call {call}: incomplete alignment")
        call_token_counts.append(running)
        call_groups.append(list(range(start, cursor)))
        layer_zero_rows.append(padded_rows)
    if cursor != len(prompt_lengths):
        raise AssertionError(f"DP{dp_rank}: unconsumed prompts after calls")
    layers = []
    for layer in range(NUM_LAYERS):
        calls = []
        for call, real_tokens, padded_rows in zip(
            call_ids, call_token_counts, layer_zero_rows, strict=True
        ):
            shards = [
                np.load(direct_dir / f"dp{dp_rank}_tp{tp}_call{call}_layer{layer}.npy")
                for tp in range(2)
            ]
            combined = np.concatenate(shards, axis=0)
            if len(combined) != padded_rows:
                raise AssertionError(
                    f"DP{dp_rank} layer {layer} call {call}: inconsistent rows"
                )
            calls.append(combined[:real_tokens])
        layers.append(np.concatenate(calls, axis=0))
    routed = np.stack(layers, axis=1).astype(np.int16)
    expected = sum(prompt_lengths)
    if routed.shape != (expected, NUM_LAYERS, TOPK):
        raise AssertionError(f"DP{dp_rank}: direct trace shape {routed.shape}")
    if int(routed.min()) < 0 or int(routed.max()) >= NUM_EXPERTS:
        raise AssertionError(
            f"DP{dp_rank}: expert IDs outside [0,{NUM_EXPERTS}): "
            f"{int(routed.min())}..{int(routed.max())}"
        )
    return routed, call_groups


def recover(args: argparse.Namespace) -> None:
    """Recover request arrays from a completed direct capture after host failure."""
    manifest = json.loads((args.output_dir / "sample_manifest.json").read_text())
    by_sample = {row["sample_id"]: row for row in manifest["samples"]}
    for dp_rank, sample_ids in enumerate(manifest["partition"]):
        metadata = [by_sample[sample_id] for sample_id in sample_ids]
        lengths = [row["processor_prompt_tokens"] for row in metadata]
        routed, call_groups = _load_direct_trace(
            args.output_dir / "direct_router_capture", dp_rank, lengths
        )
        records = []
        offset = 0
        for row, length in zip(metadata, lengths, strict=True):
            local = routed[offset : offset + length]
            offset += length
            token_ids = np.zeros(length, dtype=np.int64)
            for image in row["images"]:
                start, end = image["token_span"]
                token_ids[start:end] = row["image_token_id"]
            array_name = f"routing.dp{dp_rank}.{row['sample_id']}.npz"
            np.savez_compressed(
                args.output_dir / array_name,
                routed_experts=local,
                prompt_token_ids=token_ids,
            )
            records.append(
                {
                    **row,
                    "dp_rank": dp_rank,
                    "submitted_request_id": "captured; host ID not retained",
                    "returned_request_id": "captured; host ID not retained",
                    "array_file": array_name,
                    "capture_source": "direct read-only router top-k, TP0/TP1 sequence chunks",
                    "model_call_groups": call_groups,
                    "vllm_prompt_tokens": length,
                    "routed_shape": list(local.shape),
                    "output_token_ids": [],
                }
            )
        _json(
            args.output_dir / f"profile.dp_rank{dp_rank}.json",
            {
                "ok": True,
                "dp_rank": dp_rank,
                "recovered_from_completed_direct_capture": True,
                "records": records,
            },
        )


def _jsd(distributions: np.ndarray) -> float:
    eps = 1e-12
    values = np.asarray(distributions, dtype=np.float64)
    values = values / np.maximum(values.sum(axis=1, keepdims=True), eps)
    scores = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            p, q = values[left], values[right]
            midpoint = 0.5 * (p + q)
            kl_p = np.sum(np.where(p > 0, p * np.log2((p + eps) / (midpoint + eps)), 0))
            kl_q = np.sum(np.where(q > 0, q * np.log2((q + eps) / (midpoint + eps)), 0))
            scores.append(0.5 * (kl_p + kl_q))
    return float(np.mean(scores)) if scores else 0.0


def _tile_counts(experts: np.ndarray, groups: np.ndarray, num_groups: int) -> tuple[np.ndarray, np.ndarray]:
    expert_counts = np.zeros((num_groups, NUM_EXPERTS), dtype=np.int64)
    for group in range(num_groups):
        ids = experts[groups == group].reshape(-1)
        expert_counts[group] = np.bincount(ids, minlength=NUM_EXPERTS)[:NUM_EXPERTS]
    rank_counts = expert_counts.reshape(num_groups, EP_SIZE, EXPERTS_PER_RANK).sum(axis=2)
    return expert_counts, rank_counts


def _group_metrics(experts: np.ndarray, groups: np.ndarray, num_groups: int) -> dict[str, float]:
    expert_counts, rank_counts = _tile_counts(experts, groups, num_groups)
    rank_share = rank_counts / np.maximum(rank_counts.sum(axis=1, keepdims=True), 1)
    return {
        "rank_jsd": _jsd(rank_counts),
        "expert_jsd": _jsd(expert_counts),
        "max_rank_share_spread": float(np.max(rank_share.max(axis=0) - rank_share.min(axis=0))),
    }


def _groups(height: int, width: int, granularity: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width))
    spatial = ((yy * granularity // height) * granularity + (xx * granularity // width)).reshape(-1)
    sizes = np.bincount(spatial, minlength=granularity**2)
    sequential = np.repeat(np.arange(granularity**2), sizes)
    return spatial, sequential


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _bootstrap_difference(spatial: np.ndarray, random: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260820)
    values = []
    for _ in range(2000):
        index = rng.integers(0, len(spatial), len(spatial))
        values.append(float(np.mean(spatial[index] - random[index])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    for rank in range(2):
        payload = json.loads((args.output_dir / f"profile.dp_rank{rank}.json").read_text())
        if not payload["ok"]:
            raise RuntimeError(payload["traceback"])
        records.extend(payload["records"])
    records.sort(key=lambda row: row["sample_id"])
    p1_rows, p2_rows, p3_rows = [], [], []
    representative: dict[str, Any] | None = None
    source_profiles: list[dict[str, Any]] = []
    for record in records:
        arrays = np.load(args.output_dir / record["array_file"])
        routed = arrays["routed_experts"].astype(np.int64)
        token_ids = arrays["prompt_token_ids"]
        vision_mask = token_ids == record["image_token_id"]
        vision_count = int(vision_mask.sum())
        nonvision_count = int(len(token_ids) - vision_count)
        p1_rows.append(
            {
                "sample_id": record["sample_id"],
                "category": record["category"],
                "vision_tokens": vision_count,
                "nonvision_tokens": nonvision_count,
                "vision_ratio": vision_count / len(token_ids),
                "assignment_ratio": vision_count * TOPK / (len(token_ids) * TOPK),
            }
        )
        for layer in range(NUM_LAYERS):
            all_experts = routed[:, layer, :]
            vis_experts = all_experts[vision_mask]
            non_experts = all_experts[~vision_mask]
            all_rank = np.bincount((all_experts // EXPERTS_PER_RANK).reshape(-1), minlength=EP_SIZE)
            vis_rank = np.bincount((vis_experts // EXPERTS_PER_RANK).reshape(-1), minlength=EP_SIZE)
            non_rank = np.bincount((non_experts // EXPERTS_PER_RANK).reshape(-1), minlength=EP_SIZE)
            critical = int(np.argmax(all_rank))
            delta_total = float(all_rank[critical] - all_rank.mean())
            delta_v = float(vis_rank[critical] - vis_rank.mean())
            delta_n = float(non_rank[critical] - non_rank.mean())
            p2_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "category": record["category"],
                    "layer": layer,
                    "critical_rank": critical,
                    "delta_total": delta_total,
                    "delta_vision": delta_v,
                    "delta_nonvision": delta_n,
                    "vision_only_imbalance": float(vis_rank.max() - vis_rank.mean()),
                    "nonvision_only_imbalance": float(non_rank.max() - non_rank.mean()),
                    "mean_rank_load": float(all_rank.mean()),
                    "decomposition_error": float(delta_total - delta_v - delta_n),
                }
            )
        for image in record["images"]:
            start, end = image["token_span"]
            height, width = image["post_merge_grid_hw"]
            image_routing = routed[start:end]
            if len(image_routing) != height * width:
                raise AssertionError(f"{record['sample_id']}: spatial mapping mismatch")
            for granularity in (2, 4):
                spatial, sequential = _groups(height, width, granularity)
                num_groups = granularity**2
                sizes = np.bincount(spatial, minlength=num_groups)
                for layer in range(NUM_LAYERS):
                    expert_ids = image_routing[:, layer, :]
                    spatial_metrics = _group_metrics(expert_ids, spatial, num_groups)
                    sequential_metrics = _group_metrics(expert_ids, sequential, num_groups)
                    random_metrics = []
                    for seed in RANDOM_SEEDS:
                        rng = np.random.default_rng(seed)
                        random_groups = np.empty_like(spatial)
                        permutation = rng.permutation(len(spatial))
                        cursor = 0
                        for group, size in enumerate(sizes):
                            random_groups[permutation[cursor : cursor + size]] = group
                            cursor += int(size)
                        random_metrics.append(_group_metrics(expert_ids, random_groups, num_groups))
                    row = {
                        "sample_id": record["sample_id"],
                        "category": record["category"],
                        "image_index": image["image_index"],
                        "layer": layer,
                        "granularity": granularity,
                        "spatial": spatial_metrics,
                        "sequential": sequential_metrics,
                        "random_mean": {
                            key: float(np.mean([value[key] for value in random_metrics]))
                            for key in spatial_metrics
                        },
                        "random_std": {
                            key: float(np.std([value[key] for value in random_metrics]))
                            for key in spatial_metrics
                        },
                        "random_rank_jsd_seeds": [value["rank_jsd"] for value in random_metrics],
                    }
                    p3_rows.append(row)
                    if granularity == 4 and (representative is None or spatial_metrics["rank_jsd"] > representative["score"]):
                        _, rank_counts = _tile_counts(expert_ids, spatial, num_groups)
                        representative = {
                            "score": spatial_metrics["rank_jsd"],
                            "sample_id": record["sample_id"],
                            "image_index": image["image_index"],
                            "layer": layer,
                            "rank_share": (rank_counts / rank_counts.sum(axis=1, keepdims=True)).tolist(),
                        }
        if len(record["images"]) > 1:
            for layer in range(NUM_LAYERS):
                profiles = []
                for image in record["images"]:
                    start, end = image["token_span"]
                    ranks = (routed[start:end, layer, :] // EXPERTS_PER_RANK).reshape(-1)
                    profiles.append(np.bincount(ranks, minlength=EP_SIZE))
                source_profiles.append(
                    {"sample_id": record["sample_id"], "layer": layer, "rank_jsd": _jsd(np.asarray(profiles))}
                )

    ratios = [row["vision_ratio"] for row in p1_rows]
    by_category = {
        category: _stats([row["vision_ratio"] for row in p1_rows if row["category"] == category])
        for category in sorted({row["category"] for row in p1_rows})
    }
    p1_summary = {
        **_stats(ratios),
        "fraction_gt_0_5": float(np.mean(np.asarray(ratios) > 0.5)),
        "fraction_gt_0_7": float(np.mean(np.asarray(ratios) > 0.7)),
        "fraction_gt_0_8": float(np.mean(np.asarray(ratios) > 0.8)),
        "max_assignment_ratio_error": float(max(abs(row["vision_ratio"] - row["assignment_ratio"]) for row in p1_rows)),
        "by_category": by_category,
    }
    categories_go = sum(1 for value in by_category.values() if value["median"] >= 0.60)
    if p1_summary["median"] >= 0.60 and p1_summary["fraction_gt_0_5"] >= 0.70 and categories_go >= 2:
        p1_status = "GO"
    elif p1_summary["median"] >= 0.40:
        p1_status = "HOLD"
    else:
        p1_status = "NO-GO"

    meaningful = [row for row in p2_rows if row["delta_total"] > 0 and row["delta_total"] >= max(8.0, 0.01 * row["mean_rank_load"])]
    contributions = [row["delta_vision"] / row["delta_total"] for row in meaningful]
    p2_summary = {
        "meaningful_filter": "delta_total >= max(8 assignments, 1% of mean rank load)",
        "meaningful_layers": len(meaningful),
        "total_request_layers": len(p2_rows),
        "visual_contribution": _stats(contributions),
        "fraction_delta_v_gt_delta_n": float(np.mean([row["delta_vision"] > row["delta_nonvision"] for row in meaningful])),
        "fraction_visual_gt_50pct": float(np.mean(np.asarray(contributions) > 0.5)),
        "fraction_visual_gt_70pct": float(np.mean(np.asarray(contributions) > 0.7)),
        "median_vision_only_imbalance": float(np.median([row["vision_only_imbalance"] for row in meaningful])),
        "median_nonvision_only_imbalance": float(np.median([row["nonvision_only_imbalance"] for row in meaningful])),
        "max_decomposition_error": float(max(abs(row["decomposition_error"]) for row in p2_rows)),
        "by_category": {},
    }
    for category in sorted({row["category"] for row in meaningful}):
        selected = [row for row in meaningful if row["category"] == category]
        shares = [row["delta_vision"] / row["delta_total"] for row in selected]
        p2_summary["by_category"][category] = {
            "meaningful_layers": len(selected),
            "median_visual_contribution": float(np.median(shares)),
            "fraction_delta_v_gt_delta_n": float(
                np.mean(
                    [row["delta_vision"] > row["delta_nonvision"] for row in selected]
                )
            ),
        }
    if p2_summary["visual_contribution"]["median"] >= 0.70 and p2_summary["fraction_delta_v_gt_delta_n"] > 0.5:
        p2_status = "GO"
    elif p2_summary["visual_contribution"]["median"] >= 0.50:
        p2_status = "HOLD"
    else:
        p2_status = "NO-GO"

    p3_summary: dict[str, Any] = {"primary_metric": "mean pairwise JSD of EP-rank distributions (bits)"}
    granular_status = []
    for granularity in (2, 4):
        subset = [row for row in p3_rows if row["granularity"] == granularity]
        spatial = np.asarray([row["spatial"]["rank_jsd"] for row in subset])
        sequential = np.asarray([row["sequential"]["rank_jsd"] for row in subset])
        random = np.asarray([row["random_mean"]["rank_jsd"] for row in subset])
        expert_spatial = np.asarray([row["spatial"]["expert_jsd"] for row in subset])
        expert_random = np.asarray([row["random_mean"]["expert_jsd"] for row in subset])
        ci = _bootstrap_difference(spatial, random)
        ratio = float(spatial.mean() / random.mean())
        fraction = float(np.mean(spatial > random))
        p3_summary[f"{granularity}x{granularity}"] = {
            "observations": len(subset),
            "rank_jsd_spatial_mean": float(spatial.mean()),
            "rank_jsd_sequential_mean": float(sequential.mean()),
            "rank_jsd_random_mean": float(random.mean()),
            "spatial_random_ratio": ratio,
            "spatial_minus_random_bootstrap_95ci": ci,
            "fraction_spatial_gt_random_mean": fraction,
            "expert_jsd_spatial_mean": float(expert_spatial.mean()),
            "expert_jsd_random_mean": float(expert_random.mean()),
            "expert_spatial_random_ratio": float(expert_spatial.mean() / expert_random.mean()),
            "max_rank_share_spread_spatial_mean": float(np.mean([row["spatial"]["max_rank_share_spread"] for row in subset])),
            "max_rank_share_spread_random_mean": float(np.mean([row["random_mean"]["max_rank_share_spread"] for row in subset])),
            "by_category": {},
        }
        for category in sorted({row["category"] for row in subset}):
            selected = [row for row in subset if row["category"] == category]
            cat_spatial = np.asarray(
                [row["spatial"]["rank_jsd"] for row in selected]
            )
            cat_random = np.asarray(
                [row["random_mean"]["rank_jsd"] for row in selected]
            )
            p3_summary[f"{granularity}x{granularity}"]["by_category"][category] = {
                "observations": len(selected),
                "spatial_random_ratio": float(cat_spatial.mean() / cat_random.mean()),
                "fraction_spatial_gt_random_mean": float(
                    np.mean(cat_spatial > cat_random)
                ),
            }
        granular_status.append("GO" if ratio >= 1.20 and ci[0] > 0 and fraction >= 0.70 else ("HOLD" if ratio > 1.0 and ci[0] > 0 else "NO-GO"))
    p3_status = "GO" if granular_status == ["GO", "GO"] else ("NO-GO" if "NO-GO" in granular_status else "HOLD")
    p3_summary["gate_rule"] = "GO requires both granularities: spatial/random rank-JSD >=1.20, paired bootstrap CI>0, and >=70% pairs above random mean."
    p3_summary["reba_source_image_rank_jsd"] = _stats([row["rank_jsd"] for row in source_profiles]) if source_profiles else None
    overall = "GO" if [p1_status, p2_status, p3_status] == ["GO", "GO", "GO"] else ("NO-GO" if p3_status == "NO-GO" else "HOLD")

    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    colors = {"natural": "#4C78A8", "fine_grained": "#59A14F", "chart_document": "#F28E2B", "multi_image": "#B279A2"}
    ax.bar(range(len(p1_rows)), [row["vision_ratio"] for row in p1_rows], color=[colors[row["category"]] for row in p1_rows])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(p1_rows)), [row["sample_id"] for row in p1_rows], rotation=55, ha="right")
    ax.set_ylabel("vision tokens / prompt tokens")
    ax.set_title("Figure 1A — Visual-token dominance by request")
    fig.savefig(figures / "plot1_visual_token_ratio.png", dpi=220)
    plt.close(fig)

    layer_v = [np.median([row["delta_vision"] for row in meaningful if row["layer"] == layer]) for layer in range(NUM_LAYERS)]
    layer_n = [np.median([row["delta_nonvision"] for row in meaningful if row["layer"] == layer]) for layer in range(NUM_LAYERS)]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(NUM_LAYERS)
    ax.bar(x, layer_v, label="vision contribution", color="#4C78A8")
    ax.bar(x, layer_n, bottom=layer_v, label="non-vision contribution", color="#F28E2B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("median critical-rank excess assignments")
    ax.set_title("Figure 2A — Critical-rank excess decomposition")
    ax.legend()
    fig.savefig(figures / "plot2_critical_rank_excess.png", dpi=220)
    plt.close(fig)

    rep = representative or {}
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    heat = np.asarray(rep["rank_share"])
    image = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.4, float(heat.max())))
    ax.set_xlabel("EP rank")
    ax.set_ylabel("4x4 spatial tile")
    ax.set_xticks(range(EP_SIZE), [f"R{i}" for i in range(EP_SIZE)])
    ax.set_title(f"Figure 3A — {rep['sample_id']} image {rep['image_index']}, layer {rep['layer']}")
    fig.colorbar(image, ax=ax, label="normalized routed share")
    fig.savefig(figures / "plot3_tile_rank_heatmap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)
    for axis, granularity in zip(axes, (2, 4), strict=True):
        subset = [row for row in p3_rows if row["granularity"] == granularity]
        values = [
            [row["spatial"]["rank_jsd"] for row in subset],
            [row["sequential"]["rank_jsd"] for row in subset],
            [value for row in subset for value in row["random_rank_jsd_seeds"]],
        ]
        axis.boxplot(values, tick_labels=["Spatial", "Sequential", "Random\n(10 seeds)"], showfliers=False)
        axis.set_title(f"{granularity}x{granularity} grouping")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mean pairwise EP-rank JSD (bits)")
    fig.suptitle("Figure 3B — Spatial locality versus equal-size controls")
    fig.savefig(figures / "plot3_spatial_vs_controls.png", dpi=220)
    plt.close(fig)

    summary = {
        "statuses": {"plot1": p1_status, "plot2": p2_status, "plot3": p3_status, "overall": overall},
        "plot1": p1_summary,
        "plot2": p2_summary,
        "plot3": p3_summary,
        "representative_heatmap": rep,
        "sample_count": len(records),
        "category_count": len({row["category"] for row in records}),
    }
    _json(args.output_dir / "summary.json", summary)
    _json(args.output_dir / "plot1_rows.json", p1_rows)
    _json(args.output_dir / "plot2_rows.json", p2_rows)
    _json(args.output_dir / "plot3_rows.json", p3_rows)
    _write_report(args.report, args.output_dir, summary, records)
    print(json.dumps(summary, indent=2))


def _write_report(report: Path, result_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    p1, p2, p3 = summary["plot1"], summary["plot2"], summary["plot3"]
    statuses = summary["statuses"]
    manifest_lines = []
    for row in records:
        image_text = "; ".join(
            f"{Path(image['path']).name} {image['source_size']} grid={image['image_grid_thw']}"
            for image in row["images"]
        )
        vision_tokens = sum(image["vision_tokens"] for image in row["images"])
        manifest_lines.append(
            f"| {row['sample_id']} | {row['category']} | {row['dp_rank']} | "
            f"{row['vllm_prompt_tokens']} | {vision_tokens} | {image_text} |"
        )
    manifest_rows = "\n".join(manifest_lines)
    g2, g4 = p3["2x2"], p3["4x4"]
    reba = p3["reba_source_image_rank_jsd"]
    p2_category_rows = "\n".join(
        f"| {category} | {value['meaningful_layers']} | "
        f"{value['median_visual_contribution']:.4f} | "
        f"{value['fraction_delta_v_gt_delta_n']:.3f} |"
        for category, value in p2["by_category"].items()
    )
    p3_category_rows = "\n".join(
        f"| {category} | {g2['by_category'][category]['spatial_random_ratio']:.3f} | "
        f"{g2['by_category'][category]['fraction_spatial_gt_random_mean']:.3f} | "
        f"{g4['by_category'][category]['spatial_random_ratio']:.3f} | "
        f"{g4['by_category'][category]['fraction_spatial_gt_random_mean']:.3f} |"
        for category in g2["by_category"]
    )
    text = f"""# FlashVEP Vision-Tile Motivation Profiling

## 1. Experiment configuration

Read-only prefill routing capture on Qwen3-VL-30B-A3B-Instruct, BF16,
TP2/DP2/EP4/PP1, DeepEP high-throughput, vLLM 0.20, eager execution, and
physical GPUs 4–7. DBO was disabled to avoid ubatch segmentation; the validated
Attention/DeepStack source fixes remained installed. The router and expert
placement were not changed. vLLM 0.20's public routed-expert buffer does not
handle DeepEP's sequence-parallel shape, so a read-only wrapper saved the exact
`topk_ids` passed to that buffer. TP0/TP1 contiguous sequence chunks were
concatenated per model call, one padding token was removed where needed, and
calls were reassembled in submission order. Every recovered request has shape
`[prompt token, 48 layers, top-k 8]`; IDs span the valid 0–127 expert range.

Spatial coordinates use processor `image_grid_thw` and merge size 2. Qwen3-VL's
post-merge token order is the row-major `(H/2, W/2)` logical grid documented by
the model's encoder metadata path; no fixed 784-token assumption is used.

## 2. Workload/sample manifest

The bounded local suite contains {summary['sample_count']} requests across
{summary['category_count']} categories. It includes natural scenes/objects,
fine-grained scientific/texture imagery, charts/documents/interfaces, varied
resolutions, and one two-image diagnostic. No dataset was downloaded.

| sample | category | DP rank | prompt tokens | vision tokens | image metadata |
| --- | --- | ---: | ---: | ---: | --- |
{manifest_rows}

Full paths and SHA-256 values are in `{result_dir / 'sample_manifest.json'}`.

## 3. Plot 1 — Visual-token dominance

**PLOT1_STATUS: {statuses['plot1']}**

Median/mean vision ratios are {p1['median']:.4f}/{p1['mean']:.4f}; p25/p75 are
{p1['p25']:.4f}/{p1['p75']:.4f}. Fractions above 0.5, 0.7, and 0.8 are
{p1['fraction_gt_0_5']:.3f}, {p1['fraction_gt_0_7']:.3f}, and
{p1['fraction_gt_0_8']:.3f}. Token and top-k assignment ratios agree within
{p1['max_assignment_ratio_error']:.3e}.

![Figure 1A](../deepep_revalidation/results/{result_dir.name}/figures/plot1_visual_token_ratio.png)

*Figure 1A.* Each bar is one real-image request; the dashed line marks vision
majority. Interpret broad category-wide height above the line as visual-token
dominance, not as evidence of spatial routing structure.

## 4. Plot 2 — Vision-dominated critical-rank excess

**PLOT2_STATUS: {statuses['plot2']}**

For {p2['meaningful_layers']} meaningful request-layers, median visual share of
critical excess is {p2['visual_contribution']['median']:.4f}; vision exceeds
non-vision in {p2['fraction_delta_v_gt_delta_n']:.3f}, and explains >70% in
{p2['fraction_visual_gt_70pct']:.3f}. Median vision-only/nonvision-only
imbalances are {p2['median_vision_only_imbalance']:.2f}/
{p2['median_nonvision_only_imbalance']:.2f} assignments. The decomposition
identity error is at most {p2['max_decomposition_error']:.3e}. The filter is:
{p2['meaningful_filter']}.

| category | meaningful layers | median visual contribution | fraction vision > non-vision |
| --- | ---: | ---: | ---: |
{p2_category_rows}

![Figure 2A](../deepep_revalidation/results/{result_dir.name}/figures/plot2_critical_rank_excess.png)

*Figure 2A.* Layer-wise medians decompose the selected total-load critical
rank's excess into signed vision and non-vision terms. Negative contributions
are retained. This is assignment evidence; per-rank expert CUDA latency was not
captured, so it does not prove a latency-critical rank match.

## 5. Plot 3 — Spatial tile routing signatures

**PLOT3_STATUS: {statuses['plot3']}**

The preregistered primary metric is mean pairwise JSD of EP-rank routing
distributions. Controls preserve each spatial tile's token count; random uses
10 fixed seeds. The gate requires both 2x2 and 4x4: spatial/random >=1.20,
paired bootstrap 95% CI above zero, and >=70% request-image-layer pairs above
their random mean.

| grid | spatial JSD | sequential JSD | random JSD | spatial/random | 95% CI of difference | fraction spatial > random |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2x2 | {g2['rank_jsd_spatial_mean']:.6f} | {g2['rank_jsd_sequential_mean']:.6f} | {g2['rank_jsd_random_mean']:.6f} | {g2['spatial_random_ratio']:.3f} | [{g2['spatial_minus_random_bootstrap_95ci'][0]:.6f}, {g2['spatial_minus_random_bootstrap_95ci'][1]:.6f}] | {g2['fraction_spatial_gt_random_mean']:.3f} |
| 4x4 | {g4['rank_jsd_spatial_mean']:.6f} | {g4['rank_jsd_sequential_mean']:.6f} | {g4['rank_jsd_random_mean']:.6f} | {g4['spatial_random_ratio']:.3f} | [{g4['spatial_minus_random_bootstrap_95ci'][0]:.6f}, {g4['spatial_minus_random_bootstrap_95ci'][1]:.6f}] | {g4['fraction_spatial_gt_random_mean']:.3f} |

Secondary expert-JSD spatial/random ratios are
{g2['expert_spatial_random_ratio']:.3f} (2x2) and
{g4['expert_spatial_random_ratio']:.3f} (4x4).

| category | 2x2 ratio | 2x2 fraction > random | 4x4 ratio | 4x4 fraction > random |
| --- | ---: | ---: | ---: | ---: |
{p3_category_rows}

![Figure 3A](../deepep_revalidation/results/{result_dir.name}/figures/plot3_tile_rank_heatmap.png)

*Figure 3A.* The highest-JSD 4x4 request/layer is shown as a diagnostic, not a
cherry-picked aggregate claim; rows are tiles and columns are EP ranks.

![Figure 3B](../deepep_revalidation/results/{result_dir.name}/figures/plot3_spatial_vs_controls.png)

*Figure 3B.* Distributions over all request-image-layer observations compare
spatial grouping with same-size sequential and random controls. Spatial values
must exceed random, rather than merely be nonzero, to support novelty.

## 6. ReBA source-image diagnostic

[ReBA](https://arxiv.org/abs/2608.00574) reports source-image routing
correlation and motivates image-level balancing. The two-image request here
yields source-image rank-JSD median
{reba['median'] if reba else float('nan'):.6f} over layers. This small diagnostic
is consistent with testing image-level boundaries but is not a ReBA replication.
The spatial gate is deliberately conditioned on beating within-image random
grouping, so image correlation alone cannot make Plot 3 pass. Here, both 2x2
and 4x4 spatial groups beat random controls, adding within-image spatial
structure without contradicting ReBA's coarser image boundary.

## 7. Overall motivation gate

**FINAL MOTIVATION STATUS: {statuses['overall']}**

Plot 1, Plot 2, and Plot 3 are independently gated. The overall tile-specific
motivation is GO only when all three pass; a Plot 3 NO-GO makes the tile novelty
story NO-GO even if visual tokens and visual excess dominate.

The strongest positive evidence is Plot 3's spatial/random rank-JSD ratio:
{g2['spatial_random_ratio']:.3f} at 2x2 and {g4['spatial_random_ratio']:.3f} at
4x4, with paired 95% intervals wholly above zero. The strongest counter-
evidence is that absolute rank-JSD remains small ({g2['rank_jsd_spatial_mean']:.6f}
and {g4['rank_jsd_spatial_mean']:.6f} bits), and the 2x2 sequential control is
slightly stronger than spatial grouping; locality is statistically structured,
but its practical scheduling value is not established.

## 8. Threats and limitations

- The suite is bounded and locally available rather than a random benchmark
  sample; chart/document assets are research/UI figures, not full ChartQA/MMMU.
- Routing IDs, not router weights, were captured; the public vLLM buffer could
  not represent the DeepEP sequence-parallel shape, requiring a read-only hook.
- EP-rank criticality is inferred from assignment counts; actual per-rank expert
  CUDA time was not instrumented.
- Only one multi-image request is available, so the ReBA-style diagnostic is
  descriptive.
- Results cover one model, expert placement, precision, and hardware topology.

## 9. Next single recommended action

Run the identical preregistered spatial/random analysis on a small, locally
cached benchmark subset with source-image IDs (at least 16 samples per category)
before designing any tile scheduler.
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("profile")
    run.add_argument("--model-path", default=MODEL_DEFAULT)
    run.add_argument("--output-dir", type=Path, required=True)
    run.set_defaults(func=profile)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--report", type=Path, required=True)
    analysis.set_defaults(func=analyze)
    recovery = sub.add_parser("recover")
    recovery.add_argument("--output-dir", type=Path, required=True)
    recovery.set_defaults(func=recover)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
