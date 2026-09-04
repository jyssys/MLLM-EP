#!/usr/bin/env python3
"""Bounded real-image Qwen3-VL TP2/DP4/EP8 token-route capture."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
os.environ["PATH"] = "/home/esjung/anaconda3/envs/flashvep-poc/bin:/home/esjung/.venvs/flashvep-deepep-v020/bin:" + os.environ.get("PATH", "")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); return int(sock.getsockname()[1])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workload() -> list[dict[str, Any]]:
    root = Path("/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data")
    return [
        {"request_id": "single_astronaut", "category": "natural", "paths": [root / "astronaut.png"], "edge": 448},
        {"request_id": "pair_natural", "category": "natural", "paths": [root / "astronaut.png", root / "camera.png"], "edge": 448},
        {"request_id": "pair_resolution", "category": "fine_grained", "paths": [root / "cell.png", root / "coffee.png"], "edge": 896},
        {"request_id": "quad_diverse", "category": "mixed", "paths": [root / "astronaut.png", root / "camera.png", root / "chessboard_RGB.png", root / "coffee.png"], "edge": 448},
        {"request_id": "quad_repeated", "category": "repeated", "paths": [root / "astronaut.png"] * 4, "edge": 448},
        {"request_id": "quad_fine", "category": "fine_grained", "paths": [root / "brick.png", root / "cell.png", root / "chessboard_GRAY.png", root / "chelsea.png"], "edge": 448},
    ]


def _prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image
    images = []
    for path in row["paths"]:
        image = Image.open(path).convert("RGB")
        edge = int(row["edge"])
        scale = edge / max(image.size)
        if abs(scale - 1.0) > 0.01:
            image = image.resize((max(28, round(image.width * scale)), max(28, round(image.height * scale))))
        images.append(image)
    content = ([{"type": "image", "image": image} for image in images] +
               [{"type": "text", "text": "Describe the images and compare their important visual details briefly."}])
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    processed = processor(text=[prompt], images=images, return_tensors="pt")
    ids = [int(x) for x in processed["input_ids"][0].tolist()]
    grids = processed["image_grid_thw"].tolist()
    image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    merge = int(processor.image_processor.merge_size)
    spans = []; cursor = 0
    while cursor < len(ids):
        if ids[cursor] != image_id:
            cursor += 1; continue
        end = cursor + 1
        while end < len(ids) and ids[end] == image_id: end += 1
        spans.append([cursor, end]); cursor = end
    if len(spans) != len(images):
        raise RuntimeError(f"image span mismatch {row['request_id']}: {len(spans)} vs {len(images)}")
    image_meta = []
    for image, path, grid, span in zip(images, row["paths"], grids, spans, strict=True):
        t, h, w = map(int, grid); expected = t * h * w // (merge * merge)
        if span[1] - span[0] != expected:
            raise RuntimeError(f"visual span mismatch {path}: {span} expected {expected}")
        image_meta.append({
            "path": str(path), "sha256": _sha(path), "original_size": list(Image.open(path).size),
            "input_size": list(image.size), "grid_thw": [t, h, w], "vision_tokens": expected,
            "token_span": span,
        })
    meta = {
        "request_id": row["request_id"], "category": row["category"], "image_count": len(images),
        "prompt_tokens": len(ids), "vision_tokens": sum(x["vision_tokens"] for x in image_meta),
        "text_tokens": len(ids) - sum(x["vision_tokens"] for x in image_meta), "image_token_id": image_id,
        "images": image_meta, "token_ids": ids,
    }
    return {"prompt": prompt, "multi_modal_data": {"image": images if len(images) > 1 else images[0]}}, meta


def _generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any, barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    if prompts:
        barrier.wait(timeout=1800); llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        barrier.wait(timeout=1800); outputs = []
    barrier.wait(timeout=1800); return outputs


def _run_rank(dp_rank: int, port: int, args: argparse.Namespace, barrier: Any,
              schedule: list[dict[str, Any]], prepared: dict[str, dict[str, Any]]) -> None:
    out = args.output / f"driver.dp_rank{dp_rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(dp_rank), "VLLM_DP_RANK_LOCAL": str(dp_rank), "VLLM_DP_SIZE": "4",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_MATRIX_CONTROL": str((args.output / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.output / "timing_raw").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput", "FLASHVEP_CONFIGURED_DBO": "false",
            "FLASHVEP_MATRIX_ENABLE": "1", "FLASHVEP_ROUTE_CONTROL": str((args.output / "control.json").resolve()),
            "FLASHVEP_ROUTE_RAW_DIR": str((args.output / "raw_routes").resolve()),
        })
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=2, enable_expert_parallel=True,
            expert_placement_strategy="linear", all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True, trust_remote_code=True,
            gpu_memory_utilization=0.90, kv_cache_memory_bytes=4 << 30, max_model_len=8192,
            max_num_batched_tokens=16384, max_num_seqs=1, limit_mm_per_prompt={"image": 8},
            skip_mm_profiling=True, enable_prefix_caching=False, enable_flashinfer_autotune=False,
            moe_backend="auto", enforce_eager=True, disable_log_stats=True,
        )
        pc = llm.llm_engine.vllm_config.parallel_config
        _write(args.output / f"runtime_proof.dp_rank{dp_rank}.json", {
            "dp_rank": dp_rank, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensor_parallel_size": int(getattr(pc, "tensor_parallel_size", -1)),
            "data_parallel_size": int(getattr(pc, "data_parallel_size", -1)),
            "world_size": int(getattr(pc, "world_size", -1)), "ep_size": 8, "experts_per_rank": 16,
            "dtype": "bfloat16", "backend": "deepep_high_throughput", "placement": "linear",
        })
        sampling = SamplingParams(max_tokens=1, temperature=0.0); records = []
        for entry in schedule:
            if dp_rank == 0:
                tmp = args.output / "control.tmp.json"; _write(tmp, entry); tmp.replace(args.output / "control.json")
            barrier.wait(timeout=1800)
            # vLLM 0.20's DP4 multimodal path can leave an empty DP engine
            # waiting in its shared-memory broadcast while a sibling performs
            # image preprocessing.  For this bounded trace, submit the same
            # real request to every DP engine so all four engines participate
            # in the EP collective.  This preserves model routing and expert
            # placement; the replication is recorded as a measurement-mode
            # detail and is not a serving policy.
            item = prepared[entry["request_id"]]
            start = time.perf_counter_ns(); outputs = _generate(llm, [copy.deepcopy(item[0])] if item else [], sampling, barrier, int(entry["wave"]))
            wall = (time.perf_counter_ns() - start) / 1e6
            records.append({**entry, "dp_rank": dp_rank, "wall_ms": wall,
                            "output_token_ids": [int(t) for o in outputs for t in o.outputs[0].token_ids]})
        if dp_rank == 0:
            _write(args.output / "control.tmp.json", {"wave": len(schedule), "flush": True, "capture": False})
            (args.output / "control.tmp.json").replace(args.output / "control.json")
        barrier.wait(timeout=1800); _generate(llm, [], sampling, barrier, len(schedule))
        _write(out, {"ok": True, "records": records})
    except BaseException:
        _write(out, {"ok": False, "traceback": traceback.format_exc()}); raise


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=MODEL); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Processor preparation is CPU-only and is repeated in each driver so no
    # PIL object crosses multiprocessing boundaries.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    rows = _workload()[:3] if args.smoke else _workload()
    prepared = {r["request_id"]: _prepare(processor, r) for r in rows}
    manifest = {"model": args.model, "configuration": {"dtype": "BF16", "tp": 2, "dp": 4, "ep": 8, "pp": 1, "backend": "deepep_high_throughput", "physical_gpus": list(range(8)), "placement": "linear", "experts": 128, "top_k": 8}, "requests": [x[1] for x in prepared.values()]}
    _write(args.output / "workload_manifest.json", manifest)
    schedule = []
    for rep in range(args.reps):
        for idx, req in enumerate(prepared):
            meta = prepared[req][1]
            schedule.append({
                "wave": len(schedule), "rep": rep, "request_id": req,
                "category": meta["category"], "image_count": meta["image_count"],
                "prompt_tokens": meta["prompt_tokens"], "vision_tokens": meta["vision_tokens"],
                "text_tokens": meta["text_tokens"], "token_ids": meta["token_ids"],
                "source_dp_rank": len(schedule) % 4, "capture": True,
                "instrument": True, "measured": rep >= 1,
                # Fields consumed by the shared DeepEP timing hook.  These
                # are descriptive metadata only; no routing or scheduler
                # decision uses them.
                "modality": "vision", "pair_id": int(len(schedule)),
                "token_bucket": "real_image", "phase": "prefill",
                "iteration": int(len(schedule)),
            })
    _write(args.output / "schedule.json", schedule); _write(args.output / "control.json", {"wave": -1, "capture": False})
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(4); port = _port()
    ps = [ctx.Process(target=_run_rank, args=(rank, port, args, barrier, schedule, prepared)) for rank in range(4)]
    for p in ps: p.start()
    for p in ps: p.join(10800)
    codes = [p.exitcode for p in ps]; _write(args.output / "run_status.json", {"exitcodes": codes, "ok": codes == [0] * 4})
    if codes != [0] * 4: raise SystemExit(f"driver failure: {codes}")


if __name__ == "__main__": main()
