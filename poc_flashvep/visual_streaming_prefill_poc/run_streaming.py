"""Run a bounded real multi-image Qwen3-VL workload for streaming analysis."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
EDGE = 448


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _images() -> list[dict[str, Any]]:
    root = Path("/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data")
    names = ["astronaut.png", "brick.png", "camera.png", "chelsea.png"]
    rows = []
    for name in names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({"sample_id": path.stem, "path": str(path.resolve()),
                     "sha256": _sha(path), "category": "local_skimage"})
    return rows


def _request(processor: Any, images: list[Any], request_id: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, 1):
        content.extend([{"type": "text", "text": f"Image {index}:"},
                        {"type": "image", "image": image}])
    content.append({"type": "text", "text": "Describe the images briefly. Answer with one short sentence."})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    # vLLM's internal request API accepts only prompt and multimodal payload;
    # request_id is carried in the external schedule/control file.
    return {"prompt": prompt, "multi_modal_data": {"image": images}}


def _plan(processor: Any, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image
    loaded = [Image.open(row["path"]).convert("RGB").resize((EDGE, EDGE)) for row in rows]
    specs: list[dict[str, Any]] = []
    # Independent image encodes: one request per image.
    for i, image in enumerate(loaded):
        specs.append({"request_id": f"independent_{i+1}", "images": [i],
                      "kind": "independent_image"})
    specs += [
        {"request_id": "single_control", "images": [0], "kind": "single_control"},
        {"request_id": "multi_2", "images": [0, 1], "kind": "combined_2"},
        {"request_id": "multi_4", "images": [0, 1, 2, 3], "kind": "combined_4"},
        {"request_id": "prefix_1", "images": [0], "kind": "prefix_segment"},
        {"request_id": "prefix_2", "images": [0, 1], "kind": "prefix_segment"},
        {"request_id": "prefix_3", "images": [0, 1, 2], "kind": "prefix_segment"},
        {"request_id": "prefix_4", "images": [0, 1, 2, 3], "kind": "prefix_segment"},
    ]
    prepared = []
    for spec in specs:
        images = [loaded[i] for i in spec["images"]]
        req = _request(processor, images, spec["request_id"])
        prepared.append({**spec, "request": req,
                         "image_count": len(images),
                         "visual_tokens": 196 * len(images)})
    manifest = {"edge": EDGE, "visual_tokens_per_image": 196,
                "images": rows, "request_specs": [
                    {k: v for k, v in spec.items() if k != "request"} for spec in prepared]}
    return prepared, manifest


def _run_rank(rank: int, port: int, args: argparse.Namespace,
              prepared: list[dict[str, Any]], barrier: Any) -> None:
    result = args.output / f"driver_dp{rank}.json"
    try:
        active = args.output / "active_request.txt"
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": "1,2,3,4",
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_STREAMING_RESULT_DIR": str(args.output.resolve()),
            "FLASHVEP_STREAMING_ACTIVE_PATH": str(active.resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput
        from vllm.v1.engine import EngineCoreRequestType
        from transformers import AutoProcessor
        from PIL import Image
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=2,
                  enable_expert_parallel=True, expert_placement_strategy="linear",
                  all2all_backend="deepep_high_throughput", enable_dbo=False,
                  enable_ep_weight_filter=True, trust_remote_code=True,
                  gpu_memory_utilization=.90, kv_cache_memory_bytes=1 << 30,
                  max_model_len=4096, max_num_batched_tokens=8192, max_num_seqs=1,
                  limit_mm_per_prompt={"image": 8}, skip_mm_profiling=True,
                  mm_processor_cache_gb=0, enable_prefix_caching=False,
                  enable_flashinfer_autotune=False, enforce_eager=True)
        rows = _images()
        loaded = [Image.open(row["path"]).convert("RGB").resize((EDGE, EDGE)) for row in rows]
        sampling = SamplingParams(max_tokens=args.decode_tokens, temperature=0.0)
        barrier.wait(timeout=900)
        all_records: list[dict[str, Any]] = []
        sequence = [{"request_id": f"warmup_{i}", "images": [0], "kind": "warmup"}
                    for i in range(args.warmups)]
        # Repeating the fixed plan in one persistent engine gives paired
        # timing samples without changing the workload or model state.
        for _ in range(args.repetitions):
            sequence.extend(prepared)
        for wave, spec in enumerate(sequence):
            active.write_text(spec["request_id"] + "\n", encoding="utf-8")
            barrier.wait(timeout=900)
            if rank == 0:
                request = _request(processor, [loaded[i] for i in spec["images"]], spec["request_id"])
                llm._add_completion_requests([copy.deepcopy(request)], sampling, use_tqdm=False)
            else:
                llm.llm_engine.engine_core._send_input(
                    EngineCoreRequestType.START_DP_WAVE, (wave, -1))
            start = time.perf_counter_ns()
            outputs = llm._run_engine(RequestOutput, use_tqdm=False)
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            if rank == 0 and not spec["kind"] == "warmup":
                if len(outputs) != 1:
                    raise AssertionError((spec, len(outputs)))
                output = outputs[0]
                all_records.append({**{k: v for k, v in spec.items() if k != "request"},
                                    "repetition": (wave - args.warmups) // len(prepared),
                                    "wall_ms": wall_ms,
                                    "output_token_ids": [int(t) for t in output.outputs[0].token_ids],
                                    "output_text": output.outputs[0].text,
                                    "prompt_tokens": int(len(output.prompt_token_ids))})
            barrier.wait(timeout=900)
        result.write_text(json.dumps({"ok": True, "rank": rank, "pid": os.getpid(),
                                      "records": all_records}, indent=2) + "\n")
    except BaseException as exc:
        result.write_text(json.dumps({"ok": False, "rank": rank, "error": repr(exc)}, indent=2) + "\n")
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--decode-tokens", type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    hook_dir = repo / "poc_flashvep/visual_streaming_prefill_poc/hooks"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    os.environ["PYTHONPATH"] = f"{hook_dir}:{repo}:" + os.environ.get("PYTHONPATH", "")
    rows = _images()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    prepared, manifest = _plan(processor, rows)
    manifest.update({"model": args.model, "warmups": args.warmups,
                     "decode_tokens": args.decode_tokens, "repetitions": args.repetitions,
                     "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                                       "pp": 1, "backend": "deepep_high_throughput",
                                       "triton_experts": True, "dbo": False,
                                       "prefix_cache": False, "enforce_eager": True,
                                       "physical_gpus": [1, 2, 3, 4],
                                       "max_num_batched_tokens": 8192,
                                       "max_model_len": 4096}})
    (args.output / "workload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "active_request.txt").write_text("init\n")
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    port = _port()
    procs = [ctx.Process(target=_run_rank, args=(rank, port, args, prepared, barrier))
             for rank in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    if any(proc.exitcode != 0 for proc in procs):
        raise SystemExit(f"worker failure {[proc.exitcode for proc in procs]}")


if __name__ == "__main__":
    main()
