"""Bounded real 2-image image-ready -> partial-prefill prototype.

The fixed request uses a text gap so the first 256-token scheduler chunk ends
before image 2.  The worker hook launches image 2's real encoder on a side
stream and inserts an event wait only when the second image range is reached.
Baseline and streaming requests share one persistent vLLM engine and are
interleaved to avoid cross-process/model-load timing comparisons.
"""
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

from PIL import Image

MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
EDGE = 448


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _images() -> list[dict[str, Any]]:
    root = Path("/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data")
    out = []
    for name in ("astronaut.png", "brick.png"):
        path = root / name
        out.append({"sample_id": path.stem, "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return out


def _request(processor: Any, images: list[Image.Image]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Image 1:"}, {"type": "image", "image": images[0]},
        # Keep image 2 outside the first 256-token chunk while preserving one
        # fixed contiguous token order for baseline and streaming.
        {"type": "text", "text": " filler" * 60},
        {"type": "text", "text": "Image 2:"}, {"type": "image", "image": images[1]},
        {"type": "text", "text": "Describe the images briefly. Answer with one short sentence."},
    ]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": images}}


def _run_rank(rank: int, port: int, args: argparse.Namespace,
              processor: Any, images: list[Image.Image], barrier: Any) -> None:
    result = args.output / f"driver_dp{rank}.json"
    try:
        active_path = args.output / "active_request.txt"
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": "1,2,3,4",
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_STREAMING_RESULT_DIR": str(args.output.resolve()),
            "FLASHVEP_STREAMING_ACTIVE_PATH": str(active_path.resolve()),
            "FLASHVEP_REAL_STREAMING": "1",
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        # vLLM worker processes are forked from this rank process after it has
        # already started.  Explicitly install the fixed handoff wrapper here
        # (in addition to sitecustomize) so the monkey-patch is inherited by
        # every CUDA worker; relying only on interpreter-startup environment
        # flags misses forked workers.
        from poc_flashvep.visual_streaming_prefill_poc.hooks.real_streaming_hook import install as install_real
        install_real()
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput
        from vllm.v1.engine import EngineCoreRequestType

        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_ep_weight_filter=True, trust_remote_code=True,
            gpu_memory_utilization=.90, kv_cache_memory_bytes=1 << 30,
            max_model_len=4096, max_num_batched_tokens=256, max_num_seqs=1,
            limit_mm_per_prompt={"image": 8}, skip_mm_profiling=True,
            mm_processor_cache_gb=0, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True,
        )
        sampling = SamplingParams(max_tokens=args.decode_tokens, temperature=0.0)
        barrier.wait(timeout=900)
        specs = [{"request_id": f"warmup_{i}", "mode": "warmup"}
                 for i in range(max(1, args.warmups))]
        for rep in range(args.repetitions):
            specs.extend([
                {"request_id": f"baseline_2_r{rep}", "mode": "baseline"},
                {"request_id": f"streaming_2_r{rep}", "mode": "streaming"},
            ])
        records: list[dict[str, Any]] = []
        for wave, spec in enumerate(specs):
            active_path.write_text(spec["request_id"] + "\n", encoding="utf-8")
            barrier.wait(timeout=900)
            if rank == 0:
                request = _request(processor, images)
                llm._add_completion_requests([copy.deepcopy(request)], sampling,
                                             use_tqdm=False)
            else:
                llm.llm_engine.engine_core._send_input(
                    EngineCoreRequestType.START_DP_WAVE, (wave, -1))
            start = time.perf_counter_ns()
            outputs = llm._run_engine(RequestOutput, use_tqdm=False)
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            if rank == 0 and spec["mode"] != "warmup":
                if len(outputs) != 1:
                    raise AssertionError((spec, len(outputs)))
                output = outputs[0]
                records.append({
                    **spec, "repetition": wave - 1,
                    "wall_ms": wall_ms,
                    "output_token_ids": [int(t) for t in output.outputs[0].token_ids],
                    "output_text": output.outputs[0].text,
                    "prompt_tokens": int(len(output.prompt_token_ids)),
                })
            barrier.wait(timeout=900)
        result.write_text(json.dumps({"ok": True, "rank": rank,
                                      "pid": os.getpid(), "records": records}, indent=2) + "\n")
    except BaseException as exc:
        result.write_text(json.dumps({"ok": False, "rank": rank,
                                      "error": repr(exc)}, indent=2) + "\n")
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repetitions", type=int, default=20)
    ap.add_argument("--decode-tokens", type=int, default=8)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    hook_dir = repo / "poc_flashvep/visual_streaming_prefill_poc/hooks"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    os.environ["PYTHONPATH"] = f"{hook_dir}:{repo}:" + os.environ.get("PYTHONPATH", "")
    # sitecustomize is imported before _run_rank executes in spawned
    # interpreters, so publish this flag in the parent before workers start.
    os.environ["FLASHVEP_REAL_STREAMING"] = "1"
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    images = [Image.open(row["path"]).convert("RGB").resize((EDGE, EDGE))
              for row in _images()]
    # Validate the intended first-chunk boundary before starting GPUs.
    prompt = _request(processor, images)["prompt"]
    encoded = processor(text=[prompt], images=images, return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    spans = []
    for pos, tok in enumerate(ids):
        if tok == 151655:
            if not spans or pos > spans[-1][-1] + 1:
                spans.append([pos])
            else:
                spans[-1].append(pos)
    if len(spans) != 2 or spans[1][0] <= 256:
        raise RuntimeError(f"image-2 is not beyond first chunk: len={len(ids)} spans={[(x[0],x[-1]) for x in spans]}")
    manifest = {
        "model": args.model, "edge": EDGE, "visual_tokens_per_image": 196,
        "image_paths": _images(), "prompt_tokens": len(ids),
        "image_spans": [[x[0], x[-1]] for x in spans],
        "first_chunk_budget": 256, "warmups": args.warmups,
        "repetitions": args.repetitions, "decode_tokens": args.decode_tokens,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                           "pp": 1, "backend": "deepep_high_throughput",
                           "triton_experts": True, "dbo": False,
                           "prefix_cache": False, "enforce_eager": True,
                           "physical_gpus": [1, 2, 3, 4],
                           "max_num_batched_tokens": 256,
                           "max_model_len": 4096},
        "modes": {"baseline": "stock full multimodal encoder; budget chunking",
                  "streaming": "image1 sync; image2 side-stream encoder; event wait before image2 range"},
    }
    (args.output / "workload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "active_request.txt").write_text("init\n")
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    port = _port()
    procs = [ctx.Process(target=_run_rank, args=(rank, port, args, processor, images, barrier))
             for rank in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    if any(proc.exitcode != 0 for proc in procs):
        raise SystemExit(f"worker failure {[proc.exitcode for proc in procs]}")


if __name__ == "__main__":
    main()
