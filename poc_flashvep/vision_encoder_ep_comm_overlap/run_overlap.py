"""Run a bounded real Qwen3-VL worker hook on physical GPUs 1--4."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path
from typing import Any

MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
IMAGE = "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/astronaut.png"


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])


def _run_rank(rank: int, port: int, args: argparse.Namespace, barrier: Any) -> None:
    out = args.output / f"driver_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "FLASHVEP_VISION_OVERLAP_RESULT_DIR": str(args.output.resolve()),
            "FLASHVEP_DEEPEP_CAPTURE_PATH": str(args.capture.resolve()),
            "FLASHVEP_VISION_OVERLAP_LAYER": str(args.layer),
            "FLASHVEP_VISION_OVERLAP_BATCH": str(args.batch),
            "FLASHVEP_VISION_OVERLAP_WARMUPS": str(args.warmups),
            "FLASHVEP_VISION_OVERLAP_ITERATIONS": str(args.iterations),
        })
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        image = Image.open(IMAGE).convert("RGB")
        prompt = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image briefly."},
        ]}], tokenize=False, add_generation_prompt=True)
        request = {"prompt": prompt, "multi_modal_data": {"image": image}}
        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=4096,
            max_num_batched_tokens=8192, max_num_seqs=2,
            skip_mm_profiling=True, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True,
            disable_log_stats=False,
        )
        from vllm.outputs import RequestOutput
        from vllm.v1.engine import EngineCoreRequestType
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        # One warmup request reaches the real vision block and layer-24 hook.
        barrier.wait(timeout=900)
        # Both DP engines receive a real image request so all four EP ranks
        # reach the one-shot replay hook; this is still a normal DP2 serving
        # path, not a synthetic communication-only invocation.
        if rank in (0, 1):
            llm._add_completion_requests([copy.deepcopy(request)], sampling, use_tqdm=False)
        begin = time.perf_counter_ns()
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
        elapsed = (time.perf_counter_ns() - begin) / 1e6
        barrier.wait(timeout=900)
        out.write_text(json.dumps({"ok": True, "rank": rank, "driver_wall_ms": elapsed,
                                   "output_count": len(outputs),
                                   "output_tokens": [int(t) for o in outputs for t in o.outputs[0].token_ids]}, indent=2) + "\n")
    except BaseException as exc:
        out.write_text(json.dumps({"ok": False, "rank": rank, "error": repr(exc)}, indent=2) + "\n")
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capture", type=Path, default=Path("/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt"))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    hook_dir = repo / "poc_flashvep/vision_encoder_ep_comm_overlap/hooks"
    os.environ["PYTHONPATH"] = f"{hook_dir}:{repo}:" + os.environ.get("PYTHONPATH", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    a.output.joinpath("workload_manifest.json").write_text(json.dumps({
        "model": a.model, "image": IMAGE, "capture": str(a.capture),
        "batch_equivalent": a.batch, "layer": a.layer,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                           "pp": 1, "backend": "deepep_high_throughput",
                           "dbo": False, "physical_gpus": [1,2,3,4]},
        "warmups": a.warmups, "iterations": a.iterations,
    }, indent=2) + "\n")
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(2); port = _port()
    procs = [ctx.Process(target=_run_rank, args=(r, port, a, barrier)) for r in range(2)]
    for q in procs: q.start()
    for q in procs: q.join()
    if any(q.exitcode != 0 for q in procs): raise SystemExit(f"worker failure {[q.exitcode for q in procs]}")


if __name__ == "__main__": main()
