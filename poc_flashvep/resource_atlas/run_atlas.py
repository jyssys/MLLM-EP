"""Bounded real Qwen3-VL serving run used by the resource-atlas profiler."""
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
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _request(processor: Any, image: Any, suffix: str) -> dict[str, Any]:
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Describe this image briefly. {suffix}"},
        ]}], tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def _rank_worker(rank: int, port: int, args: argparse.Namespace, barrier: Any) -> None:
    out = args.output / f"driver_rank{rank}.json"
    try:
        # Each DP engine is a normal validated TP2/DP2 instance; its EP group
        # spans all four visible devices through vLLM's EP initialization.
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": "1,2,3,4",
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_ATLAS_RESULT_DIR": str(args.output.resolve()),
            "FLASHVEP_ATLAS_DISABLE": "0",
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "FLASHVEP_CUDA_PROFILER_API": "1" if args.cuda_profiler_api else "0",
            "FLASHVEP_PROFILER_WATCH_SECONDS": "240",
        })
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        image = Image.open(IMAGE).convert("RGB")
        request = _request(processor, image, "")
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_dbo=False,
            enable_return_routed_experts=False,
            enable_ep_weight_filter=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30,
            max_model_len=4096,
            max_num_batched_tokens=8192,
            max_num_seqs=2,
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            enforce_eager=True,
            disable_log_stats=False,
        )
        sampling = SamplingParams(max_tokens=args.decode_tokens, temperature=0.0)
        # Wait until both DP owners have initialized their engines.  Warmups
        # are outside the intended measured request, although Nsight capture
        # may include process startup when NVTX capture is unavailable.
        barrier.wait(timeout=900)
        warmup_outputs = []
        for i in range(args.warmups):
            llm._add_completion_requests([copy.deepcopy(_request(processor, image, f"warmup {i}"))], sampling, use_tqdm=False)
            warmup_outputs.extend(llm._run_engine(RequestOutput, use_tqdm=False))
        barrier.wait(timeout=900)
        # The read-only hook in CUDA-owning child workers watches this signal.
        # Starting there avoids model/NCCL/Triton initialization in the trace.
        if args.cuda_profiler_api:
            start_signal = args.output / "cuda_profiler_start.signal"
            stop_signal = args.output / "cuda_profiler_stop.signal"
            start_signal.touch()
            if stop_signal.exists():
                stop_signal.unlink()
        measured_request = copy.deepcopy(request)
        llm._add_completion_requests([measured_request], sampling, use_tqdm=False)
        start = time.perf_counter_ns()
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
        wall_ms = (time.perf_counter_ns() - start) / 1e6
        if args.cuda_profiler_api:
            (args.output / "cuda_profiler_stop.signal").touch()
        barrier.wait(timeout=900)
        out.write_text(json.dumps({
            "ok": True,
            "rank": rank,
            "pid": os.getpid(),
            "physical_visible": [1, 2, 3, 4],
            "warmups": args.warmups,
            "decode_tokens": args.decode_tokens,
            "driver_measured_wall_ms": wall_ms,
            "warmup_output_count": len(warmup_outputs),
            "output_count": len(outputs),
            "output_tokens": [int(t) for o in outputs for t in o.outputs[0].token_ids],
        }, indent=2) + "\n")
    except BaseException as exc:
        out.write_text(json.dumps({"ok": False, "rank": rank,
                                   "pid": os.getpid(), "error": repr(exc)}, indent=2) + "\n")
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--decode-tokens", type=int, default=8)
    p.add_argument("--cuda-profiler-api", action="store_true")
    a = p.parse_args()
    # The profiler creates the result directory before launching this target
    # so that its -o path can live inside it.  Each run uses a fresh timestamp;
    # workers only write their own manifest/proof files.
    a.output.mkdir(parents=True, exist_ok=True)
    for signal in (a.output / "cuda_profiler_start.signal", a.output / "cuda_profiler_stop.signal"):
        if signal.exists():
            signal.unlink()
    repo = Path(__file__).resolve().parents[2]
    hook_dir = repo / "poc_flashvep/resource_atlas/hooks"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    os.environ["PYTHONPATH"] = f"{hook_dir}:{repo}:" + os.environ.get("PYTHONPATH", "")
    a.output.joinpath("workload_manifest.json").write_text(json.dumps({
        "model": a.model, "image": IMAGE,
        "prompt_type": "single_real_image_describe",
        "warmups": a.warmups, "decode_tokens": a.decode_tokens,
        "cuda_profiler_api": a.cuda_profiler_api,
        "configuration": {
            "dtype": "BF16", "tp": 2, "dp": 2, "ep": 4, "pp": 1,
            "backend": "deepep_high_throughput", "triton_experts": True,
            "dbo": False, "prefix_cache": False, "enforce_eager": True,
            "max_num_batched_tokens": 8192, "max_model_len": 4096,
            "physical_gpus": [1, 2, 3, 4],
        },
    }, indent=2) + "\n")
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    port = _port()
    procs = [ctx.Process(target=_rank_worker, args=(r, port, a, barrier)) for r in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    if any(proc.exitcode != 0 for proc in procs):
        raise SystemExit(f"worker failure {[proc.exitcode for proc in procs]}")


if __name__ == "__main__":
    main()
