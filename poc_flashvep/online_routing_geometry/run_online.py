"""Bounded real-vLLM online trace driver.

Two synchronized DP workers submit varied request batches to the V1 engine.
The stock scheduler performs the continuous-batching work; the local hook
records every FusedMoE invocation and its actual top-k route geometry.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path


def _port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return int(p)


def _prompt(model: str, modality: str, text_tokens: int, image_size: int, fill: str, image_path: str | None = None):
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    if modality == "vision":
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (image_size, image_size), (96, 128, 160))
        content = [{"type": "image", "image": image}, {"type": "text", "text": "Describe the image and explain two details."}]
    else:
        content = [{"type": "text", "text": (" " + fill) * max(1, text_tokens)}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    out = {"prompt": text}
    if modality == "vision": out["multi_modal_data"] = {"image": image}
    return out


def _worker(dp_rank: int, args, barrier):
    os.environ.update({"VLLM_DP_RANK": str(dp_rank), "VLLM_DP_RANK_LOCAL": str(dp_rank),
                       "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
                       "VLLM_DP_MASTER_PORT": str(args.port),
                       "FLASHVEP_ONLINE_CONTEXT": args.regime})
    # Multiprocessing start methods can initialize the interpreter before the
    # environment is populated by the parent.  Install the local observer
    # explicitly as a defensive second path; this still only monkey-patches
    # the worker-side vLLM class and never changes model math.
    try:
        import sitecustomize
        if hasattr(sitecustomize, "install"):
            sitecustomize.install()
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Path(args.out, f"hook_import_error.dp{dp_rank}.txt").write_text(repr(exc), encoding="utf-8")
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    model = args.model
    real_images = [
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/astronaut.png",
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/motorcycle_left.png",
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/coffee.png",
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/rocket.jpg",
    ]
    templates = [
        _prompt(model, "text", 220, 448, "blue"),
        _prompt(model, "text", 560, 448, "red"),
        _prompt(model, "vision", 180, 448, "green", real_images[0]),
        _prompt(model, "vision", 320, 896, "yellow", real_images[1]),
    ]
    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=2,
              enable_expert_parallel=True, expert_placement_strategy="linear",
              all2all_backend="deepep_high_throughput", enable_dbo=False,
              enable_return_routed_experts=False, enable_ep_weight_filter=True,
              trust_remote_code=True, kv_cache_memory_bytes=1073741824,
              max_model_len=2048, max_num_batched_tokens=args.max_batched_tokens,
              max_num_seqs=16, skip_mm_profiling=True, enable_prefix_caching=False,
              enable_flashinfer_autotune=False, moe_backend="auto",
              enforce_eager=True, disable_log_stats=False)
    parallel = llm.llm_engine.vllm_config.parallel_config
    proof = {"dp_rank": dp_rank, "tp": int(parallel.tensor_parallel_size),
             "dp": int(parallel.data_parallel_size), "ep": 4,
             "all2all_backend": str(parallel.all2all_backend),
             "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "physical_gpus": [1, 2, 3, 4]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, f"topology.dp{dp_rank}.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    barrier.wait(600)
    # Global warmup: same shape, then varied online waves.  Each wave goes
    # through the normal V1 request queue and is intentionally not replay.
    for _ in range(args.warmups):
        prompts = [copy.deepcopy(templates[2])]
        llm.generate(prompts, sampling, use_tqdm=False)
    waves = []
    for wave in range(args.waves):
        local_n = max(1, args.concurrency // 2)
        if wave % 4 == 0: slots = [2] * local_n
        elif wave % 4 == 1: slots = [0, 1][:local_n]
        elif wave % 4 == 2: slots = [3, 2, 1, 0][:local_n]
        else: slots = [1, 3, 2, 0][:local_n]
        prompts = [copy.deepcopy(templates[i]) for i in slots]
        barrier.wait(600)
        t0 = time.perf_counter_ns()
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        t1 = time.perf_counter_ns()
        waves.append({"wave": wave, "local_requests": local_n, "wall_ms": (t1-t0)/1e6,
                      "prompt_tokens": [len(o.prompt_token_ids) for o in outs],
                      "outputs": [[int(x) for x in o.outputs[0].token_ids] for o in outs]})
        barrier.wait(600)
    Path(args.out, f"waves.dp{dp_rank}.json").write_text(json.dumps(waves, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--waves", type=int, default=12)
    ap.add_argument("--warmups", type=int, default=3); ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--max-batched-tokens", type=int, default=8192); ap.add_argument("--regime", default="mixed_online")
    args = ap.parse_args(); args.port = _port()
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(2)
    ps = [ctx.Process(target=_worker, args=(r, args, barrier)) for r in range(2)]
    for p in ps: p.start()
    for p in ps: p.join(3600)
    for p in ps:
        if p.is_alive(): p.terminate(); p.join(10)
    status = {"exit_codes": [p.exitcode for p in ps], "trace_dir": args.out}
    Path(args.out, "driver_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if all(x == 0 for x in status["exit_codes"]) else 1)


if __name__ == "__main__": main()
