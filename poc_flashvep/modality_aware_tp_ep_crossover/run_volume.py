"""Bounded real-image TP-only versus true DeepEP volume/modality runner.

The TP-only case is a single TP4 engine.  The DeepEP case launches two
independent vLLM DP workers, each with TP2; DP0 receives the real request and
DP1 participates in the empty DP wave so the EP4 collective is exercised.
No routing, placement, or model computation is changed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
SKIMAGE = "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data"
IMAGE_NAMES = ["astronaut.png", "camera.png", "coffee.png", "chelsea.png"]
TEXT = "Explain the topic with concrete mechanisms, assumptions, and edge cases. "


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _text_for_tokens(processor: Any, target: int) -> str:
    # Fixed repeated text is deterministic.  Search a bounded repetition count
    # so text-only controls land near the intended volume.
    lo, hi = 1, max(2, target // 4 + 16)
    while lo < hi:
        mid = (lo + hi) // 2
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": TEXT * mid}]}],
            tokenize=False, add_generation_prompt=True,
        )
        n = len(processor.tokenizer.encode(prompt))
        if n < target:
            lo = mid + 1
        else:
            hi = mid
    return TEXT * max(1, lo)


def _prompt(processor: Any, images: list[Image.Image], text_tokens: int) -> str:
    text = _text_for_tokens(processor, max(16, text_tokens))
    return processor.apply_chat_template(
        [{"role": "user", "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": text},
        ]}], tokenize=False, add_generation_prompt=True,
    )


def make_requests(model: str) -> list[dict[str, Any]]:
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    originals = [Image.open(Path(SKIMAGE) / name).convert("RGB") for name in IMAGE_NAMES]
    # Volume levels deliberately vary resolution/image count as well as text.
    # Actual expanded counts are recorded after vLLM tokenization and are the
    # analysis axis; labels are fixed before measuring.
    specs = [
        ("small", "text_heavy", 0, 256, 0),
        ("medium", "text_heavy", 0, 1024, 0),
        ("large", "text_heavy", 0, 4096, 0),
        ("small", "mixed", 1, 96, 448),
        ("medium", "mixed", 2, 512, 896),
        ("large", "mixed", 4, 1024, 1120),
        ("small", "vision_heavy", 1, 32, 448),
        ("medium", "vision_heavy", 2, 64, 896),
        ("large", "vision_heavy", 6, 96, 1120),
    ]
    requests: list[dict[str, Any]] = []
    for volume, workload, image_count, text_tokens, resolution in specs:
        imgs = [originals[i % len(originals)].resize((resolution, resolution))
                for i in range(image_count)] if image_count else []
        if imgs:
            prompt = _prompt(processor, imgs, text_tokens)
            item: dict[str, Any] = {
                "prompt": prompt, "multi_modal_data": {"image": imgs},
            }
        else:
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": _text_for_tokens(processor, text_tokens)}]}],
                tokenize=False, add_generation_prompt=True,
            )
            item = {"prompt": prompt}
        requests.append({
            "request_id": f"{volume}_{workload}", "volume": volume,
            "workload": workload, "modality": workload,
            "image_count": image_count, "resolution": resolution,
            "text_target_tokens": text_tokens, "request": item,
        })
    return requests


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _set_env(out: Path, control: Path, raw: Path, dp_rank: int | None = None, port: int | None = None) -> None:
    os.environ["FLASHVEP_CROSSOVER_ENABLE"] = "1"
    os.environ["FLASHVEP_CROSSOVER_CONTROL"] = str(control.resolve())
    os.environ["FLASHVEP_CROSSOVER_RAW_DIR"] = str(raw.resolve())
    if dp_rank is not None:
        os.environ.update({
            "VLLM_DP_RANK": str(dp_rank), "VLLM_DP_RANK_LOCAL": str(dp_rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
        })


def _llm_kwargs(model: str, topology: str) -> dict[str, Any]:
    ep = topology == "real_deepep"
    return {
        "model": model, "dtype": "bfloat16", "trust_remote_code": True,
        "expert_placement_strategy": "linear", "enable_expert_parallel": ep,
        "enable_ep_weight_filter": ep, "enable_return_routed_experts": False,
        "all2all_backend": "deepep_high_throughput" if ep else "allgather_reducescatter",
        "moe_backend": "triton", "enable_dbo": False,
        "gpu_memory_utilization": 0.90, "kv_cache_memory_bytes": 1 << 30,
        "max_model_len": 12288, "max_num_batched_tokens": 12288,
        "max_num_seqs": 2, "skip_mm_profiling": True,
        "mm_processor_cache_gb": 0, "enable_prefix_caching": False,
        "enable_flashinfer_autotune": False, "enforce_eager": True,
        "disable_log_stats": True,
    }


def _proof(llm: Any, requested: dict[str, Any], topology: str, dp_rank: int | None) -> dict[str, Any]:
    pc = llm.llm_engine.vllm_config.parallel_config
    fields = ("tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size",
              "enable_expert_parallel", "all2all_backend", "use_sequence_parallel_moe",
              "enable_dbo")
    return {
        "requested": requested, "topology": topology, "dp_rank": dp_rank,
        "physical_gpus": [4, 5, 6, 7], "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parallel_config": {key: _jsonable(getattr(pc, key, None)) for key in fields},
        "world_size": _jsonable(getattr(pc, "world_size", None)),
        "source_semantics": "use_all2all_kernels = dp_size > 1 and use_ep",
        "hook": "poc_flashvep.modality_aware_tp_ep_crossover.worker_hook",
    }


def _run_one(llm: Any, req: dict[str, Any], sampling: Any) -> tuple[Any, float]:
    t0 = time.perf_counter_ns()
    result = llm.generate([copy.deepcopy(req["request"])], sampling, use_tqdm=False)[0]
    return result, (time.perf_counter_ns() - t0) / 1e6


def run_tp_only(args: argparse.Namespace, requests: list[dict[str, Any]], out: Path) -> None:
    control, raw = out / "control.json", out / "worker_raw"
    raw.mkdir(parents=True, exist_ok=True)
    _set_env(out, control, raw)
    from vllm import LLM, SamplingParams
    llm = LLM(tensor_parallel_size=4, **_llm_kwargs(args.model, "tp_only"))
    _write_json(out / "runtime_proof.json", _proof(llm, {"tp": 4, "dp": 1, "ep": 0}, "tp_only", None))
    sampling = SamplingParams(max_tokens=1, temperature=0.0)
    rows: list[dict[str, Any]] = []
    refs: dict[str, list[int]] = {}
    for req in requests:
        for warm in range(args.warmups):
            _write_json(control, {"wave": f"warmup-{req['request_id']}-{warm}", "workload": req["workload"], "modality": req["modality"], "measured": False})
            _run_one(llm, req, sampling)
    wave = 0
    for iteration in range(args.iterations):
        for req in requests:
            _write_json(control, {"wave": f"measure-{wave}-{req['request_id']}", "workload": req["workload"], "modality": req["modality"], "volume": req["volume"], "iteration": iteration, "measured": True})
            result, wall = _run_one(llm, req, sampling)
            ids = [int(x) for x in (result.prompt_token_ids or [])]
            refs.setdefault(req["request_id"], ids)
            if refs[req["request_id"]] != ids:
                raise AssertionError(f"prompt tokenization changed: {req['request_id']}")
            rows.append({"topology": "tp_only", "request_id": req["request_id"], "volume": req["volume"], "workload": req["workload"], "modality": req["modality"], "image_count": req["image_count"], "iteration": iteration, "wall_ms": wall, "prompt_tokens": len(ids), "output_token_ids": [int(x) for x in result.outputs[0].token_ids]})
            wave += 1
    with (out / "request_wall.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_json(out / "run_complete.json", {"status": "complete", "rows": len(rows), "topology": "tp_only"})


def _dp_engine_run(llm: Any, prompts: list[dict[str, Any]], sampling: Any, barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def _run_dp_worker(dp_rank: int, port: int, args: argparse.Namespace, requests: list[dict[str, Any]], barrier: Any) -> None:
    out = args.output / f"driver.dp_rank{dp_rank}.json"
    control, raw = args.output / "control.json", args.output / "worker_raw"
    try:
        _set_env(args.output, control, raw, dp_rank, port)
        # DP workers are spawned after Python startup, so sitecustomize may
        # have run before the experiment flag was set.  Install the same
        # read-only hook explicitly here; child vLLM workers inherit it/env.
        from poc_flashvep.modality_aware_tp_ep_crossover.worker_hook import install
        install()
        from vllm import LLM, SamplingParams
        llm = LLM(tensor_parallel_size=2, **_llm_kwargs(args.model, "real_deepep"))
        _write_json(args.output / f"runtime_proof.dp{dp_rank}.json", _proof(llm, {"tp": 2, "dp": 2, "ep": 4, "all2all": "deepep_high_throughput"}, "real_deepep", dp_rank))
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records: list[dict[str, Any]] = []
        refs: dict[str, list[int]] = {}
        wave = 0
        for req in requests:
            for warm in range(args.warmups):
                if dp_rank == 0:
                    _write_json(control, {"wave": f"warmup-{req['request_id']}-{warm}", "workload": req["workload"], "modality": req["modality"], "measured": False})
                barrier.wait(timeout=1800)
                outputs = _dp_engine_run(llm, [copy.deepcopy(req["request"])] if dp_rank == 0 else [], sampling, barrier, wave)
                if dp_rank == 0 and outputs:
                    refs[req["request_id"]] = [int(x) for x in (outputs[0].prompt_token_ids or [])]
                wave += 1
        for iteration in range(args.iterations):
            for req in requests:
                if dp_rank == 0:
                    _write_json(control, {"wave": f"measure-{wave}-{req['request_id']}", "workload": req["workload"], "modality": req["modality"], "volume": req["volume"], "iteration": iteration, "measured": True})
                barrier.wait(timeout=1800)
                t0 = time.perf_counter_ns()
                outputs = _dp_engine_run(llm, [copy.deepcopy(req["request"])] if dp_rank == 0 else [], sampling, barrier, wave)
                wall = (time.perf_counter_ns() - t0) / 1e6
                if dp_rank == 0:
                    result = outputs[0]
                    ids = [int(x) for x in (result.prompt_token_ids or [])]
                    if ids != refs[req["request_id"]]:
                        raise AssertionError(f"prompt tokenization changed: {req['request_id']}")
                    records.append({"topology": "real_deepep", "request_id": req["request_id"], "volume": req["volume"], "workload": req["workload"], "modality": req["modality"], "image_count": req["image_count"], "iteration": iteration, "wall_ms": wall, "prompt_tokens": len(ids), "output_token_ids": [int(x) for x in result.outputs[0].token_ids]})
                wave += 1
        _write_json(out, {"ok": True, "dp_rank": dp_rank, "records": records})
    except BaseException:
        _write_json(out, {"ok": False, "dp_rank": dp_rank, "traceback": traceback.format_exc()})
        raise


def run_real_deepep(args: argparse.Namespace, requests: list[dict[str, Any]], out: Path) -> None:
    (out / "worker_raw").mkdir()
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    port = _port()
    procs = [ctx.Process(target=_run_dp_worker, args=(rank, port, args, requests, barrier)) for rank in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(7200)
    codes = [p.exitcode for p in procs]
    if codes != [0, 0]:
        raise RuntimeError(f"DeepEP DP workers failed: {codes}")
    records = json.loads((out / "driver.dp_rank0.json").read_text())["records"]
    with (out / "request_wall.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    _write_json(out / "run_complete.json", {"status": "complete", "rows": len(records), "topology": "real_deepep"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--topology", choices=("tp_only", "real_deepep"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    requests = make_requests(args.model)
    manifest = {
        "model": args.model, "visible_devices": "4,5,6,7", "physical_gpus": [4, 5, 6, 7],
        "dtype": "bfloat16", "topology": args.topology, "warmups": args.warmups,
        "iterations": args.iterations, "max_model_len": 12288, "max_num_batched_tokens": 12288,
        "requests": [{k: v for k, v in r.items() if k != "request"} for r in requests],
    }
    _write_json(args.output / "workload_manifest.json", manifest)
    if args.topology == "tp_only":
        run_tp_only(args, requests, args.output)
    else:
        run_real_deepep(args, requests, args.output)


if __name__ == "__main__":
    main()
