"""Run the exact prior 24 Vision/Text pairs as one-request live DP waves."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import _prepare_sample


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _source_dp(pair: dict[str, Any], modality: str, captures: dict[str, int]) -> int:
    if modality == "text":
        return captures[pair["text"]["request_id"]]
    source = pair["vision"]["source_route"]
    return int(source.split("routing.dp", 1)[1].split(".", 1)[0])


def _schedule(previous: Path, warmups: int, iterations: int) -> list[dict[str, Any]]:
    manifest = json.loads((previous / "workload_manifest.json").read_text())
    captures: dict[str, int] = {}
    for rank in range(2):
        payload = json.loads((previous / f"capture.dp_rank{rank}.json").read_text())
        captures.update({row["request_id"]: rank for row in payload["records"]})
    rows = []
    for pair in manifest["pairs"]:
        for modality in ("vision", "text"):
            item = pair[modality]
            rows.append({
                "request_id": item["request_id"], "modality": modality,
                "pair_id": int(pair["pair_id"]), "token_bucket": pair["token_bucket"],
                "prompt_tokens": int(item["prompt_tokens"]),
                "source_dp_rank": _source_dp(pair, modality, captures),
            })
    schedule: list[dict[str, Any]] = []

    # Preregistered representative subset: one matched pair per token bucket.
    overhead = [row for row in rows if row["pair_id"] in (0, 8, 16)]
    for row in overhead:
        for instrument in (False, True):
            for iteration in range(warmups):
                schedule.append({**row, "phase": "overhead", "instrument": instrument, "measured": False, "iteration": iteration})
        for iteration in range(iterations):
            for instrument in (False, True):
                schedule.append({**row, "phase": "overhead", "instrument": instrument, "measured": True, "iteration": iteration})
    for row in rows:
        for iteration in range(warmups + iterations):
            schedule.append({**row, "phase": "main", "instrument": True, "measured": iteration >= warmups, "iteration": iteration - warmups})
    for wave, row in enumerate(schedule):
        row["wave"] = wave
    return schedule


def _requests(previous: Path, model: str) -> dict[str, dict[str, Any]]:
    from transformers import AutoProcessor

    manifest = json.loads((previous / "workload_manifest.json").read_text())
    texts = {row["request_id"]: row for row in json.loads((previous / "text_prompts.json").read_text())}
    vision_source = Path(manifest["vision_source"])
    vision_manifest = json.loads((vision_source / "sample_manifest.json").read_text())
    vision_rows = {row["sample_id"]: row for row in vision_manifest["samples"]}
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    requests: dict[str, dict[str, Any]] = {}
    for pair in manifest["pairs"]:
        vision_id = pair["vision"]["request_id"]
        request, metadata = _prepare_sample(processor, vision_rows[vision_id])
        if metadata["processor_prompt_tokens"] != pair["vision"]["prompt_tokens"]:
            raise AssertionError((vision_id, metadata["processor_prompt_tokens"], pair["vision"]["prompt_tokens"]))
        requests[vision_id] = request
        text_id = pair["text"]["request_id"]
        requests[text_id] = {"prompt": texts[text_id]["prompt"]}
    return requests


def _generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any, barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    if prompts:
        barrier.wait(timeout=900)
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        barrier.wait(timeout=900)
        outputs = []
    barrier.wait(timeout=900)
    return outputs


def _run_rank(rank: int, port: int, args: argparse.Namespace, barrier: Any, schedule: list[dict[str, Any]]) -> None:
    output_path = args.output_dir / f"driver.dp_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank), "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_LIVE_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_LIVE_RAW_DIR": str((args.output_dir / "raw_live").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from vllm import LLM, SamplingParams

        requests = _requests(args.previous, args.model_path)
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=4096,
            max_num_batched_tokens=16384, max_num_seqs=2,
            skip_mm_profiling=True, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records = []
        for entry in schedule:
            if rank == 0:
                temporary = args.output_dir / "control.tmp.json"
                _json(temporary, entry)
                temporary.replace(args.output_dir / "control.json")
            barrier.wait(timeout=900)
            prompt = [copy.deepcopy(requests[entry["request_id"]])] if rank == entry["source_dp_rank"] else []
            start = time.perf_counter_ns()
            outputs = _generate(llm, prompt, sampling, barrier, int(entry["wave"]))
            wall_ms = (time.perf_counter_ns() - start) / 1_000_000
            tokens = [int(token) for output in outputs for token in output.outputs[0].token_ids]
            records.append({**entry, "driver_dp_rank": rank, "wall_ms": wall_ms, "output_tokens": tokens})
        flush = {**schedule[-1], "wave": len(schedule), "phase": "flush", "instrument": False, "measured": False, "flush": True, "iteration": 0}
        if rank == 0:
            temporary = args.output_dir / "control.tmp.json"; _json(temporary, flush); temporary.replace(args.output_dir / "control.json")
        barrier.wait(timeout=900)
        prompt = [copy.deepcopy(requests[flush["request_id"]])] if rank == flush["source_dp_rank"] else []
        _generate(llm, prompt, sampling, barrier, int(flush["wave"]))
        _json(output_path, {"ok": True, "records": records})
    except BaseException:
        _json(output_path, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("workload_manifest.json", "text_prompts.json"):
        shutil.copy2(args.previous / name, args.output_dir / name)
    shutil.copytree(args.previous / "routes", args.output_dir / "routes")
    schedule = _schedule(args.previous, args.warmups, args.iterations)
    _json(args.output_dir / "schedule.json", schedule)
    manifest = json.loads((args.previous / "workload_manifest.json").read_text())
    verification = {
        "source_result": str(args.previous), "pairs": len(manifest["pairs"]),
        "vision_requests": 24, "text_requests": 24,
        "manifest_sha256_source": _sha(args.previous / "workload_manifest.json"),
        "manifest_sha256_copy": _sha(args.output_dir / "workload_manifest.json"),
        "text_prompts_sha256_source": _sha(args.previous / "text_prompts.json"),
        "text_prompts_sha256_copy": _sha(args.output_dir / "text_prompts.json"),
        "max_relative_token_error": max(float(pair["relative_token_error"]) for pair in manifest["pairs"]),
        "pair_ids": [int(pair["pair_id"]) for pair in manifest["pairs"]],
    }
    _json(args.output_dir / "pair_identity_verification.json", verification)
    context = mp.get_context("spawn"); barrier = context.Barrier(2); port = _port()
    processes = [context.Process(target=_run_rank, args=(rank, port, args, barrier, schedule)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    if [process.exitcode for process in processes] != [0, 0]:
        raise RuntimeError(f"live prefill failed: {[process.exitcode for process in processes]}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
