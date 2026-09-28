#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import atomic_json, read_json
from poc_mllm_vision_conditioned_decode_reduction.scripts.capture_decode_vectors_ep4 import (
    _ep_wave, _model_kwargs, _prepare,
)


def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def probes(plan: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    answer = []
    for case in plan["cases"]:
        base = {
            **{key: case[key] for key in (
                "case_id", "request_id", "domain", "split", "question", "image_path",
                "forced_token_ids", "probe_step", "target_layer", "target_hidden_hash",
            )},
            "probe_id": f"{case['case_id']}.baseline",
            "intervene": False,
            "target_expert_id": None,
            "target_weight": None,
        }
        answer.append(base)
        for path in case["paths"]:
            answer.append({
                **base,
                "probe_id": f"{case['case_id']}.drop_slot{path['slot_rank']}",
                "intervene": True,
                "target_expert_id": int(path["expert_id"]),
                "target_weight": float(path["weight"]),
                "slot_rank": int(path["slot_rank"]),
            })
    for row in answer:
        row["logits_path"] = str((output / "logits" / f"{row['probe_id']}.npz").resolve())
    return answer


def sampling(row: dict[str, Any]):
    from vllm import SamplingParams
    forced = [int(value) for value in row["forced_token_ids"]]
    return SamplingParams(
        max_tokens=len(forced),
        temperature=0.0,
        seed=3407,
        extra_args={
            "damage_forced_token_ids": forced,
            "damage_probe_step": int(row["probe_step"]),
            "damage_logits_path": row["logits_path"],
            "damage_probe_id": row["probe_id"],
        },
    )


def worker(rank: int, rendezvous: int, args: argparse.Namespace, barrier: Any, progress: Any, rows: list[dict[str, Any]]) -> None:
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(rendezvous),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "MLLM_DAMAGE_CONTROL": str(args.control.resolve()),
            "MLLM_DAMAGE_MARKERS": str((args.output / "markers").resolve()),
            "MLLM_DAMAGE_PROBE_ENABLE": "1",
        })
        hook = str((Path(__file__).parents[1] / "hooks").resolve())
        os.environ["PYTHONPATH"] = f"{hook}:{os.environ.get('PYTHONPATH', '')}"
        from transformers import AutoProcessor
        from vllm import LLM
        from poc_mllm_vision_conditioned_decode_damage_skipping.functional_probe import install

        install()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        kwargs = _model_kwargs(args.max_model_len, args.kv_cache_gib)
        kwargs["model"] = args.model_path
        kwargs.pop("revision", None)
        kwargs["logits_processors"] = [
            "poc_mllm_vision_conditioned_decode_damage_skipping.functional_probe:DamageProbeLogitsProcessor"
        ]
        llm = LLM(**kwargs)
        progress.put((rank, -1, "MODEL_READY"))
        records = []
        for wave, row in enumerate(rows):
            source = wave % 2
            if rank == 0:
                atomic_json(args.control, {
                    "wave": wave,
                    "probe_id": row["probe_id"],
                    "source_dp_rank": source,
                    "intervene": row["intervene"],
                    "target_layer": row["target_layer"],
                    "target_hidden_hash": row["target_hidden_hash"],
                    "target_expert_id": row["target_expert_id"],
                    "target_weight": row["target_weight"],
                })
            barrier.wait(timeout=args.wave_timeout_sec)
            prepared = _prepare(processor, row) if rank == source else None
            prompts = [copy.deepcopy(prepared[0])] if prepared else []
            params = [sampling(row)] if prepared else []
            started = time.perf_counter_ns()
            outputs = _ep_wave(llm, prompts, params, barrier, wave)
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            if rank == source:
                if len(outputs) != 1:
                    raise AssertionError((wave, source, len(outputs)))
                actual = [int(value) for value in outputs[0].outputs[0].token_ids]
                if actual != row["forced_token_ids"]:
                    raise AssertionError(f"forced sequence mismatch: {row['probe_id']}")
                if not Path(row["logits_path"]).is_file():
                    raise RuntimeError(f"missing captured logits: {row['probe_id']}")
                records.append({
                    "probe_id": row["probe_id"],
                    "case_id": row["case_id"],
                    "request_id": row["request_id"],
                    "source_dp_rank": source,
                    "wave": wave,
                    "intervene": row["intervene"],
                    "target_expert_id": row["target_expert_id"],
                    "target_weight": row["target_weight"],
                    "token_count": len(actual),
                    "wall_ms_instrumented": wall_ms,
                    "forced_exact": True,
                    "logits_path": row["logits_path"],
                })
            progress.put((rank, wave, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True, "records": records})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": False, "traceback": traceback.format_exc()})
        progress.put((rank, -2, "FAILED"))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--model-path", default=(
        "/home/esjung/.cache/huggingface/hub/"
        "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
        "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    ))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-gib", type=float, default=2.0)
    parser.add_argument("--wave-timeout-sec", type=int, default=1800)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 4,5,6,7")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    args.control.parent.mkdir(parents=True, exist_ok=True)
    args.output.joinpath("logits").mkdir()
    args.output.joinpath("markers").mkdir()
    plan = read_json(args.plan)
    selected_cases = plan["cases"][args.start_case:]
    if args.limit_cases is not None:
        selected_cases = selected_cases[: args.limit_cases]
    plan = {**plan, "cases": selected_cases}
    rows = probes(plan, args.output)
    atomic_json(args.output / "manifest.json", {
        "probe_count": len(rows),
        "request_executions": len(rows),
        "interventions": sum(row["intervene"] for row in rows),
        "baselines": sum(not row["intervene"] for row in rows),
        "topology": "TRUE_TP2_DP2_EP4_PP1",
        "evidence": "INSTRUMENTED_FUNCTIONAL_DAMAGE_NOT_CLEAN_TIMING",
    })
    atomic_json(args.control, {"probe_id": "startup", "intervene": False})
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    rendezvous = port()
    workers = [
        context.Process(target=worker, args=(rank, rendezvous, args, barrier, progress, rows))
        for rank in range(2)
    ]
    for process in workers:
        process.start()
    last = time.monotonic()
    failure = None
    while any(process.is_alive() for process in workers):
        try:
            rank, wave, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "FAILED":
                failure = f"worker {rank} failed"
                break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.wave_timeout_sec:
            failure = "TIMEOUT: no worker progress"
            break
    if failure:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": "RUNTIME_UNAVAILABLE", "reason": failure})
        raise RuntimeError(failure)
    for process in workers:
        process.join()
    if any(process.exitcode != 0 for process in workers):
        raise RuntimeError([process.exitcode for process in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "probe_count": len(rows)})
    print(json.dumps({"status": "COMPLETE", "probes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
