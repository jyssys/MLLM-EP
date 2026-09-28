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

from ..contract import MODEL_PATH, TOPOLOGY, atomic_json, read_json
from poc_mllm_vision_conditioned_decode_reduction.scripts.capture_decode_vectors_ep4 import (
    _ep_wave, _model_kwargs, _prepare,
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _probes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in plan["cases"]:
        rows.append({**case, "probe_id": f"{case['probe_case_id']}.k8", "intervene": False,
                     "keep_k": 8, "kept_expert_ids": case["topk_ids"], "retained_mass": 1.0})
        for keep in (6, 4):
            rows.append({
                **case, "probe_id": f"{case['probe_case_id']}.k{keep}", "intervene": True,
                "keep_k": keep, "kept_expert_ids": case[f"keep{keep}_expert_ids"],
                "retained_mass": case[f"keep{keep}_mass"],
            })
    return rows


def _sampling(row: dict[str, Any], logits_root: Path):
    from vllm import SamplingParams
    forced = [int(value) for value in row["forced_token_ids"]]
    return SamplingParams(max_tokens=len(forced), temperature=0.0, seed=3407, extra_args={
        "vgb_forced_token_ids": forced, "vgb_probe_step": int(row["probe_step"]),
        "vgb_horizon": int(row["horizon"]), "vgb_logits_root": str(logits_root.resolve()),
        "vgb_probe_id": row["probe_id"],
    })


def _worker(rank: int, port: int, args: argparse.Namespace, barrier: Any,
            progress: Any, rows: list[dict[str, Any]]) -> None:
    llm = None
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank), "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false", "MLLM_VGB_PROBE_ENABLE": "1",
            "MLLM_VGB_PROBE_CONTROL": str(args.control.resolve()),
            "MLLM_VGB_PROBE_MARKERS": str((args.output / "markers").resolve()),
        })
        hook = str((Path(__file__).parents[1] / "hooks").resolve())
        os.environ["PYTHONPATH"] = f"{hook}:{os.environ.get('PYTHONPATH', '')}"
        from transformers import AutoProcessor
        from vllm import LLM
        from ..structured_probe import install
        install()
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        kwargs = _model_kwargs(args.max_model_len, args.kv_cache_gib)
        kwargs.update({"model": MODEL_PATH, "max_num_seqs": 1, "logits_processors": [
            "poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.structured_probe:StructuredProbeLogitsProcessor"
        ]})
        kwargs.pop("revision", None)
        llm = LLM(**kwargs)
        progress.put((rank, -1, "MODEL_READY"))
        records = []
        for wave, row in enumerate(rows):
            source = wave % 2
            if rank == 0:
                atomic_json(args.control, {
                    "wave": wave, "probe_id": row["probe_id"], "source_dp_rank": source,
                    "intervene": row["intervene"], "target_layer": row["target_layer"],
                    "target_hidden_hash": row["target_hidden_hash"], "keep_k": row["keep_k"],
                    "kept_expert_ids": row["kept_expert_ids"], "retained_mass": row["retained_mass"],
                })
            barrier.wait(timeout=args.wave_timeout_sec)
            prepared = _prepare(processor, row) if rank == source else None
            prompts = [copy.deepcopy(prepared[0])] if prepared else []
            params = [_sampling(row, args.output / "logits")] if prepared else []
            outputs = _ep_wave(llm, prompts, params, barrier, wave)
            if rank == source:
                actual = [int(value) for value in outputs[0].outputs[0].token_ids]
                if actual != row["forced_token_ids"]:
                    raise AssertionError(f"forced mismatch: {row['probe_id']}")
                for step in range(row["probe_step"], row["probe_step"] + row["horizon"]):
                    path = args.output / "logits" / f"{row['probe_id']}.step{step}.npz"
                    if not path.is_file():
                        raise RuntimeError(f"missing logits: {path}")
                records.append({"probe_id": row["probe_id"], "forced_exact": True})
            progress.put((rank, wave, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True, "records": records})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": False, "traceback": traceback.format_exc()})
        progress.put((rank, -2, "FAILED")); raise
    finally:
        if llm is not None:
            shutdown = getattr(llm, "shutdown", None)
            if callable(shutdown):
                try: shutdown()
                except BaseException: pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--kv-cache-gib", type=float, default=2.0)
    parser.add_argument("--wave-timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("Q2-3 TRUE EP4 requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True); (args.output / "logits").mkdir(); (args.output / "markers").mkdir()
    plan = read_json(args.plan)
    cases = plan["cases"][args.start_case:]
    if args.limit_cases is not None: cases = cases[:args.limit_cases]
    rows = _probes({**plan, "cases": cases})
    atomic_json(args.output / "manifest.json", {
        "schema_version": 1, "probe_count": len(rows), "condition_sites": len(cases),
        "structured_interventions": sum(row["intervene"] for row in rows),
        "topology": TOPOLOGY, "physical_gpus": [2,3,6,7],
        "evidence": "MEASURED_SEMANTIC_FUNCTIONAL_DAMAGE_NOT_PHYSICAL_WORK_OR_TIMING",
    })
    atomic_json(args.control, {"probe_id": "startup", "intervene": False})
    context = mp.get_context("spawn"); barrier, progress, port = context.Barrier(2), context.Queue(), _port()
    workers = [context.Process(target=_worker, args=(rank, port, args, barrier, progress, rows)) for rank in range(2)]
    for worker in workers: worker.start()
    last, failure = time.monotonic(), None
    while any(worker.is_alive() for worker in workers):
        try:
            rank, wave, status = progress.get(timeout=5); last = time.monotonic()
            if status == "FAILED": failure = f"worker {rank} failed"; break
        except queue.Empty: pass
        if time.monotonic() - last > args.wave_timeout_sec: failure = "TIMEOUT: no worker progress"; break
    if failure:
        for worker in workers:
            if worker.is_alive(): worker.terminate()
        for worker in workers: worker.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": "RUNTIME_UNAVAILABLE", "reason": failure})
        raise RuntimeError(failure)
    for worker in workers: worker.join()
    if any(worker.exitcode != 0 for worker in workers): raise RuntimeError([worker.exitcode for worker in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "probe_count": len(rows)})
    print(json.dumps({"status": "COMPLETE", "probes": len(rows)}, indent=2))


if __name__ == "__main__": main()
