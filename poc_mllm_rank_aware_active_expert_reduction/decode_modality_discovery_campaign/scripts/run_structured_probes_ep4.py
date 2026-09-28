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

from ..contract import MODEL_PATH, PHYSICAL_GPUS, TOPOLOGY, atomic_json, read_json
from poc_mllm_vision_conditioned_decode_reduction.scripts.capture_decode_vectors_ep4 import _ep_wave, _model_kwargs, _prepare


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); return int(sock.getsockname()[1])


def probes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in plan["cases"]:
        for variant in case["variants"]:
            rows.append({**case, **variant})
    return rows


def sampling(row: dict[str, Any], output: Path):
    from vllm import SamplingParams
    forced = [int(value) for value in row["forced_token_ids"]]
    # A functional probe needs only the frozen prefix through the four-step
    # damage horizon.  Continuing to the end of a 116+ token reference after
    # the last captured logit changes no label and roughly doubles GPU work.
    required_tokens = min(len(forced), int(row["probe_step"]) + int(row["horizon"]))
    return SamplingParams(max_tokens=required_tokens, temperature=0.0, seed=3407, extra_args={
        "dmc_forced_token_ids": forced, "dmc_probe_step": int(row["probe_step"]),
        "dmc_horizon": int(row["horizon"]), "dmc_logits_root": str((output / "logits").resolve()),
        "dmc_probe_id": row["probe_id"],
    })


def worker(rank: int, port: int, args: argparse.Namespace, barrier: Any, progress: Any, rows: list[dict[str, Any]]) -> None:
    llm = None
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank), "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput", "FLASHVEP_CONFIGURED_DBO": "false",
            "MLLM_DMC_PROBE_ENABLE": "1", "MLLM_DMC_PROBE_CONTROL": str(args.control.resolve()),
            "MLLM_DMC_PROBE_MARKERS": str((args.output / "markers").resolve()),
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
            "poc_mllm_rank_aware_active_expert_reduction.decode_modality_discovery_campaign.structured_probe:CampaignProbeLogitsProcessor"
        ]})
        kwargs.pop("revision", None)
        llm = LLM(**kwargs)
        progress.put((rank, -1, "MODEL_READY")); records = []
        for wave, row in enumerate(rows):
            source = wave % 2
            if rank == 0:
                atomic_json(args.control, {
                    "probe_id": row["probe_id"], "source_dp_rank": source,
                    "intervene": bool(row["intervene"]), "target_layer": int(row["target_layer"]),
                    "target_hidden_hash": row["target_hidden_hash"],
                    "removed_expert_ids": [int(value) for value in row["removed_expert_ids"]],
                    "retained_weight_mass": float(sum(
                        weight for slot, weight in enumerate(row["native_weights"], start=1)
                        if slot not in set(int(value) for value in row["removed_slots"])
                    )),
                })
            barrier.wait(timeout=args.wave_timeout_sec)
            prepared = _prepare(processor, row) if rank == source else None
            outputs = _ep_wave(llm, [copy.deepcopy(prepared[0])] if prepared else [],
                               [sampling(row, args.output)] if prepared else [], barrier, wave)
            if rank == source:
                actual = [int(value) for value in outputs[0].outputs[0].token_ids]
                expected = row["forced_token_ids"][:int(row["probe_step"]) + int(row["horizon"])]
                if actual != expected:
                    raise AssertionError(f"forced mismatch {row['probe_id']}")
                records.append({"probe_id": row["probe_id"], "forced_exact": True,
                                "executed_tokens": len(actual)})
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
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--kv-cache-gib", type=float, default=2.0)
    parser.add_argument("--wave-timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("campaign probes require CUDA_VISIBLE_DEVICES=2,3,6,7")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); (args.output / "logits").mkdir(); (args.output / "markers").mkdir()
    rows = probes(read_json(args.plan))
    atomic_json(args.output / "manifest.json", {
        "schema_version": 1, "probes": len(rows), "topology": TOPOLOGY,
        "physical_gpus": list(PHYSICAL_GPUS), "evidence": "EXACT_FUNCTIONAL_INTERVENTION_NOT_CLEAN_TIMING",
    })
    atomic_json(args.control, {"probe_id": "startup", "intervene": False})
    context = mp.get_context("spawn"); barrier, progress, port = context.Barrier(2), context.Queue(), _port()
    workers = [context.Process(target=worker, args=(rank, port, args, barrier, progress, rows)) for rank in range(2)]
    for item in workers: item.start()
    last, failure = time.monotonic(), None
    while any(item.is_alive() for item in workers):
        try:
            rank, _, status = progress.get(timeout=5); last = time.monotonic()
            if status == "FAILED": failure = f"worker {rank} failed"; break
        except queue.Empty: pass
        if time.monotonic() - last > args.wave_timeout_sec: failure = "TIMEOUT: no progress"; break
    if failure:
        for item in workers:
            if item.is_alive(): item.terminate()
        for item in workers: item.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": "RUNTIME_UNAVAILABLE", "reason": failure})
        raise RuntimeError(failure)
    for item in workers: item.join()
    if any(item.exitcode != 0 for item in workers): raise RuntimeError([item.exitcode for item in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "probes": len(rows)})
    print(json.dumps({"status": "COMPLETE", "probes": len(rows)}, indent=2))


if __name__ == "__main__": main()
