"""Run the preregistered 24-image functional expert-output capture."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import socket
import traceback
from pathlib import Path
from typing import Any

from poc_flashvep.cross_modal_routing_imprint.capture import MODEL, _prepare
from poc_flashvep.prerouter_visual_signal.run_capture import _base_suite

LAYERS = [4, 8, 12, 20, 24, 28, 36, 40, 44, 47]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0)); return int(sock.getsockname()[1])


def _suite() -> list[dict[str, Any]]:
    base = _base_suite()
    selected = []
    for category in ("natural", "fine_grained", "chart_document"):
        selected.extend(sorted((row for row in base if row["category"] == category),
                               key=lambda row: row["sample_id"])[:8])
    if len(selected) != 24:
        raise AssertionError(len(selected))
    return selected


def _generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any,
              barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    if prompts:
        barrier.wait(timeout=900)
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        barrier.wait(timeout=900)
        outputs = []
    barrier.wait(timeout=900)
    return outputs


def _run_rank(rank: int, port: int, args: argparse.Namespace,
              prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
              schedule: list[dict[str, Any]], barrier: Any) -> None:
    output = args.output_dir / f"driver.dp{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank), "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_FUNCTIONAL_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_FUNCTIONAL_RAW": str((args.output_dir / "raw").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=1024,
            max_num_batched_tokens=1024, max_num_seqs=1,
            limit_mm_per_prompt={"image": 1}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False, enforce_eager=True)
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records = []
        for entry in schedule:
            if rank == 0:
                temporary = args.output_dir / "control.tmp.json"
                _json(temporary, entry); temporary.replace(args.output_dir / "control.json")
            barrier.wait(timeout=900)
            item = prepared[entry["sample_id"]]
            prompts = [copy.deepcopy(item[0])] if rank == entry["source_dp_rank"] else []
            outputs = _generate(llm, prompts, sampling, barrier, int(entry["wave"]))
            if rank == entry["source_dp_rank"]:
                if len(outputs) != 1:
                    raise AssertionError((entry, len(outputs)))
                result = outputs[0]
                records.append({**item[1], **entry,
                                "returned_request_id": result.request_id,
                                "output_token_ids": list(result.outputs[0].token_ids)})
        _json(output, {"ok": True, "records": records})
    except BaseException:
        _json(output, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=MODEL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _json(args.output_dir / "control.json", {"capture": False, "capture_id": "warmup"})
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prepared_list = [_prepare(processor, row) for row in _suite()]
    prepared = {item[1]["sample_id"]: item for item in prepared_list}
    schedule = [{"wave": index, "sample_id": item[1]["sample_id"],
                 "capture_id": item[1]["sample_id"], "capture": True,
                 "source_dp_rank": index % 2}
                for index, item in enumerate(prepared_list)]
    policy = {
        "layers": LAYERS, "regions": {"early": [4, 8, 12], "middle": [20, 24, 28],
                                       "late": [36, 40, 44, 47]},
        "samples_per_category": 8, "fixed_edge": 448,
        "fixed_prompt": "Describe the image briefly.", "hash_sample_fraction": 0.25,
        "functional_k_threshold": {"cosine": 0.99, "relative_l2": 0.05},
        "m_values": [1, 2, 3, 4, 6, 8], "router_mass_threshold": 0.95,
        "go": "late visual functional-K >=30% below text with clustered CI excluding zero, diversity agrees, and >=50% gap remains after router-effective-K adjustment",
        "hold": "10-30% significant/consistent gap or a >=30% gap confined to some preregistered layers; otherwise NO-GO",
    }
    manifest = {"model": args.model_path,
                "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                                  "pp": 1, "all2all": "deepep_high_throughput",
                                  "physical_gpus": [1, 2, 3, 4], "dbo": False},
                "policy": policy,
                "schedule": schedule, "samples": [item[1] for item in prepared_list]}
    _json(args.output_dir / "manifest.json", manifest)
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(2); port = _port()
    processes = [ctx.Process(target=_run_rank, args=(rank, port, args, prepared, schedule, barrier))
                 for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]: raise RuntimeError(f"capture failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__": main()
