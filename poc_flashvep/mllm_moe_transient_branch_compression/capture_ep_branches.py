"""Fresh full-token branch-output capture on physical GPUs exposed by the caller.

The stock vLLM output is preserved. Diagnostic expert calls only materialize
counterfactual per-slot outputs for offline analysis.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import socket
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from poc_flashvep.prerouter_visual_signal.run_capture import MODEL, _base_suite

LAYERS = [4, 12, 24, 36, 44, 47]
EDGES = [336, 448, 672]
PROMPT = "Describe all visually important details, including text, numbers, and spatial relationships."


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(limit: int) -> list[dict[str, Any]]:
    base = _base_suite()
    output = []
    per_category = max(1, limit // 3)
    for category in ("natural", "fine_grained", "chart_document"):
        local = sorted((row for row in base if row["category"] == category),
                       key=lambda row: row["sample_id"])[:per_category]
        for index, row in enumerate(local):
            item = copy.deepcopy(row)
            item["edge"] = EDGES[index % len(EDGES)]
            item["base_sample_id"] = item["sample_id"]
            item["sample_id"] = f"{item['sample_id']}_e{item['edge']}"
            output.append(item)
    return output[:limit]


def prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image

    source = Path(row["image_paths"][0])
    original = Image.open(source).convert("RGB")
    edge = int(row["edge"])
    image = original.resize((edge, edge))
    content = [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True,
    )
    processed = processor(text=[prompt], images=[image], return_tensors="pt")
    ids = processed["input_ids"][0].tolist()
    image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    positions = [index for index, token in enumerate(ids) if token == image_id]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise AssertionError(f"{row['sample_id']}: non-contiguous visual span")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    if grid[0] * grid[1] * grid[2] // (merge * merge) != len(positions):
        raise AssertionError(f"{row['sample_id']}: grid/token mismatch")
    metadata = {
        "sample_id": row["sample_id"], "base_sample_id": row["base_sample_id"],
        "category": row["category"], "source_path": str(source),
        "source_sha256": sha256(source), "original_size": list(original.size),
        "input_size": list(image.size), "fixed_edge": edge, "fixed_prompt": PROMPT,
        "prompt_tokens": len(ids), "vision_tokens": len(positions),
        "visual_span": [positions[0], positions[-1] + 1],
        "post_visual_span": [positions[-1] + 1, len(ids)], "image_token_id": image_id,
        "image_grid_thw": grid, "merge_size": merge,
    }
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, metadata


def generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any,
             barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    if prompts:
        barrier.wait(timeout=1800)
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        barrier.wait(timeout=1800)
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def run_rank(rank: int, rendezvous_port: int, args: argparse.Namespace,
             prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
             schedule: list[dict[str, Any]], barrier: Any) -> None:
    output = args.output_dir / f"driver.dp{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(rendezvous_port),
            "FLASHVEP_FUNCTIONAL_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_FUNCTIONAL_RAW": str((args.output_dir / "raw").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false", "FLASHVEP_FUNCTIONAL_SAVE_HIDDEN": "1",
            "FLASHVEP_FUNCTIONAL_LAYERS": ",".join(map(str, LAYERS)),
            "FLASHVEP_FUNCTIONAL_SAMPLE_ALL": "1",
        })
        from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
        from poc_flashvep.visual_expert_functional_redundancy import instrumentation
        install_backend_probe()
        instrumentation.install()
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
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records = []
        for entry in schedule:
            if rank == 0:
                temporary = args.output_dir / "control.tmp.json"
                write_json(temporary, entry)
                temporary.replace(args.output_dir / "control.json")
            barrier.wait(timeout=1800)
            item = prepared[entry["sample_id"]]
            prompts = [copy.deepcopy(item[0])] if rank == entry["source_dp_rank"] else []
            outputs = generate(llm, prompts, sampling, barrier, int(entry["wave"]))
            if rank == entry["source_dp_rank"]:
                if len(outputs) != 1:
                    raise AssertionError((entry, len(outputs)))
                records.append({**item[1], **entry,
                                "returned_request_id": outputs[0].request_id,
                                "output_token_ids": list(outputs[0].outputs[0].token_ids)})
        write_json(output, {"ok": True, "records": records})
    except BaseException:
        write_json(output, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--sample-limit", type=int, default=9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "control.json", {"capture": False, "capture_id": "warmup"})
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prepared_list = [prepare(processor, row) for row in suite(args.sample_limit)]
    prepared = {item[1]["sample_id"]: item for item in prepared_list}
    schedule = [{"wave": index, "sample_id": item[1]["sample_id"],
                 "capture_id": item[1]["sample_id"], "capture": True,
                 "source_dp_rank": index % 2}
                for index, item in enumerate(prepared_list)]
    manifest = {
        "model": args.model_path,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                          "pp": 1, "all2all": "deepep_high_throughput",
                          "physical_gpus": [4, 5, 6, 7], "dbo": False},
        "policy": {"layers": LAYERS, "full_token_capture": True,
                   "edges": EDGES, "prompt": PROMPT},
        "schedule": schedule, "samples": [item[1] for item in prepared_list],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    rendezvous_port = port()
    processes = [context.Process(target=run_rank,
                                 args=(rank, rendezvous_port, args, prepared, schedule, barrier))
                 for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"capture failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
