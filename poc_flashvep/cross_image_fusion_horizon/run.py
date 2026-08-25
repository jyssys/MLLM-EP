"""Run a fixed 24-pair, two-image causal fusion-horizon experiment."""

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

EDGE = 448
HORIZONS = [4, 8, 12, 16, 24, 32]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sources() -> list[dict[str, Any]]:
    rows = sorted((row for row in _base_suite() if len(row["image_paths"]) == 1),
                  key=lambda row: row["sample_id"])
    unique = []
    seen = set()
    for row in rows:
        path = str(Path(row["image_paths"][0]).resolve())
        if path in seen:
            continue
        seen.add(path)
        unique.append({**row, "path": path})
    if len(unique) < 18:
        raise AssertionError(f"only {len(unique)} unique images")
    return unique[:24]


def _brightness(path: str) -> float:
    from PIL import Image
    image = np.asarray(Image.open(path).convert("RGB").resize((EDGE, EDGE)), dtype=np.float32)
    return float(np.mean(image @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)))


def _pair_specs() -> list[dict[str, Any]]:
    sources = _sources()
    specs = []
    for index in range(6):
        source = sources[index]
        specs.append({"pair_id": f"identity_same_{index:02d}", "task": "identity",
                      "first": source, "second": source, "answer": "YES",
                      "question": "Are the two images identical? Answer only YES or NO."})
    for index in range(6):
        specs.append({"pair_id": f"identity_different_{index:02d}", "task": "identity",
                      "first": sources[6 + index], "second": sources[12 + index],
                      "answer": "NO",
                      "question": "Are the two images identical? Answer only YES or NO."})
    ordered = sorted(sources, key=lambda row: _brightness(row["path"]))
    for index in range(6):
        dark, bright = ordered[index], ordered[-1 - index]
        specs.append({"pair_id": f"brightness_second_{index:02d}", "task": "brightness",
                      "first": dark, "second": bright, "answer": "SECOND",
                      "question": "Which image is brighter overall? Answer only FIRST or SECOND."})
        specs.append({"pair_id": f"brightness_first_{index:02d}", "task": "brightness",
                      "first": bright, "second": dark, "answer": "FIRST",
                      "question": "Which image is brighter overall? Answer only FIRST or SECOND."})
    if len(specs) != 24:
        raise AssertionError(len(specs))
    return specs


def _prepare(processor: Any, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image
    images = [Image.open(spec[key]["path"]).convert("RGB").resize((EDGE, EDGE))
              for key in ("first", "second")]
    content = []
    for index, image in enumerate(images, start=1):
        content.extend([{"type": "text", "text": f"Image {index}:"},
                        {"type": "image", "image": image}])
    content.append({"type": "text", "text": spec["question"]})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    processed = processor(text=[prompt], images=images, return_tensors="pt")
    ids = processed["input_ids"][0].tolist()
    image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    spans = []
    cursor = 0
    while cursor < len(ids):
        if ids[cursor] != image_id:
            cursor += 1
            continue
        end = cursor
        while end < len(ids) and ids[end] == image_id:
            end += 1
        spans.append([cursor, end])
        cursor = end
    if len(spans) != 2 or any(end - start != 196 for start, end in spans):
        raise AssertionError((spec["pair_id"], spans))
    # Skip the second image's structural vision-end token.
    post_start = spans[1][1] + 1
    metadata = {
        "pair_id": spec["pair_id"], "task": spec["task"],
        "expected_answer": spec["answer"], "question": spec["question"],
        "prompt_tokens": len(ids), "image_spans": spans, "post_start": post_start,
        "input_edge": EDGE, "visual_tokens_per_image": 196,
        "images": [{"sample_id": spec[key]["sample_id"],
                    "category": spec[key]["category"], "path": spec[key]["path"],
                    "sha256": _sha(Path(spec[key]["path"])),
                    "brightness": _brightness(spec[key]["path"])}
                   for key in ("first", "second")],
    }
    return {"prompt": prompt, "multi_modal_data": {"image": images}}, metadata


def _conditions(smoke: bool) -> list[dict[str, Any]]:
    conditions = [{"condition": "stock", "intervention": "stock", "horizon": 0},
                  {"condition": "visual_h0", "intervention": "visual", "horizon": 0},
                  {"condition": "full_h0", "intervention": "full", "horizon": 0}]
    for horizon in HORIZONS:
        conditions.extend([
            {"condition": f"visual_h{horizon}", "intervention": "visual",
             "horizon": horizon},
            {"condition": f"full_h{horizon}", "intervention": "full",
             "horizon": horizon},
        ])
    return conditions[:5] if smoke else conditions


def _generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any,
              barrier: Any, wave: int) -> list[Any]:
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


def _run_rank(rank: int, port: int, args: argparse.Namespace,
              prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
              schedule: list[dict[str, Any]], barrier: Any) -> None:
    path = args.output_dir / f"driver.dp{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_FUSION_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_FUSION_LOGITS": str((args.output_dir / "raw_logits").resolve()),
            "FLASHVEP_FUSION_INTERACTION": str((args.output_dir / "interaction").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_ep_weight_filter=True, trust_remote_code=True,
            gpu_memory_utilization=.90, kv_cache_memory_bytes=1 << 30,
            max_model_len=512, max_num_batched_tokens=512, max_num_seqs=1,
            limit_mm_per_prompt={"image": 2}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True)
        sampling = SamplingParams(max_tokens=4, temperature=0.0)
        records = []
        for entry in schedule:
            if rank == 0:
                temporary = args.output_dir / "control.tmp.json"
                _json(temporary, entry)
                temporary.replace(args.output_dir / "control.json")
            barrier.wait(timeout=900)
            prompt, metadata = prepared[entry["pair_id"]]
            prompts = [copy.deepcopy(prompt)] if rank == entry["source_dp_rank"] else []
            outputs = _generate(llm, prompts, sampling, barrier, int(entry["wave"]))
            if rank == entry["source_dp_rank"]:
                if len(outputs) != 1:
                    raise AssertionError((entry, len(outputs)))
                output = outputs[0]
                records.append({**metadata, **entry,
                                "request_id": output.request_id,
                                "output_token_ids": list(output.outputs[0].token_ids),
                                "output_text": output.outputs[0].text})
        _json(path, {"ok": True, "records": records})
    except BaseException:
        _json(path, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _json(args.output_dir / "control.json", {"capture": False, "capture_id": "warmup"})
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    specs = _pair_specs()[:1] if args.smoke else _pair_specs()
    prepared_list = [_prepare(processor, spec) for spec in specs]
    prepared = {metadata["pair_id"]: (prompt, metadata)
                for prompt, metadata in prepared_list}
    conditions = _conditions(args.smoke)
    schedule = []
    wave = 0
    for condition in conditions:
        for index, (_, metadata) in enumerate(prepared_list):
            capture_id = f"{metadata['pair_id']}__{condition['condition']}"
            schedule.append({**condition, "wave": wave, "capture": True,
                             "capture_id": capture_id, "pair_id": metadata["pair_id"],
                             "prompt_tokens": metadata["prompt_tokens"],
                             "image_spans": metadata["image_spans"],
                             "post_start": metadata["post_start"],
                             "source_dp_rank": index % 2,
                             "interaction_capture": (condition["condition"] == "stock" and
                                                     index < 8)})
            wave += 1
    manifest = {
        "model": args.model_path,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                          "pp": 1, "all2all": "deepep_high_throughput",
                          "physical_gpus": [1, 2, 3, 4], "dbo": False},
        "policy": {"horizons": [0, *HORIZONS], "primary": "full",
                   "accuracy_drop_go": 0.02, "minimum_go_horizon": 12,
                   "degradation_boundary_drop": 0.10,
                   "tasks": {"identity": 12, "brightness": 12},
                   "baseline_correct_primary": True,
                   "interaction_subset": "first 8 fixed pairs"},
        "pairs": [metadata for _, metadata in prepared_list],
        "conditions": conditions,
        "schedule": schedule,
    }
    _json(args.output_dir / "manifest.json", manifest)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    port = _port()
    processes = [ctx.Process(target=_run_rank,
                             args=(rank, port, args, prepared, schedule, barrier))
                 for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"run failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
