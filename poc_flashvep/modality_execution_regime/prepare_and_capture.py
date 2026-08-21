"""Build matched Vision/Text workloads and capture stock vLLM router IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import socket
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import (
    MODEL_DEFAULT,
    NUM_EXPERTS,
    NUM_LAYERS,
    TOPK,
    _load_direct_trace,
)


VISION_RESULT = Path(
    "/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/"
    "tile_slack_mechanism_20260820_150852/stage_a"
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _select_vision(source: Path) -> list[dict[str, Any]]:
    manifest = json.loads((source / "sample_manifest.json").read_text())
    rows = sorted(manifest["samples"], key=lambda row: row["processor_prompt_tokens"])
    # Eight samples in each preregistered range; every prompt is >100 tokens so
    # it is distinguishable from DeepEP's startup-only dummy calls.
    ranges = ((128, 276, "small"), (277, 700, "medium"), (800, 3000, "large"))
    selected: list[dict[str, Any]] = []
    for low, high, bucket in ranges:
        candidates = [row for row in rows if low <= row["processor_prompt_tokens"] <= high]
        if len(candidates) < 8:
            raise AssertionError(f"{bucket}: only {len(candidates)} vision samples")
        # Span each interval rather than taking eight adjacent lengths.
        indices = np.linspace(0, len(candidates) - 1, 8).round().astype(int)
        for index in indices:
            row = dict(candidates[int(index)])
            row["token_bucket"] = bucket
            selected.append(row)
    if len({row["sample_id"] for row in selected}) != 24:
        raise AssertionError("vision selection contains duplicates")
    return selected


def _text_sources(repo: Path) -> list[Path]:
    roots = (repo / "docs", repo / "external" / "lmms-eval" / "docs")
    paths = sorted(path for root in roots for path in root.rglob("*.md"))
    usable = [path for path in paths if path.stat().st_size >= 2500]
    if len(usable) < 12:
        raise AssertionError(f"only {len(usable)} usable local text files")
    return usable


def _clean_text(path: Path) -> str:
    lines = []
    fenced = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not line or line.startswith("!["):
            continue
        line = line.lstrip("#>*- ").strip()
        if len(line) >= 24:
            lines.append(line)
    return "\n\n".join(lines)


def _make_text_prompt(tokenizer: Any, sources: list[Path], pair: int, target: int) -> tuple[str, dict[str, Any]]:
    ordered = sources[pair % len(sources) :] + sources[: pair % len(sources)]
    pieces: list[str] = []
    used: list[Path] = []
    for source in ordered:
        text = _clean_text(source)
        if text:
            pieces.append(text)
            used.append(source)
        if sum(len(item) for item in pieces) >= target * 8:
            break
    corpus = "\n\n".join(pieces)
    corpus_ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]
    if len(corpus_ids) < target:
        raise AssertionError(f"text corpus too short for {target} tokens")

    def render(n: int) -> tuple[str, list[int]]:
        body = tokenizer.decode(corpus_ids[:n], skip_special_tokens=True)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, tokenizer(prompt, add_special_tokens=False)["input_ids"]

    best: tuple[int, str, list[int]] | None = None
    lo, hi = 1, min(len(corpus_ids), target + 256)
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt, ids = render(mid)
        candidate = (abs(len(ids) - target), prompt, ids)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if len(ids) < target:
            lo = mid + 1
        elif len(ids) > target:
            hi = mid - 1
        else:
            best = candidate
            break
    assert best is not None
    if best[0] / target > 0.05:
        raise AssertionError(f"cannot match target {target}: got {len(best[2])}")
    provenance = {
        "source_files": [str(path) for path in used],
        "source_sha256": {str(path): _sha(path) for path in used},
        "rule": "distinct local documentation prose, tokenized prefix; no sentence repetition",
    }
    return best[1], {**provenance, "prompt_tokens": len(best[2])}


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=False)
    route_dir = args.output_dir / "routes"
    route_dir.mkdir()
    selected = _select_vision(args.vision_source)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    sources = _text_sources(args.repo_root)
    pairs = []
    text_requests = []
    for pair, vision in enumerate(selected):
        target = int(vision["processor_prompt_tokens"])
        prompt, provenance = _make_text_prompt(tokenizer, sources, pair, target)
        text_id = f"text_{pair:02d}_{vision['sample_id']}"
        matches = list(args.vision_source.glob(f"routing.dp*.{vision['sample_id']}.npz"))
        if len(matches) != 1:
            raise AssertionError(f"{vision['sample_id']}: route matches {matches}")
        source_npz = matches[0]
        vision_name = f"vision.{vision['sample_id']}.npz"
        shutil.copy2(source_npz, route_dir / vision_name)
        text_name = f"text.{text_id}.npz"
        pair_row = {
            "pair_id": pair,
            "token_bucket": vision["token_bucket"],
            "vision": {
                "request_id": vision["sample_id"],
                "category": vision["category"],
                "prompt_tokens": target,
                "route_file": f"routes/{vision_name}",
                "source_route": str(source_npz),
                "image_paths": vision["image_paths"],
                "vision_token_spans": [image["token_span"] for image in vision["images"]],
                "vision_tokens": int(vision["processor_vision_tokens"]),
            },
            "text": {
                "request_id": text_id,
                "category": "text_only",
                "prompt_tokens": provenance["prompt_tokens"],
                "route_file": f"routes/{text_name}",
                **provenance,
            },
        }
        pair_row["relative_token_error"] = abs(
            provenance["prompt_tokens"] - target
        ) / target
        pairs.append(pair_row)
        text_requests.append(
            {
                "request_id": text_id,
                "prompt": prompt,
                "prompt_tokens": provenance["prompt_tokens"],
                "route_file": text_name,
                "pair_id": pair,
                "token_bucket": vision["token_bucket"],
            }
        )
    _json(args.output_dir / "text_prompts.json", text_requests)
    _json(
        args.output_dir / "workload_manifest.json",
        {
            "model": args.model_path,
            "configuration": {
                "dtype": "BF16", "tp": 2, "dp": 2, "ep": 4, "pp": 1,
                "all2all": "deepep_high_throughput", "dbo": False,
                "enforce_eager": True, "physical_gpus": [4, 5, 6, 7],
            },
            "software": {"python": platform.python_version()},
            "vision_source": str(args.vision_source),
            "matching": "effective decoder input tokens, nearest within 5%",
            "pairs": pairs,
        },
    )


def _partition(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = [[], []]
    loads = [0, 0]
    for row in sorted(rows, key=lambda value: -value["prompt_tokens"]):
        rank = 0 if loads[0] <= loads[1] else 1
        result[rank].append(row)
        loads[rank] += row["prompt_tokens"]
    return result


def _capture_rank(rank: int, port: int, args: argparse.Namespace, rows: list[dict[str, Any]], barrier: Any) -> None:
    result_path = args.output_dir / f"capture.dp_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_DIRECT_ROUTING_DIR": str((args.output_dir / "direct_router_capture").resolve()),
        })
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput

        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=True, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=4096,
            max_num_batched_tokens=16384, max_num_seqs=16,
            skip_mm_profiling=True, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True,
        )
        barrier.wait(timeout=900)
        submitted = llm._add_completion_requests(
            [row["prompt"] for row in rows],
            SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False,
        )
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
        barrier.wait(timeout=900)
        lengths = [len(output.prompt_token_ids or []) for output in outputs]
        routed, call_groups = _load_direct_trace(
            args.output_dir / "direct_router_capture", rank, lengths
        )
        offset = 0
        records = []
        for submitted_id, row, output, length in zip(submitted, rows, outputs, lengths, strict=True):
            expected = int(row["prompt_tokens"])
            if length != expected:
                raise AssertionError(f"{row['request_id']}: tokenizer length {length} != {expected}")
            local = routed[offset : offset + length]
            offset += length
            if local.shape != (length, NUM_LAYERS, TOPK):
                raise AssertionError((row["request_id"], local.shape))
            np.savez_compressed(
                args.output_dir / "routes" / row["route_file"],
                routed_experts=local,
                prompt_token_ids=np.asarray(output.prompt_token_ids, dtype=np.int64),
            )
            records.append({
                "request_id": row["request_id"], "dp_rank": rank,
                "submitted_request_id": submitted_id,
                "returned_request_id": output.request_id,
                "prompt_tokens": length, "routed_shape": list(local.shape),
                "model_call_groups": call_groups,
                "output_token_ids": list(output.outputs[0].token_ids),
            })
        _json(result_path, {"ok": True, "records": records})
    except Exception:
        _json(result_path, {"ok": False, "traceback": traceback.format_exc()})
        raise


def capture(args: argparse.Namespace) -> None:
    rows = json.loads((args.output_dir / "text_prompts.json").read_text())
    partitions = _partition(rows)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _port()
    processes = [
        context.Process(target=_capture_rank, args=(rank, port, args, partitions[rank], barrier))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if [process.exitcode for process in processes] != [0, 0]:
        raise RuntimeError("text route capture failed")
    missing = [row["route_file"] for row in rows if not (args.output_dir / "routes" / row["route_file"]).is_file()]
    if missing:
        raise AssertionError(f"missing text routes: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("capture", capture)):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--model-path", default=MODEL_DEFAULT)
        command.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
        command.add_argument("--vision-source", type=Path, default=VISION_RESULT)
        command.set_defaults(func=function)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
