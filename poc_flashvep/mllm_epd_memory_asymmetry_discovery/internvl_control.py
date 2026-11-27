"""InternVL3 native Embedding parity and exact-prefix cache witness."""

import argparse
from collections import defaultdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from ..encoder_assisted_decode_attention.internvl_native import (
    choose_cases,
    model_payload,
    prepare_case,
    runtime_config,
)
from ..encoder_assisted_decode_attention.download_models import CHECKPOINTS
from .workload import file_digest, ordered_images, write_frozen


def start_capture(model, capture_embeddings=False):
    import torch

    if hasattr(model, "_memory_control"):
        raise RuntimeError("InternVL hooks already armed")
    state = {"gpu": [], "embedding_calls": [], "embedding_tensors": [] if capture_embeddings else None,
             "original_embed": model.embed_multimodal}

    def embed(**kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tensors = state["original_embed"](**kwargs)
        end.record()
        if tensors:
            state["embedding_calls"].append(("pixels" if "pixel_values_flat" in kwargs else "embedding",
                                             [list(t.shape) for t in tensors], start, end))
            if capture_embeddings:
                state["embedding_tensors"].extend(t.detach() for t in tensors)
        return tensors

    def start_forward(_module, _args, kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        state["gpu"].append((kwargs["positions"].numel(), start, end))

    def finish_forward(_module, _args, _output):
        state["gpu"][-1][2].record()

    model.embed_multimodal = embed
    state["before"] = model.language_model.model.register_forward_pre_hook(start_forward, with_kwargs=True)
    state["after"] = model.language_model.model.register_forward_hook(finish_forward)
    model._memory_control = state


def finish_capture(model):
    state = model._memory_control
    state["before"].remove()
    state["after"].remove()
    model.embed_multimodal = state["original_embed"]
    del model._memory_control
    embeddings = state["embedding_tensors"]
    return {
        "vision_calls": [{"input": kind, "shapes": shapes, "gpu_event_ms": before.elapsed_time(after)}
                         for kind, shapes, before, after in state["embedding_calls"]],
        "language_calls": [{"position_rows": rows, "gpu_event_ms": before.elapsed_time(after)}
                           for rows, before, after in state["gpu"]],
    }, (None if embeddings is None else [t.cpu() for t in embeddings])


def run_once(engine, sampling, inputs, salt, capture=False):
    import torch

    # Stock worker encoder outputs can survive a finished request. A new
    # prefix salt only namespaces APC, so reset the separate encoder cache.
    engine.llm_engine.reset_encoder_cache()
    engine.apply_model(partial(start_capture, capture_embeddings=capture))
    started = perf_counter()
    output, = engine.generate([{**inputs, "cache_salt": salt}], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    wall_ms = (perf_counter() - started) * 1000
    (hooks, tensors), = engine.apply_model(finish_capture)
    completion, = output.outputs
    return {
        "model_prompt_tokens": len(output.prompt_token_ids), "num_cached_tokens": output.num_cached_tokens,
        "output_token_ids": list(completion.token_ids), "wall_ms": wall_ms, **hooks,
    }, output.prompt_token_ids, tensors


def metadata(inputs, processor, meta):
    features = processor(text=inputs["prompt"], images=inputs["multi_modal_data"]["image"], return_tensors="pt")
    ids = features["input_ids"].flatten()
    return meta | {
        "native_processor_tokens": len(ids),
        "native_visual_rows": int((ids == processor.ctx_image_token_id).sum()),
        "native_tiles_per_image": features["image_num_patches"].tolist(),
    }


def document_pair(path):
    if path is None:
        return None
    groups = defaultdict(list)
    for row in json.loads(path.read_bytes())["requests"]:
        groups[ordered_images(row)].append(row)
    pairs = [(group[0], item) for group in groups.values() for item in group[1:]
             if item["question"] != group[0]["question"]]
    if not pairs:
        raise ValueError("No distinct questions with identical ordered images")
    # Smallest compatible real multi-image document first; fixed before results.
    return min(pairs, key=lambda pair: (pair[0]["image_count"] < 2, pair[0]["image_count"],
                                       pair[0]["total_input_tokens"], pair[0]["request_id"], pair[1]["request_id"]))


def embeddings_info(tensors):
    import torch

    return [{"shape": list(t.shape), "dtype": str(t.dtype), "bytes": t.numel() * t.element_size(),
             "sha256": hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for t in tensors]


def confirmed(actual, expected, condition):
    if actual != expected:
        raise ValueError(f"{condition}: found {actual!r}, expected {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--main100", required=True, type=Path)
    parser.add_argument("--paired-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--kv-gib", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.tokens < 16 or args.kv_gib < 1 or not 1 <= args.max_cases <= 4:
        parser.error("Need at least 16 generation tokens and 1 GiB KV, with one to four smoke requests")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    config, tokenizer, processor, geometry, rope = runtime_config(args.model.resolve())
    requests = choose_cases(args.main100)[:args.max_cases]
    pair = document_pair(args.paired_manifest)
    native = []
    for target, row in requests + [(item["image_count"], item) for item in pair or ()]:
        inputs, meta = prepare_case(target, row, tokenizer)
        info = metadata(inputs, processor, meta)
        info["eligible"] = info["native_processor_tokens"] + args.tokens <= config.max_model_len
        native.append((inputs, info))
    kv_bytes = geometry["num_hidden_layers"] * geometry["num_key_value_heads"] * geometry["head_dim"] * 4
    checkpoint, revision, _ = CHECKPOINTS["internvl3_8b"]
    result = {
        "evidence": "DIAGNOSTIC" if args.preflight_only else "MEASURED_PHYSICAL",
        "model": checkpoint, "revision": revision, "geometry": geometry, "rope": rope,
        "main100_sha256": file_digest(args.main100),
        "pair_manifest_sha256": file_digest(args.paired_manifest) if pair else None,
        "processor": type(processor).__name__, "bf16_full_layer_kv_bytes_per_row": kv_bytes,
        "cases": [info for _, info in native],
        "scope": "Serial in-process vLLM, fresh salt per cold draw; processor cache disabled and encodings reset per draw. "
                 "Embeddings come from the exact native run and are round-tripped through host before injection. "
                 "No store connector, EPD, or online serving. CUDA events include instrumented call enclosures. "
                 "APC is the native local exact prefix cache. Fixed 16+ greedy tokens with ignore_eos.",
    }
    if args.preflight_only:
        write_frozen(args.out, result)
        return

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    model = LLM(model=str(args.model.resolve()), model_impl="vllm", trust_remote_code=True, dtype="bfloat16",
                enforce_eager=True, enable_prefix_caching=True, enable_mm_embeds=True, mm_processor_cache_gb=0,
                tensor_parallel_size=1, max_num_batched_tokens=8192, max_num_seqs=4, block_size=128,
                kv_cache_memory_bytes=args.kv_gib * 2**30,
                limit_mm_per_prompt={"image": max(meta["actual_images"] for _, meta in native), "video": 0})
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.tokens)
    result["runtime"] = {"vllm": vllm.__version__, "torch": str(torch.__version__),
                         "gpu": torch.cuda.get_device_name(), "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                         "kv_budget_bytes": args.kv_gib * 2**30}
    result["resident_weights"], = model.apply_model(model_payload)
    model.generate([{**native[0][0], "cache_salt": "warm-up"}], sampling, use_tqdm=False)
    result["embedding_round_trip"] = []
    for i, (inputs, meta) in enumerate(native[:len(requests)]):
        if not meta["eligible"]:
            continue
        cold, ids, embeddings = run_once(model, sampling, inputs, f"embed-{i}-native1", capture=True)
        aa, ids_aa, second = run_once(model, sampling, inputs, f"embed-{i}-native2", capture=True)
        confirmed(len(ids), meta["native_processor_tokens"], "Raw image token count")
        confirmed(list(map(len, embeddings)), [x * processor.image_seq_length for x in meta["native_tiles_per_image"]],
                  "Native embedding rows per image")
        confirmed([cold["num_cached_tokens"], aa["num_cached_tokens"]], [0, 0], "Cold request native APC rows")
        representation = embeddings_info(embeddings)
        row = {
            "request_id": meta["request_id"], "representations": representation,
            "embedding_bytes": sum(row["bytes"] for row in representation),
            "full_prompt_logical_kv_bytes": len(ids) * kv_bytes,
            "native_AA_prompt_ids_match": ids == ids_aa,
            "native_embedding_bitwise_match": representation == embeddings_info(second),
            "native_AA_greedy_match": cold["output_token_ids"] == aa["output_token_ids"],
            "native": cold, "native_repeated": aa,
        }
        if len({tuple(t.shape) for t in embeddings}) == 1:
            inject_inputs = {**inputs, "multi_modal_data": {"image": embeddings}}
            inject, ids_inject, returned = run_once(model, sampling, inject_inputs, f"embed-{i}-injected", capture=True)
            confirmed(inject["num_cached_tokens"], 0, "Injected request native APC rows")
            row.update(injected=inject, injection_status="EXECUTED",
                       injected_prompt_ids_match=ids == ids_inject,
                       injected_embedding_bitwise_match=representation == embeddings_info(returned),
                       injected_greedy_match=cold["output_token_ids"] == inject["output_token_ids"])
        else:
            row["injection_status"] = "STOCK_RAGGED_TENSOR_SCHEMA_UNSUPPORTED"
        result["embedding_round_trip"].append(row)
    if pair and all(meta["eligible"] for _, meta in native[-2:]):
        seed_inputs, seed_meta = native[-2]
        query_inputs, query_meta = native[-1]
        _, seed_ids, _ = run_once(model, sampling, seed_inputs, "apc-doc")
        cold, cold_ids, _ = run_once(model, sampling, query_inputs, "apc-cold-query")
        warm, warm_ids, _ = run_once(model, sampling, query_inputs, "apc-doc")
        same = next((i for i, (x, y) in enumerate(zip(seed_ids, cold_ids)) if x != y),
                    min(len(seed_ids), len(cold_ids)))
        confirmed(cold["num_cached_tokens"], 0, "Cold query APC rows")
        result["native_document_apc"] = {
            "seed_request_id": seed_meta["request_id"], "query_request_id": query_meta["request_id"],
            "unrounded_exact_prefix_prompt_ids": same, "cached_logical_kv_bytes": warm["num_cached_tokens"] * kv_bytes,
            "query_prompt_ids_match": cold_ids == warm_ids,
            "greedy_match": cold["output_token_ids"] == warm["output_token_ids"],
            "cold": cold, "warm": warm,
        }
    write_frozen(args.out, result)


if __name__ == "__main__":
    main()
