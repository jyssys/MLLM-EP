"""Native InternVL3 diagnostic with frozen images and optional real post-RoPE Q/K/V."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

from .download_models import CHECKPOINTS


TARGET_IMAGE_COUNTS = (1, 3, 8, 16)
IMAGE_PLACEHOLDER = "<image>"
REVIEWED_LOCAL_CONFIG = {
    "config.json": "cd0c54d0aa85fcd979cd2af524c0c08e26cdca278f05e96b635f298e34afae39",
    "configuration_intern_vit.py": "22dd687a5f142985407675fc6b1dc688c2ded93e54192543ecf28ef495a6efa8",
    "configuration_internvl_chat.py": "6524e6deae6c318543d1022a599f011120f5a970eaa25f1118e7f5d0e346710c",
    "tokenizer_config.json": "705d8fbba57dae6bff1e6a1d695ba116fbf27761a4e8373bb68f5d4f0d88f19d",
}


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")


def choose_cases(manifest):
    rows = json.loads(manifest.read_text())["requests"]
    if len(rows) != 100:
        raise ValueError("Expected the complete frozen Main100 manifest")
    entries = []
    for target in TARGET_IMAGE_COUNTS:
        row = min(rows, key=lambda item: (abs(item["image_count"] - target), item["request_id"]))
        if row["image_count"] < 1 or row["image_count"] != len(row["images"]):
            raise ValueError("Invalid frozen image roster")
        entries.append((target, row))
    if len({row["request_id"] for _, row in entries}) != len(entries):
        raise ValueError("Cohort selection requires four distinct frozen requests")
    return entries


def prepare_case(target, row, tokenizer):
    from PIL import Image

    question = row.get("question") or row["question_original"]
    paths = [Path(image["path"]) for image in row["images"]]
    images = []
    for path, item in zip(paths, row["images"], strict=True):
        if file_digest(path) != item["SHA256"]:
            raise ValueError(f"Frozen media digest differs for {row['request_id']}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    placeholders = question.count(IMAGE_PLACEHOLDER)
    if placeholders == 0:
        content = (IMAGE_PLACEHOLDER + "\n") * len(images) + question
    elif placeholders == len(images):
        content = question
    else:
        raise ValueError(f"Unexpected image placeholders for {row['request_id']}")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    )
    meta = {
        "request_id": row["request_id"], "dataset": row["source"], "target_images": target,
        "actual_images": len(images), "image_sizes": [list(image.size) for image in images],
        "original_question_sha256": hashlib.sha256(row["question_original"].encode()).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "same_request_qwenvl_tokens": row["total_input_tokens"],
    }
    return {"prompt": prompt, "multi_modal_data": {"image": images}}, meta


def runtime_config(model):
    for path, digest in REVIEWED_LOCAL_CONFIG.items():
        if file_digest(model / path) != digest:
            raise ValueError(f"Unreviewed or changed local snapshot file {path}")

    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.tokenizers import get_tokenizer
    from vllm.transformers_utils.config import get_hf_text_config

    # vLLM instantiates the installed InternVL implementation. The audited local
    # InternVL configuration class is the only custom checkpoint code required.
    config = ModelConfig(model=str(model), dtype="bfloat16", model_impl="vllm", trust_remote_code=True)
    text = get_hf_text_config(config.hf_config)
    tokenizer = get_tokenizer(str(model), trust_remote_code=False)
    processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tokenizer).info.get_hf_processor()
    geometry = {key: getattr(text, key) for key in (
        "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
    )}
    geometry["head_dim"] = geometry["hidden_size"] // geometry["num_attention_heads"]
    return config, tokenizer, processor, geometry, text.rope_parameters


def prepare(manifest, model, output_tokens):
    config, tokenizer, processor, geometry, rope = runtime_config(model)
    cases = []
    for target, row in choose_cases(manifest):
        inputs, meta = prepare_case(target, row, tokenizer)
        features = processor(text=inputs["prompt"], images=inputs["multi_modal_data"]["image"], return_tensors="pt")
        tokens = features["input_ids"]
        image_slots = int((tokens == processor.ctx_image_token_id).sum())
        meta.update({
            "prefill_tokens": tokens.shape[-1], "image_slots": image_slots,
            "tiles_per_image": features["image_num_patches"].tolist(),
            "eligible": tokens.shape[-1] + output_tokens <= config.max_model_len,
        })
        cases.append((inputs, meta))
    config_meta = {
        "native_architecture": config.architectures,
        "max_model_len": config.max_model_len, "head_geometry": geometry,
        "rope": rope, "manifest_sha256": file_digest(manifest), "processor": type(processor).__name__,
        "image_context_token_id": processor.ctx_image_token_id,
        "checkpoint": CHECKPOINTS["internvl3_8b"][0], "revision": CHECKPOINTS["internvl3_8b"][1],
    }
    return cases, config_meta


def model_payload(model):
    import torch

    bytes_by_group = {"vision_encoder": 0, "vision_projector": 0, "language_model": 0, "other": 0}
    mapping = {"vision_model": "vision_encoder", "mlp1": "vision_projector", "language_model": "language_model"}
    for name, tensor in model.named_parameters():
        root = name.split(".", maxsplit=1)[0]
        bytes_by_group[mapping.get(root, "other")] += tensor.numel() * tensor.element_size()
    return {
        "actual_model_class": type(model).__name__, "parameter_bytes_by_group": bytes_by_group,
        "torch_allocated_bytes": torch.cuda.memory_allocated(), "torch_reserved_bytes": torch.cuda.memory_reserved(),
    }


def memory_peaks(_model, *, reset=False):
    import torch

    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {"max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved()}


def start_qkv_capture(model, prompt_tokens):
    """Observe exactly one request outside the timed diagnostic."""
    state = {"keys": [], "values": [], "count": 0, "query": None}

    def capture(_attention, args):
        q, k, v = args[:3]
        if state["query"] is None:
            after = state["count"] + k.shape[0]
            if state["count"] >= prompt_tokens and q.shape[0] == 1:
                state["query"] = q.detach().to(device="cpu", copy=True)
            state["keys"].append(k.detach().to(device="cpu", copy=True))
            state["values"].append(v.detach().to(device="cpu", copy=True))
            state["count"] = after

    attention = model.language_model.model.layers[0].self_attn.attn
    state["hook"] = attention.register_forward_pre_hook(capture)
    model._eada_qkv_capture = state


def finish_qkv_capture(model):
    import torch

    state = model._eada_qkv_capture
    state["hook"].remove()
    del model._eada_qkv_capture
    if state["query"] is None:
        raise RuntimeError("First native single-token decode query was not captured")
    return {"q": state["query"], "k": torch.cat(state["keys"]), "v": torch.cat(state["values"])}


def run_native(args, cases, geometry):
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(args.model), model_impl="vllm", trust_remote_code=True, dtype="bfloat16",
        enforce_eager=True, enable_prefix_caching=False, disable_log_stats=False,
        tensor_parallel_size=1, max_num_batched_tokens=8192, max_num_seqs=4,
        kv_cache_memory_bytes=args.kv_gib * 2**30,
        limit_mm_per_prompt={"image": max(meta["actual_images"] for _, meta in cases), "video": 0},
        mm_processor_cache_gb=0,
    )
    payload, = engine.apply_model(model_payload)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)
    eligible = [(inputs, meta) for inputs, meta in cases if meta["eligible"]]
    if not eligible:
        raise ValueError("No frozen InternVL case fits the native context limit")
    engine.generate([eligible[0][0]], sampling_params=sampling, use_tqdm=False)
    results = []
    peaks = {}
    for inputs, meta in eligible:
        engine.apply_model(partial(memory_peaks, reset=True))
        for repeat in range(args.repeats):
            started = time.perf_counter()
            output, = engine.generate([inputs], sampling_params=sampling, use_tqdm=False)
            elapsed = (time.perf_counter() - started) * 1000
            generated, = output.outputs
            prompt_tokens = len(output.prompt_token_ids)
            if prompt_tokens != meta["prefill_tokens"] or len(generated.token_ids) != args.tokens:
                raise RuntimeError("Native model saw a different length from the frozen prepared workload")
            metrics = asdict(output.metrics) if output.metrics is not None else None
            results.append({
                "request_id": meta["request_id"], "repeat": repeat, "wall_ms": elapsed,
                "model_request_stats": metrics, "prompt_tokens": prompt_tokens,
                "output_token_ids": generated.token_ids, "completion_reason": generated.finish_reason,
            })
        peaks[meta["request_id"]], = engine.apply_model(memory_peaks)
    qkv = None
    if args.capture_qkv is not None:
        import torch

        inputs, meta = eligible[-1]
        length = meta["prefill_tokens"]
        engine.apply_model(partial(start_qkv_capture, prompt_tokens=length))
        native, = engine.generate([inputs], sampling_params=sampling, use_tqdm=False)
        captured, = engine.apply_model(finish_qkv_capture)
        heads = geometry["head_geometry"]
        dim = heads["head_dim"]
        q = captured["q"].reshape(1, heads["num_attention_heads"], dim)
        k = captured["k"].reshape(-1, heads["num_key_value_heads"], dim)
        v = captured["v"].reshape_as(k)
        if k.shape[0] != length + 1:
            raise RuntimeError("The captured KV is not the full prefix and first decoded row")
        visual = torch.tensor([*native.prompt_token_ids, -1]).eq(geometry["image_context_token_id"])
        if visual.shape[0] != k.shape[0] or visual.sum().item() != meta["image_slots"]:
            raise RuntimeError("Native prefill visual rows differ from the frozen processor output")
        provenance = {"checkpoint": geometry["checkpoint"], "revision": geometry["revision"],
                      "request_id": meta["request_id"], "layer_index": 0,
                      "capture_point": "native_post_rope_first_decode"}
        args.capture_qkv.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": q, "k": k, "v": v, "visual": visual, "provenance": provenance}, args.capture_qkv)
        qkv = provenance | {"sha256": file_digest(args.capture_qkv), "visual_rows": int(visual.sum()),
                            "q_shape": list(q.shape), "kv_shape": list(k.shape), "dtype": str(q.dtype)}
    return {"scope": "ISOLATED_NATIVE_INFERENCE", "allocated_kv_budget_bytes": args.kv_gib * 2**30,
            "model": payload, "requests": results, "case_memory_peaks": peaks, "qkv_capture": qkv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", default=16, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--kv-gib", default=8, type=int)
    parser.add_argument("--capture-qkv", type=Path, help="Optional external file with actual bfloat16 tensors")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.tokens < 16 or args.repeats < 1 or args.kv_gib < 1:
        parser.error("Need at least 16 decode tokens, one repeat and 1 GiB of KV budget")
    # Collect measurements in the same process as the only worker so the
    # diagnostic hooks do not require serialization of Python callables.
    if not args.preflight_only:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    cases, config = prepare(args.manifest, args.model.resolve(), args.tokens)
    receipt = {"config": config, "cases": [meta for _, meta in cases], "scope": "CPU_PREFLIGHT"}
    if not args.preflight_only:
        receipt.update(run_native(args, cases, config))
    write_json(args.output, receipt)


if __name__ == "__main__":
    main()
