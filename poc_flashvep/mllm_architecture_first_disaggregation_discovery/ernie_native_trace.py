"""Trace actual vLLM ERNIE-VL expert calls on intact, independently served requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def install_trace(model):
    trace = []
    handles = []
    allocated = {}
    frame = {"index": -1}
    pending = {}

    def new_frame(_module, _args, _kwargs):
        frame["index"] += 1

    handles.append(model.register_forward_pre_hook(new_frame, with_kwargs=True))

    def track(group, name):
        def hook(_module, args, kwargs):
            tensor = next((value for value in (*args, *kwargs.values()) if torch.is_tensor(value)), None)
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            pending[name] = (start, {"invocation": frame["index"], "module": name, "group": group,
                                     "first_tensor_shape": list(tensor.shape) if tensor is not None else None})
        return hook

    def finish(name):
        def hook(_module, _args, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start, row = pending.pop(name)
            trace.append((start, end, row))
        return hook

    for name, module in model.named_modules():
        part = name.rsplit(".", 1)[-1]
        if part not in ("vision_experts", "text_experts", "vision_experts_gate", "text_experts_gate",
                        "vision_model", "resampler_model", "shared_experts"):
            continue
        handles.append(module.register_forward_pre_hook(track(part, name), with_kwargs=True))
        handles.append(module.register_forward_hook(finish(name)))
        for parameter in module.parameters():
            allocated.setdefault(part, {})[id(parameter)] = parameter.numel() * parameter.element_size()
    if not any(group == "vision_experts" for group in allocated):
        raise RuntimeError("No named visual bank was found in the native model")
    model._architecture_trace = (trace, handles, frame)
    appearances = {}
    for tensors in allocated.values():
        for ident in tensors:
            appearances[ident] = appearances.get(ident, 0) + 1
    return {
        "class": type(model).__name__, "group_weight_payload": {
            part: sum(tensors.values()) for part, tensors in allocated.items()
        },
        "group_exclusive_weight_payload": {
            part: sum(size for ident, size in tensors.items() if appearances[ident] == 1)
            for part, tensors in allocated.items()
        },
        "total_parameter_bytes": sum(tensor.numel() * tensor.element_size() for tensor in model.parameters()),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "groups": sorted(allocated),
    }


def drain_trace(model):
    trace, _, frame = model._architecture_trace
    torch.cuda.synchronize()
    result = [{**row, "cuda_stream_ms": start.elapsed_time(end)} for start, end, row in trace]
    trace.clear()
    frame["index"] = -1
    return result


def arm_text_continuation_elision(model):
    """Release only the exclusive visual-bank parameters after one image prefill.

    This is the terminal request in an eager process. Any later visual use
    fails at the module entrance. The operation preserves all KV and shared
    parameters and measures live Torch allocation before the first decode.
    """
    trace, _, frame = model._architecture_trace
    targets = []
    ownership = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        ownership.setdefault(id(parameter), []).append(name)
    names = ("vision_experts", "vision_experts_gate")
    skipped_shared = 0
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in names:
            continue
        for path, parameter in module.named_parameters():
            owner, _, leaf = path.rpartition(".")
            if len(ownership[id(parameter)]) != 1:
                skipped_shared += 1
                continue
            if leaf not in {"weight", "w13_weight", "w2_weight"} or parameter.device.type != "cuda":
                raise ValueError(f"Unexpected visual bank parameter {name}.{path}")
            targets.append((module.get_submodule(owner), leaf))
    if not targets:
        raise RuntimeError("The loaded model has no eligible visual expert tensors")
    model._architecture_elision = {}

    def reject(_module, _args):
        raise RuntimeError("A visual bank was reached after entering text-only continuation")

    def release(_module, _args, _kwargs):
        if frame["index"] != 1:
            return
        if not any(row["group"] == "vision_experts" and row["invocation"] == 0 for _, _, row in trace):
            raise ValueError("Elision requires exactly one observed image prefill before decode")
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        eligible_bytes = 0
        for owner, leaf in targets:
            weight = owner.get_parameter(leaf)
            eligible_bytes += weight.numel() * weight.element_size()
            setattr(owner, leaf, torch.nn.Parameter(torch.empty(0, dtype=weight.dtype, device="meta"),
                                                    requires_grad=False))
        del weight
        for name, module in model.named_modules():
            if name.rsplit(".", 1)[-1] in names:
                module.register_forward_pre_hook(reject, prepend=True)
        torch.cuda.synchronize()
        model._architecture_elision.update({"before_bytes": before, "after_bytes": torch.cuda.memory_allocated(),
                                            "unique_parameter_bytes": eligible_bytes, "tensor_count": len(targets),
                                            "kept_shared_tensor_references": skipped_shared,
                                            "scope": "Live CUDA allocated bytes after image prefill. KV retained."})

    model.register_forward_pre_hook(release, with_kwargs=True)


def read_elision(model):
    return model._architecture_elision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--trust-reviewed-custom-code", action="store_true")
    parser.add_argument("--check-elision", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())["requests"][:args.count]
    media = [Path(item["path"]).resolve() for row in rows for item in row["images"]]
    if not rows or len(media) != len(rows):
        raise ValueError("This diagnostic expects intact one-image requests")
    from vllm import LLM, SamplingParams

    kwargs = {"model": str(args.snapshot), "dtype": "bfloat16", "tensor_parallel_size": 1,
              "enforce_eager": True, "enable_prefix_caching": False, "disable_log_stats": True,
              "max_model_len": 8192, "max_num_seqs": 4, "max_num_batched_tokens": 8192,
              "limit_mm_per_prompt": {"image": 1, "video": 0}, "kv_cache_memory_bytes": 8 * 1024**3,
              "allowed_local_media_path": os.path.commonpath([str(path.parent) for path in media]),
              "trust_remote_code": args.trust_reviewed_custom_code}
    llm = LLM(**kwargs)
    settings = llm.apply_model(install_trace)
    trace = []
    final_chat = None
    sampling = SamplingParams(temperature=0., min_tokens=args.steps, max_tokens=args.steps)
    for index, row in enumerate(rows):
        path = media[index]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["images"][0]["SHA256"]:
                raise ValueError("The media hash differs from the frozen request")
        question = row["question"] or row["question_original"]
        content = [{"type": "image_url", "image_url": {"url": path.as_uri()}},
                   {"type": "text", "text": question}]
        start = time.monotonic()
        final_chat = [{"role": "user", "content": content}]
        outputs = llm.chat(final_chat, sampling, use_tqdm=False)
        elapsed = (time.monotonic() - start) * 1000
        final_tokens = list(outputs[0].outputs[0].token_ids)
        trace.append({"request_id": row["request_id"], "condition": "image_text", "prompt_tokens":
                      len(outputs[0].prompt_token_ids), "output_tokens": len(final_tokens),
                      "diagnostic_wall_ms": elapsed, "trace": llm.apply_model(drain_trace)})
    control = llm.chat([{"role": "user", "content": "Continue counting: one, two, three,"}],
                       sampling, use_tqdm=False)
    trace.append({"condition": "text_only_control", "prompt_tokens": len(control[0].prompt_token_ids),
                  "output_tokens": len(control[0].outputs[0].token_ids), "trace": llm.apply_model(drain_trace)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump({"native": settings, "config": kwargs, "requests": trace,
                   "evidence": "DIAGNOSTIC", "phase": "Serial, fixed-length, hooks enabled"},
                  stream, indent=2)
        stream.write("\n")
    if args.check_elision:
        llm.apply_model(arm_text_continuation_elision)
        candidate = llm.chat(final_chat, sampling, use_tqdm=False)
        candidate_tokens = list(candidate[0].outputs[0].token_ids)
        measured = llm.apply_model(read_elision)
        result = {"request_id": rows[-1]["request_id"], "baseline_token_ids": final_tokens,
                  "candidate_token_ids": candidate_tokens, "equal_tokens": final_tokens == candidate_tokens,
                  "memory": measured, "trace": llm.apply_model(drain_trace),
                  "evidence": "DIAGNOSTIC", "scope": "Isolated native single-device terminal continuation"}
        with args.out.with_name(args.out.stem + "_elision.json").open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
