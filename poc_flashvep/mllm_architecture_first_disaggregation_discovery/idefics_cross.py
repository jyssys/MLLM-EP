"""Exact separation of IDEFICS visual projection state from decoder queries."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import torch
import torch.nn.functional as F


REVISION = "7065afb8a5bb03015700b315f6751dc3dd4e16eb"
ARMS = (
    "native_recompute", "generic_memoized_d", "generic_packed_d", "projected_handoff_p",
    "generic_visibility_cached_d", "projected_visibility_handoff_p",
    "generic_visibility_padded_d", "projected_visibility_padded_p",
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def measure(device, operation):
    """CUDA elapsed at one device boundary plus separate synchronized host enclosure."""
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        host = time.perf_counter_ns()
        before.record()
        out = operation()
        after.record()
        after.synchronize()
    return out, {"cuda_ms": before.elapsed_time(after), "wall_ms": (time.perf_counter_ns() - host) / 1e6}


def transfer(dest, tensors):
    """Measure an explicit peer-copy enclosure, including destination allocation."""
    for source in {value.device for value in tensors}:
        torch.cuda.synchronize(source)
    torch.cuda.synchronize(dest)
    began = time.perf_counter_ns()
    with torch.cuda.device(dest):
        copied = [tensor.to(dest, non_blocking=True, copy=True) for tensor in tensors]
        torch.cuda.synchronize(dest)
    return copied, {
        "wall_ms": (time.perf_counter_ns() - began) / 1e6,
        "logical_bytes": sum(value.numel() * value.element_size() for value in tensors),
        "copy_count": len(tensors),
    }


class ProjectionBank:
    """Request-scoped output memoization of frozen Linear's independent of query states."""

    def __init__(self, model):
        self.attentions = [block.cross_attn for block in model.model.gated_cross_attn_layers]
        self.modules = [
            item for attention in self.attentions for item in (attention.k_proj, attention.v_proj)
        ]
        self.original = [module.forward for module in self.modules]
        self.original_k_norm = [module.k_layer_norm.forward for module in self.attentions]
        self.width = self.modules[0].out_features
        if any(module.out_features != self.width or module.bias is not None for module in self.modules):
            raise ValueError("Expected uniform, bias-free K/V projections")
        self.packed_weights = None
        self.reset()
        for index, module in enumerate(self.modules):
            module.forward = self.dispatch(index)
        for index, module in enumerate(self.attentions):
            module.k_layer_norm.forward = self.norm_dispatch(index)

    def reset(self, source=None, cache=False, projected=None):
        self.source_signature = self.signature(source.flatten(1, 2)) if source is not None else None
        self.keep = cache
        self.outputs = {} if projected is None else dict(enumerate(projected))
        self.normalized = set(range(len(self.attentions))) if projected is not None else set()
        self.call_count = 0
        self.actual_gemm = 0
        self.cache_hits = 0
        self.fused_gemm = 0
        self.norm_calls = 0
        self.actual_norm = 0

    @staticmethod
    def signature(tensor):
        return (tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype, tensor.device)

    def dispatch(self, index):
        def forward(tensor):
            self.call_count += 1
            if self.keep or index in self.outputs:
                if self.signature(tensor) != self.source_signature:
                    raise RuntimeError("Projection reuse requires the same immutable request tensor")
                if index in self.outputs:
                    self.cache_hits += 1
                    return self.outputs[index]
            value = self.original[index](tensor)
            self.actual_gemm += 1
            if self.keep:
                self.outputs[index] = value
            return value
        return forward

    def norm_dispatch(self, index):
        def forward(tensor):
            self.norm_calls += 1
            if self.keep or index in self.normalized:
                if tensor.data_ptr() != self.outputs[2 * index].data_ptr():
                    raise RuntimeError("Key normalization was not applied to the matching projected memory")
                if index in self.normalized:
                    return tensor
            value = self.original_k_norm[index](tensor)
            self.actual_norm += 1
            if self.keep:
                # Reuse normalized storage in the original Linear output layout.
                # The request holds only one resident copy of each K and V.
                self.outputs[2 * index].copy_(
                    value.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], self.width),
                )
                self.normalized.add(index)
            return value
        return forward

    def prepare_weights(self):
        self.packed_weights = torch.cat([module.weight for module in self.modules], dim=0).contiguous()
        return self.packed_weights.numel() * self.packed_weights.element_size()

    def precompute(self, source):
        if self.signature(source.flatten(1, 2)) != self.source_signature:
            raise RuntimeError("Packed projection source changed")
        values = self.project_packed(source)
        self.outputs = dict(enumerate(values.unbind(0)))

    def project_packed(self, source):
        packed = F.linear(source.flatten(1, 2), self.packed_weights)
        batch, rows = packed.shape[:2]
        values = packed.view(batch, rows, len(self.modules), self.width).permute(2, 0, 1, 3).contiguous()
        self.fused_gemm += 1
        return values

    def project_packed_normalized(self, source):
        slots = list(self.project_packed(source).unbind(0))
        for index, pair in enumerate(self.attentions):
            k = slots[2*index]
            headed = k.unflatten(-1, (pair.num_heads, pair.head_dim)).transpose(1, 2)
            slots[2*index] = self.original_k_norm[index](headed).transpose(1, 2).reshape_as(k)
            self.actual_norm += 1
        return torch.stack(slots)

    def bind_normalized_projections(self, packed):
        self.outputs = dict(enumerate(packed.unbind(0)))
        self.normalized = set(range(len(self.attentions)))

    def snapshot(self):
        return {"projection_calls": self.call_count, "individual_gemms": self.actual_gemm,
                "cache_hits": self.cache_hits, "fused_gemms": self.fused_gemm,
                "key_norm_calls": self.norm_calls, "actual_key_norm_evaluations": self.actual_norm}

    def export(self, source, visible):
        if len(self.normalized) != len(self.attentions):
            raise RuntimeError("Missing normalized key projections")
        return torch.stack([
            select_image_slots(self.outputs[i].unflatten(1, source.shape[1:3]), visible).flatten(1, 2)
            for i in range(len(self.modules))
        ])


def select_image_slots(value, active):
    if active == tuple(range(value.shape[1])):
        return value
    if active == tuple(range(active[0], active[-1] + 1)):
        return value[:, active[0]:active[-1]+1]
    return value.index_select(1, torch.tensor(active, device=value.device))


def restore_masked_slots(packed, original_shape, selected):
    """Keep native SDPA geometry while transferring and projecting live slots."""
    if len(selected) == original_shape[1]:
        return packed
    slots = packed.unflatten(2, (len(selected), original_shape[2]))
    target = slots.new_zeros((slots.shape[0], slots.shape[1], original_shape[1], slots.shape[3], slots.shape[4]))
    target.index_copy_(2, torch.tensor(selected, device=packed.device), slots)
    return target.flatten(2, 3)


def load_input(row, processor, cap):
    from PIL import Image

    images = []
    for item in row["images"]:
        path = Path(item["path"])
        if sha(path) != item["SHA256"]:
            raise ValueError(f"Frozen input image changed for {row['request_id']}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    prompt = ["User:"]
    if row.get("content_order"):
        for piece in row["content_order"]:
            prompt.append(images[piece["image_index"]] if piece["type"] == "image" else piece["text"])
    elif "<image>" in row["question"]:
        texts = row["question"].split("<image>")
        if len(texts) != len(images) + 1:
            raise ValueError("Literal image references must match the original image count")
        for text, image in zip(texts[:-1], images, strict=True):
            prompt.extend((text, image))
        prompt.append(texts[-1])
    else:
        prompt.extend(images)
        prompt.append(row["question"])
    prompt.append("<end_of_utterance>\nAssistant:")
    try:
        processed = processor(text=[prompt], return_tensors="pt", add_end_of_utterance_token=False)
    finally:
        for image in images:
            image.close()
    if (processed["input_ids"].shape[0] != 1 or processed["pixel_values"].shape[1] != len(images)
            or (processed["input_ids"] == processor.image_token_id).sum().item() != len(images)):
        raise ValueError("Original request image order/count changed")
    if processed["input_ids"].shape[1] + cap > 2048:
        raise ValueError("Idefics original prompt and fixed output exceed native context")
    return processed


def encode(model, pixels):
    batch, images = pixels.shape[:2]
    latent = model.model.vision_model(pixel_values=pixels.flatten(0, 1)).last_hidden_state
    latent = model.model.perceiver_resampler(latent)
    return latent.view(batch, images, *latent.shape[1:])


def prompt_forward(model, batch, states):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                 image_attention_mask=batch["image_attention_mask"], perceiver_embeddings=states,
                 use_cache=True, logits_to_keep=1)


def decode(model, cache, state, mask, steps, forced=None, first=None):
    logits = []
    chosen = []
    for step in range(steps):
        token = first if step == 0 else chosen[-1]
        if forced is not None:
            token = forced[step]
        out = model(input_ids=token, past_key_values=cache, perceiver_embeddings=state,
                    image_attention_mask=mask, use_cache=True, logits_to_keep=1)
        logits.append(out.logits[0, -1])
        chosen.append(out.logits.argmax(dim=-1))
    return torch.stack(logits), chosen, cache


def copy_text_cache(source, device):
    from transformers import DynamicCache

    tensors = [value for layer in source.layers for value in (layer.keys, layer.values)]
    copied, stats = transfer(device, tensors)
    return DynamicCache(zip(copied[::2], copied[1::2], strict=True)), stats


def extraction(pbank, d_bank, prefill, d_device, p_state, d_mask, active, d_model, mode, forced):
    if mode not in ARMS:
        raise ValueError(mode)
    capture_stats = pbank.snapshot()
    complete_started = time.perf_counter_ns()
    cache, cache_move = copy_text_cache(prefill.past_key_values, d_device)
    project_pack = None
    projected_move = None
    owner_projected = mode.startswith("projected_")
    compact = "visibility" in mode
    same_shape = "padded" in mode
    selected = active if compact and active else tuple(range(p_state.shape[1]))
    if compact:
        compact_source, select_source = measure(p_state.device, lambda: select_image_slots(p_state, selected))
        if not same_shape:
            d_mask = d_mask[..., list(selected)]
    else:
        compact_source, select_source = p_state, None
    if owner_projected:
        if len(pbank.outputs) != len(pbank.modules):
            raise RuntimeError("Every source K/V projection must have been captured")
        packed, project_pack = measure(p_state.device, lambda: pbank.export(p_state, selected))
        (packed_d,), projected_move = transfer(d_device, [packed])
        projections = packed_d.unbind(0)
        # Stock forward checks state shape and builds attention masks; projected
        # memory supplies actual K/V. A cheap zero placeholder carries shape only.
        shape = p_state.shape if same_shape else compact_source.shape
        state = torch.zeros(shape, device=d_device, dtype=p_state.dtype)
        image_move = {"logical_bytes": 0, "wall_ms": 0.0, "copy_count": 0}
        d_bank.reset(state, projected=projections if not same_shape else None)
    else:
        (state,), image_move = transfer(d_device, [compact_source])
        d_bank.reset(state, cache=mode != "native_recompute")
    packed_ms = None
    restore_ms = None
    if mode in {"generic_packed_d", "generic_visibility_cached_d"}:
        _, packed_ms = measure(d_device, lambda: d_bank.precompute(state))
    if same_shape:
        if not owner_projected:
            live_source = state
            state = torch.zeros(p_state.shape, device=d_device, dtype=p_state.dtype)
            d_bank.reset(state)
            packed_d, packed_ms = measure(d_device, lambda: d_bank.project_packed_normalized(live_source))
        restored, restore_ms = measure(d_device, lambda: restore_masked_slots(packed_d, p_state.shape, selected))
        d_bank.bind_normalized_projections(restored)
    outputs, timing = measure(d_device, lambda: decode(
        d_model, cache, state, d_mask, len(forced), forced=forced,
    ))
    overall_ms = (time.perf_counter_ns() - complete_started) / 1e6
    cross_allocations = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                         for v in d_bank.outputs.values()}
    result = {
        "mode": mode, "p_cross_projection": capture_stats, "text_kv_copy": cache_move,
        "visual_raw_copy": image_move, "visual_projected_copy": projected_move,
        "pack_on_P": project_pack, "packed_project_on_D": packed_ms, "select_source_on_P": select_source,
        "restore_masked_geometry_on_D": restore_ms,
        "decode": timing, "d_cross_projection": d_bank.snapshot(),
        "decode_source_slots": list(selected),
        "d_cross_cache_resident_bytes": sum(v.numel()*v.element_size() for v in d_bank.outputs.values()),
        "d_cross_cache_storage_bytes": sum(cross_allocations.values()),
        "measured_post_prefill_wall_ms": overall_ms,
    }
    result["component_enclosure_sum_ms"] = (
        cache_move["wall_ms"] + image_move["wall_ms"] + timing["wall_ms"]
        + sum(stage["wall_ms"] for stage in (project_pack, projected_move, packed_ms, select_source, restore_ms)
              if stage is not None)
    )
    return result, outputs


def numerical(control, candidate):
    ref, _, ref_cache = control
    target, _, target_cache = candidate
    delta = target.float() - ref.float()
    return {
        "step_reference_count": target.shape[0], "logits_bitwise": bool(torch.equal(ref, target)),
        "logit_max_abs": float(delta.abs().max()),
        "logit_rel_l2": float(delta.norm() / ref.float().norm().clamp_min(1e-12)),
        "greedy_matches": int((target.argmax(-1) == ref.argmax(-1)).sum()),
        "self_cache_layers": len(ref_cache.layers),
        "self_cache_exact_layers": sum(
            torch.equal(base.keys, other.keys) and torch.equal(base.values, other.values)
            for base, other in zip(ref_cache.layers, target_cache.layers, strict=True)
        ),
    }


def verify_projected_state(p_bank, d_bank, p_state, d_state):
    """Compare P-captured K/V to independent native D projections outside timing."""
    (p_projected,), _ = transfer(d_state.device, [p_bank.export(p_state, tuple(range(p_state.shape[1])))])
    latent = d_state.flatten(1, 2)
    keys, values = [], []
    for i, attention in enumerate(d_bank.attentions):
        key = d_bank.original[2*i](latent)
        normed = d_bank.original_k_norm[i](
            key.unflatten(-1, (attention.num_heads, attention.head_dim)).transpose(1, 2),
        ).transpose(1, 2).reshape_as(key)
        actual_value = d_bank.original[2*i+1](latent)
        keys.append(bool(torch.equal(p_projected[2*i], normed)))
        values.append(bool(torch.equal(p_projected[2*i+1], actual_value)))
    return {"exact_post_norm_k_layers": sum(keys), "exact_v_layers": sum(values), "cross_layers": len(keys)}


def run_request(row, processor, p_model, d_model, p_bank, d_bank, args, raw_stock_probe=False):
    p_dev, d_dev = next(p_model.parameters()).device, next(d_model.parameters()).device
    inputs = load_input(row, processor, args.steps + 1)
    p_batch = {name: value.to(p_dev) for name, value in inputs.items() if name != "pixel_values"}
    d_batch = {name: value.to(d_dev) for name, value in inputs.items() if name != "pixel_values"}
    pixels = inputs["pixel_values"].to(p_dev, dtype=torch.bfloat16)
    p_states, encoding = measure(p_dev, lambda: encode(p_model, pixels))
    p_bank.reset()
    stock_pixel_check = None
    if raw_stock_probe:
        try:
            output = p_model(**p_batch, pixel_values=pixels, use_cache=True, logits_to_keep=1)
            isolated = prompt_forward(p_model, p_batch, p_states)
            stock_pixel_check = {"success": True, "staged_logits_exact": bool(torch.equal(
                output.logits, isolated.logits,
            ))}
        except RuntimeError as error:
            if "shape" not in str(error):
                raise
            stock_pixel_check = {"success": False, "error": str(error)}
    p_bank.reset(p_states, cache=True)
    control_prompt = prompt_forward(p_model, p_batch, p_states)
    (d_states,), _ = transfer(d_dev, [p_states])
    state_witness = verify_projected_state(p_bank, d_bank, p_states, d_states)
    p_bank.reset()
    ref_cache, _ = copy_text_cache(control_prompt.past_key_values, d_dev)
    d_bank.reset()
    first = control_prompt.logits.argmax(dim=-1).to(d_dev)
    ctrl = decode(d_model, ref_cache, d_states, d_batch["image_attention_mask"][:, -1:, :], args.steps, first=first)
    forced = [first, *ctrl[1][:-1]]
    logits, tokens, _ = ctrl
    full_trajectory = [int(first.item()), *(int(token.item()) for token in tokens)]
    original_eos = p_model.generation_config.eos_token_id
    eoses = {original_eos} if isinstance(original_eos, int) else set(original_eos)
    eoses.add(processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>"))
    first_eos = next((i + 1 for i, token in enumerate(full_trajectory) if token in eoses), None)
    records = []
    mask_cpu = inputs["image_attention_mask"][0, -1].tolist()
    active = tuple(i for i, enabled in enumerate(mask_cpu) if enabled)
    for rep in range(args.repeats):
        for arm in ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]:
            p_bank.reset(p_states, cache=arm.startswith("projected_"))
            prefill, p_meas = measure(p_dev, lambda: prompt_forward(p_model, p_batch, p_states))
            result, tests = extraction(
                p_bank, d_bank, prefill, d_dev, p_states, d_batch["image_attention_mask"][:, -1:, :],
                active, d_model, arm, forced,
            )
            result.update({
                "repeat": rep, "text_prefill": p_meas, "first_token_p_logits_exact": torch.equal(prefill.logits, control_prompt.logits),
                "correctness": numerical(ctrl, tests),
            })
            records.append(result)
            p_bank.reset()
            d_bank.reset()
    return {
        "request_id": row["request_id"], "study_source": row.get("study_source"), "evidence": "ISOLATED_GPU_MEASUREMENT",
        "vision_perceiver": encoding, "stock_full_pixel_preflight": stock_pixel_check,
        "full_projected_state_witness": state_witness,
        "text_prompt_tokens": inputs["input_ids"].shape[1], "first_eos_or_none": first_eos,
        "forced_tokens_from_native": [int(token.item()) for token in forced], "native_trajectory": full_trajectory,
        "vision_latent_shape": list(p_states.shape), "image_mask_rows": inputs["image_attention_mask"][0].int().tolist(),
        "per_image_visible_prompt_rows": inputs["image_attention_mask"][0].sum(0).int().tolist(),
        "last_image_mask": inputs["image_attention_mask"][0, -1].int().tolist(),
        "vision_dtype": str(p_states.dtype), "experiments": records,
    }


def main():
    from transformers import AutoProcessor, IdeficsForVisionText2Text, __version__ as transformers_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--request", action="append", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.snapshot.name != REVISION or not args.steps > 0 or not args.repeats > 0:
        raise ValueError("Use the pinned original BF16 weights and positive replay/repeat counts")
    if torch.cuda.device_count() != 2 or not torch.cuda.can_device_access_peer(0, 1):
        raise RuntimeError("Exactly two explicitly allocated CUDA devices with peer access are required")
    manifest = json.loads(args.manifest.read_text())["requests"]
    selected = {row["request_id"]: row for row in manifest if row["request_id"] in args.request}
    if len(set(args.request)) != len(args.request) or selected.keys() != set(args.request):
        raise ValueError("Require unique frozen request IDs")
    args.run.mkdir(parents=True, exist_ok=False)
    configuration = {
        "checkpoint": "HuggingFaceM4/idefics-9b-instruct", "revision": REVISION,
        "manifest_SHA256": sha(args.manifest), "source_SHA256": sha(Path(__file__)),
        "model_config_SHA256": sha(args.snapshot / "config.json"),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers_version,
        "cuda_peer_access": [torch.cuda.can_device_access_peer(0, 1), torch.cuda.can_device_access_peer(1, 0)],
        "gpu_properties": [str(torch.cuda.get_device_properties(i)) for i in (0, 1)],
        "dtype": "bfloat16", "attention": "sdpa", "steps": args.steps, "repeats": args.repeats, "request_ids": args.request,
    }
    write(args.run / "config.json", configuration)
    with torch.inference_mode():
        processor = AutoProcessor.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
        def load(i):
            return IdeficsForVisionText2Text.from_pretrained(
                args.snapshot, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
                use_safetensors=True,
            ).eval().to(torch.device("cuda", i))
        p_model, d_model = load(0), load(1)
        config = p_model.config
        if (config.num_hidden_layers, config.cross_layer_interval, config.perceiver_config.resampler_n_latents) != (32, 4, 64):
            raise ValueError("Checkpoint architecture no longer matches the analysis contract")
        p_bank, d_bank = ProjectionBank(p_model), ProjectionBank(d_model)
        packed_bytes = d_bank.prepare_weights()
        write(args.run / "architecture.json", {
            "self_attention_layers": len(p_model.model.layers), "cross_attention_layers": len(p_bank.modules)//2,
            "cross_every": config.cross_layer_interval, "qk_layer_norms": config.qk_layer_norms,
            "latent_rows_per_image": config.perceiver_config.resampler_n_latents,
            "query_head_count": config.num_attention_heads, "vision_width": config.vision_config.embed_dim,
            "text_width": config.hidden_size, "extra_one_time_packed_weights_bytes": packed_bytes,
        })
        # Exclude one full warm-up per arm from the timing cohort.
        warm = argparse.Namespace(steps=min(4, args.steps), repeats=1)
        write(args.run / "warmup.json", run_request(selected[args.request[0]], processor, p_model, d_model, p_bank, d_bank,
                                                    warm, raw_stock_probe=True))
        for index, request_id in enumerate(args.request):
            result = run_request(selected[request_id], processor, p_model, d_model, p_bank, d_bank, args)
            write(args.run / f"request_{index:03d}.json", result)
            print("IDEFICS_CROSS", request_id, len(result["experiments"]), median(
                row["decode"]["cuda_ms"] for row in result["experiments"]
            ), flush=True)
    write(args.run / "complete.json", {"requests": len(selected), "experiments": len(selected)*args.repeats*len(ARMS)})


if __name__ == "__main__":
    main()
