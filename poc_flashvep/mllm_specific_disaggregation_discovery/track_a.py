"""Replay real Qwen KV and decode on H200 with exact FlashAttention-3 shards."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from .remote_attention import DistributedAttention, PackedKV, PeerCopies, bytes_of
from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt

MODES = ("local", "balanced", "prefix", "visual", "equal_count_prefix")


def collect_prompt(model, processor, row, device):
    images = []
    for image in row["images"]:
        path = Path(image["path"])
        if rt.sha(path) != image["SHA256"]:
            raise ValueError(f"Changed image for {row['request_id']}")
        with Image.open(path) as item:
            images.append(item.convert("RGB"))
    try:
        batch = processor(text=[row["prompt"]], images=images, return_tensors="pt")
    finally:
        for image in images:
            image.close()
    ids = batch["input_ids"]
    path = Path(row["token_metadata_file"])
    if rt.sha(path) != row["token_metadata_SHA256"]:
        raise ValueError(f"Changed processor receipt for {row['request_id']}")
    with np.load(path) as metadata:
        if list(ids[0].tolist()) != metadata["input_ids"].reshape(-1).tolist():
            raise ValueError("Processor token IDs differ from frozen contract")
        expected_pos = metadata["positions"].reshape(3, 1, -1)
        rope_delta = int(metadata["rope_delta"].reshape(-1)[0])
    if batch["image_grid_thw"].tolist() != row["grid_thw"]:
        raise ValueError("Processor grid differs from frozen contract")
    prepare_args = {name: batch[name].to(device) for name in ("input_ids", "image_grid_thw", "mm_token_type_ids")}
    pixels = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    model.model.rope_deltas = None
    output = model(**prepare_args, pixel_values=pixels, use_cache=True, logits_to_keep=1)
    actual_pos, actual_delta = model.model.get_rope_index(**prepare_args)
    if not np.array_equal(actual_pos.cpu().numpy(), expected_pos) or int(actual_delta.item()) != rope_delta:
        raise ValueError("MRoPE positions differ from frozen contract")
    visual = ids[0].eq(model.config.image_token_id).nonzero().flatten()
    if len(visual) != row["visual_tokens"]:
        raise ValueError("Visual KV mask differs from processor metadata")
    return output, visual, rope_delta


def shard_indices(length, visual, mode):
    if mode == "local":
        remote = torch.empty(0, dtype=torch.long)
    elif mode == "balanced":
        remote = torch.arange(length // 2)
    elif mode == "prefix":
        remote = torch.arange(length)
    elif mode == "visual":
        remote = visual
    elif mode == "equal_count_prefix":
        remote = torch.arange(len(visual))
    else:
        raise ValueError(mode)
    mask = torch.ones(length, dtype=torch.bool)
    mask[remote] = False
    return mask.nonzero().flatten(), remote


class AppendingKV:
    """Live decode-side KV allocation with room for the declared output reserve."""
    def __init__(self, keys, values, indices, reserve):
        self.prefix = len(indices)
        shape = (self.prefix + reserve, *keys.shape[1:])
        self.keys = torch.empty(shape, device=keys.device, dtype=keys.dtype)
        self.values = torch.empty_like(self.keys)
        self.keys[: self.prefix].copy_(keys.index_select(0, indices))
        self.values[: self.prefix].copy_(values.index_select(0, indices))

    def put(self, key, value, step):
        index = self.prefix + step
        self.keys[index].copy_(key[0])
        self.values[index].copy_(value[0])

    def view(self, step, cuts):
        end = self.prefix + step + 1
        # Construct a view with prebuilt lengths. No per-step host→device copy.
        return KVView(self.keys[:end], self.values[:end], (end,), end, cuts[step])


@dataclass
class KVView:
    keys: torch.Tensor
    values: torch.Tensor
    lengths: tuple
    max_length: int
    cut: torch.Tensor


class DecodeState:
    def __init__(self, layers, length, visual, mode, reserve):
        self.local = []
        self.remote = []
        self.mode = mode
        self.local_indices, self.remote_indices = shard_indices(length, visual, mode)
        loc_idx = self.local_indices.to("cuda:0")
        rem_idx = self.remote_indices.to("cuda:0")
        lengths = torch.arange(len(loc_idx) + 1, len(loc_idx) + reserve + 1, dtype=torch.int32)
        self.local_cuts = torch.stack((torch.zeros_like(lengths), lengths), dim=1).to("cuda:0")
        for layer in layers:
            keys, values = (value[0].transpose(0, 1) for value in (layer.keys, layer.values))
            self.local.append(AppendingKV(keys, values, loc_idx, reserve))
            if len(rem_idx):
                with torch.cuda.device(1):
                    gathered_k = keys.index_select(0, rem_idx).to("cuda:1")
                    gathered_v = values.index_select(0, rem_idx).to("cuda:1")
                    self.remote.append(PackedKV(gathered_k, gathered_v, (len(rem_idx),)))

    def allocated_kv(self):
        return {
            "D_prompt_rows": len(self.local_indices), "P_prompt_rows": len(self.remote_indices),
            "D_allocated_bytes": sum(bytes_of(layer.keys) + bytes_of(layer.values) for layer in self.local),
            "P_allocated_bytes": sum(bytes_of(layer.keys) + bytes_of(layer.values) for layer in self.remote),
        }


class FixedBatchKV:
    """Ragged pack with one real write slot at the end of every request."""
    def __init__(self, pack):
        self.pack = pack
        self.positions = pack.cut[1:].long() - 1

    def put(self, key, value, step):
        self.pack.keys.index_copy_(0, self.positions, key)
        self.pack.values.index_copy_(0, self.positions, value)

    def view(self, step, cuts):
        return self.pack


class FixedBatchState:
    def __init__(self, caches, visuals, mode):
        self.local, self.remote = [], []
        self.local_cuts = None
        membership = [shard_indices(cache.get_seq_length(), visual, mode)
                      for cache, visual in zip(caches, visuals, strict=True)]
        for layer_index in range(len(caches[0].layers)):
            local, remote = [], []
            for cache, (local_indices, remote_indices) in zip(caches, membership, strict=True):
                layer = cache.layers[layer_index]
                keys = layer.keys[0].transpose(0, 1)
                values = layer.values[0].transpose(0, 1)
                loc, rem = local_indices.to("cuda:0"), remote_indices.to("cuda:0")
                local.append(tuple(torch.cat((item.index_select(0, loc), item[:1])) for item in (keys, values)))
                remote.append(tuple(item.index_select(0, rem) for item in (keys, values)))
            self.local.append(FixedBatchKV(concatenate_kv(local, "cuda:0")))
            if mode != "local":
                self.remote.append(concatenate_kv(remote, "cuda:1"))


class Events:
    def __init__(self):
        self.events = {}

    def __call__(self, tag):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[tag] = event

    def intervals(self):
        pairs = {
            "total": ("begin", "end"), "qkv_norm_rope": ("begin", "projected"),
            "write_kv": ("projected", "kv_written"), "attention_total": ("kv_written", "attention_done"),
            "local_attention": ("local_begin", "local_attention_done"),
            "remote_attention": ("query_transferred", "remote_attention_done"),
            "peer_query": ("remote_begin", "query_transferred"),
            "peer_result": ("remote_attention_done", "remote_result_transferred"),
            "merge": ("merge_begin", "merge_done"), "D_wait": ("local_attention_done", "merge_begin"),
            "output_projection_residual": ("attention_done", "projected_out"),
            "mlp_norm_residual": ("projected_out", "end"),
        }
        return {name: self.events[start].elapsed_time(self.events[end])
                for name, (start, end) in pairs.items() if start in self.events and end in self.events}


class Decoder:
    def __init__(self, model, peer, batch=1):
        self.text = model.model.language_model
        self.lm_head = model.lm_head
        cfg = self.text.config
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.hidden_size // self.heads
        self.batch = batch
        self.rope_section = cfg.rope_parameters["mrope_section"]
        self.kernel = DistributedAttention((batch, self.heads, self.dim), self.dim ** -0.5, peer)

    def step(self, token, position, state, step, *, instrument=False):
        x = self.text.embed_tokens(token.view(self.batch, 1))
        cos, sin = self.text.rotary_emb(x, position)
        timings = []
        for index, layer in enumerate(self.text.layers):
            timing = Events() if instrument else None
            if timing:
                timing("begin")
            x_norm = layer.input_layernorm(x)
            block = layer.self_attn
            q = block.q_proj(x_norm).view(self.batch, 1, self.heads, self.dim).transpose(1, 2)
            k = block.k_proj(x_norm).view(self.batch, 1, self.kv_heads, self.dim).transpose(1, 2)
            v = block.v_proj(x_norm).view(self.batch, self.kv_heads, self.dim)
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_section)
            q = q[:, :, 0].contiguous()
            if timing:
                timing("projected")
            local = state.local[index]
            local.put(k[:, :, 0], v, step)
            view = local.view(step, state.local_cuts)
            if timing:
                timing("kv_written")
            if state.remote:
                out = self.kernel.split(q, view, state.remote[index], timing)
            else:
                out = self.kernel.local(q, view)
            if timing:
                timing("attention_done")
            x = x + block.o_proj(out.view_as(x))
            if timing:
                timing("projected_out")
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            if timing:
                timing("end")
                timings.append(timing)
        logits = self.lm_head(self.text.norm(x)[:, 0])
        return (logits[0] if self.batch == 1 else logits), timings


def errors(reference, target):
    difference = reference.float() - target.float()
    return {"relative_l2": float(difference.norm() / reference.float().norm().clamp_min(1e-20)),
            "max_abs": float(difference.abs().max()), "equal": bool(torch.equal(reference, target))}


def positions(length, offset, reserve):
    return (torch.arange(length + offset, length + offset + reserve, device="cuda:0")
            .view(-1, 1, 1, 1).expand(-1, 3, 1, 1))


def native_prompt_cache(model, processor, row, reference, steps):
    output, visual, rope = collect_prompt(model, processor, row, "cuda:0")
    prompt_logit = output.logits[0, 0].clone()
    length = output.past_key_values.get_seq_length()
    if length != row["total_input_tokens"]:
        raise ValueError("Frozen token count and actual KV differ")
    tokens = torch.tensor(reference[:steps], dtype=torch.long, device="cuda:0")
    p = positions(length, rope, steps)
    # The untouched HF SDPA 1-token path is a correctness control for the custom full-FA3 loop.
    # Capture the next logit on a separate cache to keep the original prefill immutable.
    from transformers.cache_utils import DynamicCache
    cloned = DynamicCache(ddp_cache_data=((layer.keys.clone(), layer.values.clone())
                                          for layer in output.past_key_values.layers))
    state = model.model.language_model(input_ids=tokens[0].view(1, 1), position_ids=p[0],
                                       past_key_values=cloned, use_cache=True)
    hf_logit = model.lm_head(state.last_hidden_state[0, 0])
    del state, cloned
    return output.past_key_values, visual, rope, prompt_logit, hf_logit


def timed_steps(decoder, state, tokens, pos, *, profile=False):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    begin.record()
    logits = []
    traces = []
    for index, token in enumerate(tokens.unbind()):
        result, timing = decoder.step(token, pos[index], state, index, instrument=profile)
        logits.append(result)
        if profile:
            traces.append(timing)
    end.record()
    end.synchronize()
    elapsed = (time.perf_counter() - wall) * 1000
    return torch.stack(logits), {"gpu_ms": begin.elapsed_time(end), "wall_ms": elapsed,
                                 "layers": [[event.intervals() for event in step] for step in traces]}


def natural(decoder, state, first, pos, eoses, limit):
    token = first.argmax()
    ids = [int(token)]
    for index in range(limit - 1):
        if ids[-1] in eoses:
            break
        logits, _ = decoder.step(token, pos[index], state, index)
        token = logits.argmax()
        ids.append(int(token))
    return ids


def counter_delta(before, after):
    if before is None or after is None or before.keys() != after.keys():
        return None
    delta = {key: after[key] - before[key] for key in before}
    return delta if min(delta.values(), default=0) >= 0 else None


def link_snapshot():
    """Whole-device NVLink DATA counter in KiB units from the current driver."""
    import pynvml as nvml

    uuids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    try:
        nvml.nvmlInit()
        try:
            values = {}
            for role, uuid in zip(("D", "P"), uuids, strict=True):
                handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
                queries = {"tx": nvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
                           "rx": nvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX}
                for name, kind in queries.items():
                    counters = nvml.nvmlDeviceGetFieldValues(handle, [(kind, link) for link in range(18)])
                    if len(counters) != 18 or any(value.nvmlReturn != nvml.NVML_SUCCESS for value in counters):
                        return None
                    values[f"{role}_{name}_bytes"] = 1024 * sum(int(value.value.ullVal) for value in counters)
            return values
        finally:
            nvml.nvmlShutdown()
    except (OSError, nvml.NVMLError):
        return None


def transfer_probe(cache, decoder, visual, repeats):
    """Measure pure prompt-payload peer transfer with real KV layouts already packed."""
    peer = decoder.kernel.peer
    length = cache.get_seq_length()
    full_tensors = [(item.keys[0].transpose(0, 1).contiguous().to("cuda:1"),
                     item.values[0].transpose(0, 1).contiguous().to("cuda:1"))
                    for item in cache.layers]
    results = {}
    for mode in MODES:
        local, _ = shard_indices(length, visual, mode)
        idx = local.to("cuda:1")
        tensors = [tensor for pair in full_tensors for tensor in pair]
        if mode in ("local", "balanced", "prefix", "equal_count_prefix"):
            # These layouts retain a contiguous prompt prefix on the memory worker.
            sources = [tensor[length - len(local):] for tensor in tensors]
            packing_ms = 0.0
        else:
            with torch.cuda.device(1):
                start = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                start.record()
                sources = [tensor.index_select(0, idx) for tensor in tensors]
                finish.record()
                finish.synchronize()
                packing_ms = start.elapsed_time(finish)
        targets = [torch.empty_like(src, device="cuda:0") for src in sources]
        torch.cuda.synchronize(1)
        transfer = torch.cuda.Stream(device=0)
        results[mode] = {"bytes": sum(bytes_of(src) for src in sources), "cuda_ms": [], "wall_ms": [],
                         "P_packing_cuda_ms": packing_ms}
        if len(local) == 0:
            results[mode].update(cuda_ms=[0.0] * repeats, wall_ms=[0.0] * repeats)
            continue
        before = link_snapshot()
        with torch.cuda.stream(transfer):
            for index in range(repeats + 3):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                start.record()
                for target, source in zip(targets, sources, strict=True):
                    peer(target, source, transfer)
                end.record()
                end.synchronize()
                if index >= 3:
                    results[mode]["cuda_ms"].append(start.elapsed_time(end))
                    results[mode]["wall_ms"].append((time.perf_counter() - begin) * 1000)
        results[mode]["NVLink_data_delta_bytes_repeats_plus_3"] = counter_delta(before, link_snapshot())
        if not all(torch.equal(target, source.to("cuda:0"))
                   for target, source in zip(targets, sources, strict=True)):
            raise ValueError("Initial peer transfer altered KV")
        results[mode]["checked_exact_tensor_copies"] = len(targets)
    return results


def decoder_input(decoder, cache, token, pos, visual):
    """Actual layer-1 Q and KV for a single teacher token after original prefill."""
    layer = decoder.text.layers[0]
    x = decoder.text.embed_tokens(token.view(1, 1))
    cos, sin = decoder.text.rotary_emb(x, pos)
    x = layer.input_layernorm(x)
    projection = layer.self_attn
    q = projection.q_proj(x).view(1, 1, decoder.heads, decoder.dim).transpose(1, 2)
    k = projection.k_proj(x).view(1, 1, decoder.kv_heads, decoder.dim).transpose(1, 2)
    v = projection.v_proj(x).view(1, decoder.kv_heads, decoder.dim)
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, decoder.rope_section)
    keys = torch.cat((cache.layers[0].keys[0].transpose(0, 1), k[0].transpose(0, 1)))
    values = torch.cat((cache.layers[0].values[0].transpose(0, 1), v))
    return q[0].transpose(0, 1).contiguous(), keys, values, visual


def concatenate_kv(parts, device):
    lengths = [len(k) for k, _ in parts]
    keys, values = (torch.cat(tensors).to(device) for tensors in zip(*parts, strict=True))
    return PackedKV(keys, values, lengths)


def pack(cases, mode):
    """Pack actual ragged layer-1 KV in stable positional order after the one-token write."""
    remote_parts = []
    local_parts = []
    for _, k, v, visual in cases:
        local, remote = shard_indices(len(k) - 1, visual, mode)
        local = torch.cat((local, torch.tensor([len(k) - 1]))).to("cuda:0")
        remote = remote.to("cuda:0")
        local_parts.append((k.index_select(0, local), v.index_select(0, local)))
        remote_parts.append((k.index_select(0, remote), v.index_select(0, remote)))

    return concatenate_kv(local_parts, "cuda:0"), concatenate_kv(remote_parts, "cuda:1")


def fixed_ragged_decode(model, peer, cases):
    """Full28 fixed one-token geometry with actual BF16 projection and MLP weights."""
    caches, visuals, deltas, tokens = zip(*cases, strict=True)
    decoder = Decoder(model, peer, batch=len(cases))
    pos = torch.tensor([cache.get_seq_length() + delta for cache, delta in zip(caches, deltas, strict=True)],
                       dtype=torch.long, device="cuda:0").view(1, 1, len(cases), 1).expand(1, 3, -1, 1)
    token = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
    states = {mode: FixedBatchState(caches, visuals, mode) for mode in MODES}
    timings = {mode: [] for mode in MODES}
    aligned_logits = {}
    order_rng = random.Random(2026112705)
    for repetition in range(12):
        order = list(MODES)
        order_rng.shuffle(order)
        for mode in order:
            logits, timing = timed_steps(decoder, states[mode], token, pos)
            if repetition >= 3:
                timings[mode].append(timing)
            if repetition == 0:
                aligned_logits[mode] = logits
    profiles = {}
    for mode in MODES:
        _, profile = timed_steps(decoder, states[mode], token, pos, profile=True)
        profiles[mode] = profile
    return {
        "batch": len(cases), "K_rows_before_decode": [cache.get_seq_length() for cache in caches],
        "layouts": {
            mode: {"timing": timings[mode], "diagnostic_profile": profiles[mode],
                   "logit_error_vs_local": errors(aligned_logits["local"], aligned_logits[mode]),
                   "equal_greedy": int((aligned_logits["local"].argmax(-1) == aligned_logits[mode].argmax(-1)).sum())}
            for mode in MODES
        },
        "scope": "Fixed ragged one-token decode step on real contexts and Full28 weights. No scheduler or continuous batch.",
    }


def fp32_attention(cases):
    outputs = []
    for query, keys, values, _ in cases:
        groups = query.shape[1] // keys.shape[1]
        q = query[0].view(keys.shape[1], groups, query.shape[-1]).float()
        k = keys.transpose(0, 1).float()
        v = values.transpose(0, 1).float()
        weights = torch.softmax((q @ k.transpose(1, 2)) * q.shape[-1] ** -0.5, dim=-1)
        outputs.append((weights @ v).reshape(1, query.shape[1], query.shape[-1]))
    return torch.cat(outputs)


def microbenchmark(peer, cases, *, repeats=16):
    q = torch.cat([item[0] for item in cases])
    path = DistributedAttention(q.shape, q.shape[-1] ** -0.5, peer)
    reference = fp32_attention(cases)
    plans = {mode: pack(cases, mode) for mode in MODES}
    for index in (0, 1):
        torch.cuda.synchronize(index)
    results = {mode: {"wall_ms": [], "CUDA_ms": []} for mode in MODES}
    order_rng = random.Random(2026112704)
    for repeat in range(repeats + 5):
        order = list(MODES)
        order_rng.shuffle(order)
        for mode in order:
            local, remote = plans[mode]
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            begin.record()
            if mode == "local":
                out = path.local(q, local)
            else:
                out = path.split(q, local, remote)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall) * 1000
            if repeat >= 5:
                results[mode]["wall_ms"].append(wall_ms)
                results[mode]["CUDA_ms"].append(begin.elapsed_time(end))
            else:
                results[mode].update(
                    fp32_reference_error=errors(reference, out), output_shape=list(out.shape),
                    D_rows=local.lengths, P_rows=remote.lengths,
                )
    # Amplify tiny messages for whole-device DATA counters in a separate
    # diagnostic interval. These events do not affect the wall/CUDA timings above.
    for mode in ("local", "visual"):
        local, remote = plans[mode]
        count = 128
        before = link_snapshot()
        for _ in range(count):
            if mode == "local":
                path.local(q, local)
            else:
                path.split(q, local, remote)
        torch.cuda.synchronize(0)
        results[mode]["counter_iterations"] = count
        results[mode]["NVLink_data_delta_bytes"] = counter_delta(before, link_snapshot())
    return {"cases": len(cases), "input_context_with_teacher_token": [len(item[1]) for item in cases],
            "recurrent_link_bytes": path.link_bytes, "layouts": results}


def trial(model, processor, decoder, row, reference, steps, repeats, natural_limit):
    known_tokens = reference["token_ids"]
    # The final emitted token never enters the next decode step.
    used = min(len(known_tokens) - 1, steps)
    if used < 1:
        raise ValueError("Choose a representative with at least one consumed generated token")
    cache, visual, delta, first, hf_logit = native_prompt_cache(model, processor, row, known_tokens, used)
    token = torch.tensor(known_tokens[:used], dtype=torch.long, device="cuda:0")
    pos = positions(cache.get_seq_length(), delta, max(used, natural_limit))
    evidence = {
        "request_id": row["request_id"], "input_tokens": row["total_input_tokens"],
        "visual_tokens": len(visual), "native_output_tokens": len(known_tokens), "teacher_steps": used,
        "HF_prompt_argmax": int(first.argmax()), "native_first_token": known_tokens[0],
        "physical_p2p_initial_copy": transfer_probe(cache, decoder, visual, repeats),
        "per_layer_recurrent_link_bytes": decoder.kernel.link_bytes,
        "modes": {},
    }
    for mode in MODES:
        state = DecodeState(cache.layers, row["total_input_tokens"], visual, mode, max(used, natural_limit))
        evidence["modes"][mode] = {"KV": state.allocated_kv(), "timing": []}
        del state
    wall = {mode: [] for mode in MODES}
    steady_logits = {}
    # Reduce drift by rotating the modes between independent allocations.
    ordering = random.Random(2026112703)
    for repetition in range(repeats + 1):
        order = list(MODES)
        ordering.shuffle(order)
        for mode in order:
            base = [torch.cuda.memory_allocated(device) for device in (0, 1)]
            for device in (0, 1):
                torch.cuda.reset_peak_memory_stats(device)
            state = DecodeState(cache.layers, row["total_input_tokens"], visual, mode, max(used, natural_limit))
            output, timing = timed_steps(decoder, state, token, pos)
            if repetition:
                wall[mode].append(timing | {"repetition": repetition,
                                          "D_incremental_allocator_peak_bytes": torch.cuda.max_memory_allocated(0) - base[0],
                                          "P_incremental_allocator_peak_bytes": torch.cuda.max_memory_allocated(1) - base[1]})
            if repetition == 0:
                steady_logits[mode] = output
            del state
    reference_logits = steady_logits["local"]
    evidence["local_first_decode_vs_untouched_HF_SDPA"] = errors(hf_logit, reference_logits[0])
    evidence["local_first_decode_argmax_matches_HF"] = int(reference_logits[0].argmax()) == int(hf_logit.argmax())
    eoses = model.generation_config.eos_token_id
    eoses = {eoses} if isinstance(eoses, int) else set(eoses)
    reference_natural = None
    reference_generated_kv = None
    profiled_steps = min(2, used)
    for mode in MODES:
        mode_evidence = evidence["modes"][mode]
        state = DecodeState(cache.layers, row["total_input_tokens"], visual, mode, max(used, natural_limit))
        _, profiled = timed_steps(decoder, state, token[:profiled_steps], pos, profile=True)
        candidate = steady_logits[mode]
        greedy_match = candidate.argmax(-1) == reference_logits.argmax(-1)
        generated_kv = [torch.stack((layer.keys[layer.prefix : layer.prefix + profiled_steps],
                                     layer.values[layer.prefix : layer.prefix + profiled_steps]))
                        for layer in state.local]
        if mode == "local":
            reference_generated_kv = generated_kv
        disagreement = []
        for index in (~greedy_match).nonzero().flatten().tolist():
            reference_top = reference_logits[index].topk(2)
            disagreement.append({"teacher_step": index, "local_argmax": int(reference_top.indices[0]),
                                 "candidate_argmax": int(candidate[index].argmax()),
                                 "local_top1_margin": float(reference_top.values[0] - reference_top.values[1])})
        mode_evidence.update(
            timing=wall[mode], diagnostic_profile=profiled,
            logits=errors(reference_logits, candidate),
            teacher_greedy_matches=int(greedy_match.sum()), teacher_greedy_count=len(greedy_match),
            teacher_greedy_disagreement=disagreement,
            first_two_teacher_tokens_generated_kv_by_layer=[
                errors(ref, actual) for ref, actual in zip(reference_generated_kv, generated_kv, strict=True)
            ],
        )
        if natural_limit:
            ids = natural(decoder, state, first, pos, eoses, natural_limit)
            if mode == "local":
                reference_natural = ids
            mode_evidence["natural"] = {"token_ids": ids, "EOS": ids[-1] in eoses,
                                        "cap_tokens": natural_limit, "equals_FA3_local": ids == reference_natural,
                                        "text": processor.tokenizer.decode(
                                            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)}
        del state
    layer_input = decoder_input(decoder, cache, token[0], pos[0], visual)
    evidence["layer_1_attention"] = microbenchmark(decoder.kernel.peer, [layer_input])
    return evidence, layer_input, (cache, visual, delta, known_tokens[0])


def choose_requests(tier, reference):
    eligible = [row for row in tier if len(reference[row["request_id"]]["token_ids"]) >= 2]
    input_cut = float(np.median([row["total_input_tokens"] for row in eligible]))
    output_cut = float(np.median([len(reference[row["request_id"]]["token_ids"]) for row in eligible]))
    strata = defaultdict(list)
    for row in eligible:
        count = len(reference[row["request_id"]]["token_ids"])
        strata[(row["total_input_tokens"] >= input_cut, count >= output_cut)].append(row)
    chosen = []
    for group in sorted(strata):
        ordered = sorted(strata[group], key=lambda row: (row["total_input_tokens"], row["request_id"]))
        chosen.append(ordered[len(ordered) // 2])
    longest = max(eligible, key=lambda row: row["total_input_tokens"])
    if longest not in chosen:
        chosen.append(longest)
    return chosen


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tier", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--natural-limit", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1 or args.natural_limit < 0:
        parser.error("At least two teacher steps, one timing repeat and nonnegative Natural cap")
    peer = PeerCopies()
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    decoder = Decoder(model, peer)
    with args.native.open() as stream:
        reference_rows = [json.loads(line) for line in stream]
    reference = {row["request_id"]: row for row in reference_rows if row["status"] == "MEASURED_ONLINE"}
    tier = rt.read(args.tier)["requests"]
    selected = choose_requests(tier, reference)
    if args.smoke:
        selected = selected[:1]
    args.out.mkdir(mode=0o700)
    rt.write(args.out / "experiment.json", {
        "argv": vars(args) | {name: str(getattr(args, name)) for name in ("model", "tier", "native", "out")},
        "tier_SHA256": rt.sha(args.tier), "teacher_SHA256": rt.sha(args.native),
        "first_device": torch.cuda.get_device_name(0), "second_device": torch.cuda.get_device_name(1),
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")},
        "source_SHA256": {name: rt.sha(Path(__file__).with_name(name)) for name in ("track_a.py", "remote_attention.py")},
        "replay": [row["request_id"] for row in selected],
        "scope": "Real Qwen Full28 checkpoint and E/KV. Isolated single-request H200 replay; no vLLM scheduler.",
    })
    layer_inputs, full_inputs = [], []
    for index, row in enumerate(selected):
        result, layer_input, full_input = trial(model, processor, decoder, row, reference[row["request_id"]],
                                                args.steps, args.repeats, args.natural_limit)
        layer_inputs.append(layer_input)
        if len(full_inputs) < 4:
            full_inputs.append(full_input)
        rt.write(args.out / f"case_{index + 1}.json", result)
        print(row["request_id"], [(mode, item["teacher_greedy_matches"], item["teacher_greedy_count"])
                                  for mode, item in result["modes"].items()], flush=True)
    if len(layer_inputs) >= 4:
        rt.write(args.out / "ragged_batch4_attention.json", microbenchmark(peer, layer_inputs[:4]))
        rt.write(args.out / "ragged_batch4_full28.json", fixed_ragged_decode(model, peer, full_inputs))


if __name__ == "__main__":
    main()
