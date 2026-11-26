"""Checkpoint-backed, two-device Qwen2.5-VL decoder-layer diagnostic.

Timing uses warmed resident weights, real BF16 operators and separate CUDA
streams per device. It has no encoder, decode loop, server or KV page manager.
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm.vllm_flash_attn import flash_attn_varlen_func


class Layer:
    def __init__(self, weights, cfg, device, tp_rank=None):
        self.hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.nq, self.nk = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.eps = cfg["rms_norm_eps"]
        self.input_norm = weights["input_layernorm.weight"].to(device)
        self.post_norm = weights["post_attention_layernorm.weight"].to(device)
        if tp_rank is None:
            q, k, m = slice(None), slice(None), slice(None)
        else:
            q = rank_slice(cfg["hidden_size"], tp_rank)
            k = rank_slice(self.nk * self.hd, tp_rank)
            m = rank_slice(cfg["intermediate_size"], tp_rank)
            self.nq //= 2
            self.nk //= 2
        self.qkv = torch.cat([
            weights[f"self_attn.{key}_proj.weight"][sl]
            for key, sl in zip("qkv", (q, k, k))
        ]).to(device)
        self.bias = torch.cat([
            weights[f"self_attn.{key}_proj.bias"][sl]
            for key, sl in zip("qkv", (q, k, k))
        ]).to(device)
        self.out = weights["self_attn.o_proj.weight"][:, q].contiguous().to(device)
        self.gate_up = torch.cat([
            weights[f"mlp.{key}_proj.weight"][m] for key in ("gate", "up")
        ]).to(device)
        self.down = weights["mlp.down_proj.weight"][:, m].contiguous().to(device)

    def project(self, x, rope):
        norm = F.rms_norm(x, (x.shape[-1],), self.input_norm, self.eps)
        q, k, v = F.linear(norm, self.qkv, self.bias).split(
            [self.nq * self.hd, self.nk * self.hd, self.nk * self.hd], dim=-1
        )
        q = apply_rope(q.reshape(-1, self.nq, self.hd), rope)
        k = apply_rope(k.reshape(-1, self.nk, self.hd), rope)
        v = v.contiguous().view(-1, self.nk, self.hd)
        return q, k, v

    def apply_out(self, a):
        return F.linear(a.flatten(1), self.out)

    def ffn(self, x):
        norm = F.rms_norm(x, (x.shape[-1],), self.post_norm, self.eps)
        g, u = F.linear(norm, self.gate_up).chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, self.down)


def rank_slice(size, rank):
    if size % 2:
        raise ValueError(f"size {size} cannot be partitioned into two equal ranks")
    return slice(rank * (size // 2), (rank + 1) * (size // 2))


def load_layers(snapshot, layers):
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = {f"model.layers.{layer}.": layer for layer in layers}
    keys = {key: name for key, name in weight_map.items() if key.startswith(tuple(prefix))}
    tensors = {layer: {} for layer in layers}
    for shard in sorted(set(keys.values())):
        with safe_open(snapshot / shard, framework="pt", device="cpu") as checkpoint:
            for name, file in keys.items():
                if file == shard:
                    for group, layer in prefix.items():
                        if name.startswith(group):
                            tensors[layer][name.removeprefix(group)] = checkpoint.get_tensor(name)
    return tensors, {key: keys[key] for key in sorted(keys)}


def source_rows(path):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data["requests"]


def select_shapes(rows, ids):
    """Deterministic metadata selection before performance measurements."""
    if ids:
        lookup = {x["request_id"]: x for x in rows}
        return [lookup[key] for key in ids]
    selected = []
    for lo, hi in ((1, 1), (2, 8), (9, 10_000)):
        eligible = sorted(
            (x for x in rows if lo <= x["image_count"] <= hi),
            key=lambda row: (row["total_input_tokens"], row["request_id"]),
        )
        if eligible:
            selected.append(eligible[len(eligible) // 2])
    return selected


def position_indices(row, merge):
    length = row["total_input_tokens"]
    spans = row["image_visual_spans"]
    grids = row["grid_thw"]
    if len(spans) != len(grids):
        raise ValueError("visual spans and image grids differ")
    pos = torch.empty((3, length), dtype=torch.long)
    offset = cursor = 0
    for (lo, hi), (t, h, w) in zip(spans, grids):
        h, w = h // merge, w // merge
        if t != 1 or not cursor <= lo < hi <= length or t * h * w != hi - lo:
            raise ValueError(f"unsupported still-image span or grid in {row['request_id']}")
        pos[:, cursor:lo] = offset + torch.arange(lo - cursor)
        offset += lo - cursor
        pos[0, lo:hi] = offset
        pos[1, lo:hi] = offset + torch.arange(h).repeat_interleave(w)
        pos[2, lo:hi] = offset + torch.arange(w).repeat(h)
        offset += max(h, w)
        cursor = hi
    pos[:, cursor:] = offset + torch.arange(length - cursor)
    return pos


def frozen_positions(row, merge):
    path = Path(row["token_metadata_file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["token_metadata_SHA256"]:
        raise ValueError(f"frozen token metadata changed for {row['request_id']}")
    with np.load(path) as data:
        pos = torch.from_numpy(data["positions"][:, 0, :])
    if not torch.equal(pos, position_indices(row, merge)):
        raise ValueError(f"reconstructed MRoPE disagrees with frozen metadata: {row['request_id']}")
    return pos


class VisionInputs:
    """Original HF processor, vision weights and token table, outside timing."""
    def __init__(self, snapshot):
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True).vision_config
        cfg._attn_implementation = "sdpa"
        vision = Qwen2_5_VisionTransformerPretrainedModel(cfg)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        selected = {k: v for k, v in index.items() if k.startswith("visual.") or k == "model.embed_tokens.weight"}
        state = {}
        for shard in sorted(set(selected.values())):
            with safe_open(snapshot / shard, framework="pt", device="cpu") as f:
                for key, path in selected.items():
                    if path != shard:
                        continue
                    if key == "model.embed_tokens.weight":
                        self.tokens = f.get_tensor(key)
                    else:
                        state[key.removeprefix("visual.")] = f.get_tensor(key)
        vision.load_state_dict(state, strict=True, assign=True)
        self.model = vision.eval().to("cuda:0")

    def __call__(self, row):
        from PIL import Image

        pictures = []
        for img in row["images"]:
            path = Path(img["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != img["SHA256"]:
                raise ValueError(f"frozen image file changed: {path.name}")
            with Image.open(path) as f:
                pictures.append(f.convert("RGB"))
        encoded = self.processor(text=[row["prompt"]], images=pictures, return_tensors="pt")
        ids, grid = encoded["input_ids"][0], encoded["image_grid_thw"]
        with np.load(row["token_metadata_file"]) as data:
            if not torch.equal(ids, torch.from_numpy(data["input_ids"])):
                raise ValueError(f"processor changed token IDs for {row['request_id']}")
            if not torch.equal(grid, torch.from_numpy(data["image_grid_thw"])):
                raise ValueError(f"processor changed grid for {row['request_id']}")
        pixels = encoded["pixel_values"].to("cuda:0", dtype=self.model.dtype)
        visual = self.model(pixels, grid_thw=grid.to("cuda:0")).pooler_output.cpu()
        x = F.embedding(ids, self.tokens)
        mask = ids == self.processor.image_token_id
        if int(mask.sum()) != len(visual) or visual.shape[-1] != x.shape[-1]:
            raise ValueError(f"encoder output shape differs from frozen spans for {row['request_id']}")
        x[mask] = visual
        return x, mask


def rope_table(position, cfg):
    dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    inv = cfg["rope_theta"] ** (-torch.arange(0, dim, 2).float() / dim)
    angles = position[:, :, None].float() * inv
    angles = torch.cat((angles, angles), dim=-1)
    sections = cfg["rope_scaling"]["mrope_section"] * 2
    return tuple(torch.cat([m[i % 3] for i, m in enumerate(fn(angles).split(sections, dim=-1))], -1)
                 .to(torch.bfloat16)[:, None] for fn in (torch.cos, torch.sin))


def apply_rope(x, rope):
    a, b = x.chunk(2, dim=-1)
    return x * rope[0] + torch.cat((-b, a), dim=-1) * rope[1]


def rope_slice(rope, lo, hi=None):
    return tuple(table[lo:hi] for table in rope)


def attn(q, k, v, lengths, *, causal=True, lse=False):
    return flash_attn_varlen_func(
        q, k, v, len(q), lengths[len(q)], len(k), lengths[len(k)],
        causal=causal, return_softmax_lse=lse, fa_version=3,
    )


def merge_attention(left, right):
    left_out, left_lse = left
    right_out, right_lse = right
    total = torch.logaddexp(left_lse, right_lse)
    coeff = lambda lse: (lse - total).exp().T[..., None]
    return (left_out.float() * coeff(left_lse) + right_out.float() * coeff(right_lse)).bfloat16()


def movement(x):
    return x.numel() * x.element_size()


def placement_cuts(row):
    length = row["total_input_tokens"]
    # Both policies use the same candidate count and arithmetic shape objective.
    # Candidate 0.46/0.50/0.54 straddles the approximately balanced critical path.
    generic = [max(1, min(length - 1, round(length * f))) for f in (.46, .50, .54)]
    span_ends = sorted({end for _, end in row["image_visual_spans"] if 0 < end < length})
    near = sorted(span_ends, key=lambda p: (abs(p - length / 2), p))[:3]
    best = min(generic, key=lambda p: (abs(p - length / 2), p))
    arms = {"generic_46": generic[0], "generic_50": generic[1], "generic_54": generic[2]}
    if near:
        boundary = min(near, key=lambda p: (abs(p - best), p))
        arms.update({
            "boundary_balanced": boundary,
            "boundary_same_cut_label_off": boundary,
            "last_image_then_remaining_tokens": span_ends[-1],
        })
    return arms


class Context:
    def __init__(self, x, ro, cut):
        self.owners = [torch.device(f"cuda:{i}") for i in range(2)]
        self.x = [x.to(device) for device in self.owners]
        self.ro = [tuple(t.to(device) for t in ro) for device in self.owners]
        self.n, self.p = len(x), cut
        self.cu = [{n: torch.tensor([0, n], dtype=torch.int32, device=device)
                    for n in {self.n, self.p, self.n - self.p}}
                   for device in self.owners]
        self.stream = [torch.cuda.Stream(device) for device in self.owners]
        self.sync()

    def use(self, index):
        return torch.cuda.stream(self.stream[index])

    def copy(self, x, target):
        source = x.device.index
        if source == target:
            return x
        # torch's inter-device copy brackets source and destination current
        # streams with peer events. Set both streams before submitting it.
        with self.use(source), self.use(target):
            return x.to(self.owners[target], non_blocking=True)

    def sync(self):
        for device in self.owners:
            torch.cuda.synchronize(device)


def one_layer(x, rope, length_table, layer, record=None):
    if record is not None:
        record.append(torch.cuda.Event(enable_timing=True))
        record[-1].record()
    q, k, v = layer.project(x, rope)
    if record is not None:
        record.append(torch.cuda.Event(enable_timing=True))
        record[-1].record()
    a = attn(q, k, v, length_table)
    if record is not None:
        record.append(torch.cuda.Event(enable_timing=True))
        record[-1].record()
    x = x + layer.apply_out(a)
    if record is not None:
        record.append(torch.cuda.Event(enable_timing=True))
        record[-1].record()
    x = x + layer.ffn(x)
    if record is not None:
        record.append(torch.cuda.Event(enable_timing=True))
        record[-1].record()
    return x


def single(ctx, layers):
    with ctx.use(0):
        x = ctx.x[0]
        for layer in layers[0]:
            x = one_layer(x, ctx.ro[0], ctx.cu[0], layer)
    return x, 0


def pipeline(ctx, layers):
    nlayer = len(layers[0])
    with ctx.use(0):
        x = ctx.x[0]
        for layer in layers[0][:nlayer // 2]:
            x = one_layer(x, ctx.ro[0], ctx.cu[0], layer)
        remote = ctx.copy(x, 1)
    size = movement(remote)
    with ctx.use(1):
        for layer in layers[1][nlayer // 2:]:
            remote = one_layer(remote, ctx.ro[1], ctx.cu[1], layer)
        result = ctx.copy(remote, 0)
    return result, 2 * size


def token_split(ctx, layers, *, context_parallel=False, local=False):
    p = ctx.p
    with ctx.use(0):
        x = ctx.x[0][:p]
    with ctx.use(1) if not local else ctx.use(0):
        y = ctx.x[int(not local)][p:]
    target = int(not local)
    size = 0
    for pl, sl in zip(layers[0], layers[target]):
        with ctx.use(0):
            pq, pk, pv = pl.project(x, rope_slice(ctx.ro[0], 0, p))
        with ctx.use(target):
            sq, sk, sv = sl.project(y, rope_slice(ctx.ro[target], p))
            if context_parallel:
                q_on_p = ctx.copy(sq, 0)
                size += movement(q_on_p)
                own_partial = attn(sq, sk, sv, ctx.cu[target], lse=True)
        with ctx.use(0):
            if context_parallel:
                on_prefix = attn(q_on_p, pk, pv, ctx.cu[0], causal=False, lse=True)
                remote = tuple(ctx.copy(t, target) for t in on_prefix)
                size += sum(movement(t) for t in remote)
            elif not local:
                remote = tuple(ctx.copy(t, target) for t in (pk, pv))
                size += sum(movement(t) for t in remote)
            else:
                remote = (pk, pv)
            first = attn(pq, pk, pv, ctx.cu[0])
            x = x + pl.apply_out(first)
            x = x + pl.ffn(x)
        with ctx.use(target):
            if context_parallel:
                second = merge_attention(remote, own_partial)
            else:
                k, v = torch.cat((remote[0], sk)), torch.cat((remote[1], sv))
                second = attn(sq, k, v, ctx.cu[target])
            y = y + sl.apply_out(second)
            y = y + sl.ffn(y)
    with ctx.use(target):
        back = ctx.copy(y, 0)
        size += movement(back) if not local else 0
    with ctx.use(0):
        return torch.cat((x, back)), size


def tensor_parallel(ctx, layers):
    # Replicated inputs/residuals. Two P2P point-to-point reductions per layer,
    # with one rank's BF16 partial moved in each direction at each reduction.
    def join(partials, inputs):
        copies = []
        for rank in range(2):
            with ctx.use(rank):
                copies.append(ctx.copy(partials[rank], 1 - rank))
        result = []
        for rank in range(2):
            with ctx.use(rank):
                merged = (partials[rank].float() + copies[1 - rank].float()).bfloat16()
                result.append(inputs[rank] + merged)
        return result, sum(movement(x) for x in copies)

    x = ctx.x[:]
    size = 0
    for pair in zip(*layers):
        partials = []
        for rank, layer in enumerate(pair):
            with ctx.use(rank):
                q, k, v = layer.project(x[rank], ctx.ro[rank])
                partials.append(layer.apply_out(attn(q, k, v, ctx.cu[rank])))
        x, moved = join(partials, x)
        size += moved
        partials = []
        for rank, layer in enumerate(pair):
            with ctx.use(rank):
                partials.append(layer.ffn(x[rank]))
        x, moved = join(partials, x)
        size += moved
    return x[0], size


def difference(reference, candidate):
    a, b = reference.float(), candidate.float()
    d = (a - b).abs()
    return {
        "relative_l2": (d.norm() / a.norm()).item(),
        "max_abs": d.max().item(),
        "changed_fraction": (a != b).float().mean().item(),
        "finite": bool(torch.isfinite(b).all().item()),
    }


def evaluate(ctx, operations, warmups, repeats, seed):
    evaluated = {}
    for name, run in operations.items():
        for _ in range(warmups):
            run()
            ctx.sync()
        resident = [torch.cuda.memory_allocated(device) for device in ctx.owners]
        for device in ctx.owners:
            torch.cuda.reset_peak_memory_stats(device)
        output, moved = run()
        ctx.sync()
        extra = [torch.cuda.max_memory_allocated(device) - before
                 for device, before in zip(ctx.owners, resident)]
        evaluated[name] = {
            "wall_ms": [], "bytes_p2p_logical": moved, "out": output,
            "extra_peak_allocated_bytes_by_device": extra,
        }
    order = list(operations)
    random_order = random.Random(seed)
    for repeat in range(repeats):
        random_order.shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            output, moved = operations[name]()
            ctx.sync()
            evaluated[name]["wall_ms"].append((time.perf_counter_ns() - start) / 1e6)
    return evaluated


def segments(ctx, layer, inputs, repeats):
    result = {label: [] for label in inputs}
    names = list(inputs)
    with ctx.use(0):
        for x in inputs.values():
            for _ in range(3):
                one_layer(x, rope_slice(ctx.ro[0], 0, len(x)), ctx.cu[0], layer)
        ctx.stream[0].synchronize()
        for i in range(repeats):
            for label in names[i % len(names):] + names[:i % len(names)]:
                x = inputs[label]
                events = []
                one_layer(x, rope_slice(ctx.ro[0], 0, len(x)), ctx.cu[0], layer, events)
                ctx.stream[0].synchronize()
                result[label].append(dict(zip(("rms_qkv_mrope_ms", "attention_ms", "o_residual_ms", "mlp_ms"),
                                              (a.elapsed_time(b) for a, b in zip(events, events[1:])))))
    return result


def run(args):
    torch.set_num_threads(4)
    snapshot = args.snapshot
    full_cfg = json.loads((snapshot / "config.json").read_text())
    cfg = full_cfg.get("text_config", full_cfg)
    cpu_weights, keys = load_layers(snapshot, args.layers)
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Expose exactly the two reserved GPUs for this run")
    if not all(torch.cuda.can_device_access_peer(a, b) for a, b in ((0, 1), (1, 0))):
        raise RuntimeError("Bidirectional CUDA P2P is needed for this diagnostic")
    full = [[Layer(cpu_weights[i], cfg, f"cuda:{rank}") for i in args.layers] for rank in range(2)]
    tp = [[Layer(cpu_weights[i], cfg, f"cuda:{rank}", rank) for i in args.layers] for rank in range(2)]
    shapes = select_shapes(source_rows(args.manifest), args.id)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    vision = VisionInputs(snapshot) if args.real_activations else None
    receipt = {
        "evidence": "ISOLATED_GPU_MEASUREMENT",
        "activation_source": (
            "HF_ORIGINAL_PROCESSOR_AND_VISION_BF16_WITH_FROZEN_TOKEN_IDS_AND_MROPE"
            if vision else "SYNTHETIC_ACTIVATION_REAL_CHECKPOINT_REAL_MANIFEST_GEOMETRY"
        ),
        "model_revision": snapshot.name,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "weight_index_sha256": hashlib.sha256((snapshot / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "tensors": keys,
        "layers": args.layers,
        "batch": 1,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "attention": "vllm_flash_attn_varlen_fa3_bf16",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "gpu": [{"name": torch.cuda.get_device_name(i), "uuid": str(torch.cuda.get_device_properties(i).uuid)}
                for i in range(2)],
        "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cases": [],
    }
    for row in shapes:
        length = row["total_input_tokens"]
        if vision:
            x, image_mask = vision(row)
        else:
            x = torch.randn((length, cfg["hidden_size"]), generator=generator, dtype=torch.bfloat16) * 0.10
            image_mask = None
        ro = rope_table(frozen_positions(row, full_cfg["vision_config"]["spatial_merge_size"]), cfg)
        cuts = placement_cuts(row)
        case = {"request_id": row["request_id"], "source": row["source"], "image_count": row["image_count"],
                "length": length, "visual_tokens": row["visual_tokens"], "image_visual_spans": row["image_visual_spans"],
                "grid_thw": row["grid_thw"], "cuts": cuts, "arms": {}}
        primary = Context(x, ro, cuts["generic_50"])
        operations = {
            "single_full": lambda: single(primary, full),
            "pp2_serial_layers": lambda: pipeline(primary, full),
            "tp2_p2p_bf16_partials": lambda: tensor_parallel(primary, tp),
            "cp2_partial_lse": lambda: token_split(primary, full, context_parallel=True),
        }
        for name, cut in cuts.items():
            ctx = Context(x, ro, cut)
            operations[name] = lambda ctx=ctx: token_split(ctx, full)
        local = Context(x, ro, cuts["generic_50"])
        operations["local_split_50"] = lambda: token_split(local, full, local=True)
        results = evaluate(primary, operations, args.warmups, args.repeats, args.seed)
        pairs = {
            "generic_50_vs_same_cut_on_one_gpu": ("local_split_50", "generic_50"),
        }
        if "boundary_balanced" in cuts:
            pairs["same_cut_labels"] = ("boundary_same_cut_label_off", "boundary_balanced")
        case["control_pairs"] = {
            label: difference(results[left]["out"], results[right]["out"])
            for label, (left, right) in pairs.items()
        }
        changed = x.clone()
        changed[primary.p:] *= -1
        changed_ctx = Context(changed, ro, primary.p)
        changed_out, _ = token_split(changed_ctx, full)
        changed_ctx.sync()
        original_out = results["generic_50"]["out"]
        case["causal_negative_control"] = {
            "changed_input": "all_suffix_rows_sign_flipped",
            "prefix": difference(original_out[:primary.p], changed_out[:primary.p]),
            "suffix": difference(original_out[primary.p:], changed_out[primary.p:]),
        }
        reference = results["single_full"]["out"]
        for name, result in results.items():
            summary = difference(reference, result.pop("out"))
            summary["passes_diagnostic_2pct_l2"] = summary["finite"] and summary["relative_l2"] <= 0.02
            result["median_ms"] = statistics.median(result["wall_ms"])
            result["p95_ms"] = sorted(result["wall_ms"])[math.ceil(.95 * args.repeats) - 1]
            result["state_difference_vs_single_full"] = summary
        # Same operation, input values, shape, MRoPE and weights under two labels.
        segment_inputs = {"full": primary.x[0]}
        reference_shape = primary.x[0][:primary.p]
        segment_inputs.update({"vision_label_only": reference_shape, "text_label_only": reference_shape})
        if vision:
            n = min(64, int(image_mask.sum()), int((~image_mask).sum()))
            primary.cu[0][n] = torch.tensor([0, n], dtype=torch.int32, device="cuda:0")
            case["content_control_rows"] = n
            segment_inputs["actual_vision_rows_compacted"] = x[image_mask][:n].to("cuda:0")
            segment_inputs["actual_nonvision_rows_compacted"] = x[~image_mask][:n].to("cuda:0")
        primary.sync()
        case["segments"] = segments(primary, full[0][0], segment_inputs, args.repeats)
        case["arms"] = results
        receipt["cases"].append(case)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(row["request_id"], length, "measured", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--real-activations", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seed", type=int, default=2026112701)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if len(args.layers) < 2 or args.warmups < 1 or args.repeats < 2:
        parser.error("at least 2 layers, 1 warmup and 2 repeats are required")
    with torch.inference_mode():
        run(args)


if __name__ == "__main__":
    main()
