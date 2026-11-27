"""CogVLM2 D-only expert reachability and checkpoint residency experiment."""

import argparse
import ast
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, ModuleType

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer


REPO = "zai-org/cogvlm2-llama3-chat-19B"
REVISION = "f592f291cf528389b2e4776b1e84ecdf6d71fbe3"
DIGESTS = {
    "modeling_cogvlm.py": "25c75840576aacbdee8c7196645b7cf34ac296f8f278bec4cffaefc8830aebf5",
    "configuration_cogvlm.py": "bd34325043bd50926af69f542f879c7e6f462a95dc92b48e18218554684e0d03",
    "visual.py": "529c0543c448975d25746fe4ec4e3d8234d4f2446fe5708a319a88ebc42aca18",
    "util.py": "390811eb63be829b07cd2bdef09bdc31851f93163b24e5a837d16b6757b7e85d",
    "config.json": "0396940e55177ac212fd8ccfcc9df440d2c34b330b2722f8242db598702e5bf3",
    "model.safetensors.index.json": "25d1d78f4e50491a130aa2dfdce4e2e5764d0f32bf44a452747f623d97ec7f5d",
    "LICENSE": "3149da85443e5802bf4e97678c0d1cba8917382ac5aa1ecb2c0353c508c49a14",
}
EXPERT_MODULES = (
    "self_attn.vision_expert_query_key_value", "self_attn.vision_expert_dense", "mlp.vision_mlp",
)
LAYOUTS = ("resident_original", "resident_text_fast", "visual_pruned", "generic_reachability_pruned")
AA_LAYOUT = "resident_text_fast_aa"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_snapshot(directory):
    details = {name: sha(directory / name) for name in DIGESTS}
    if details != DIGESTS:
        raise ValueError("Official CogVLM2 source, license or index differs from the audited pin")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    return json.loads((directory / "config.json").read_text()), index


def generic_inactive_modules(model_source):
    """Derive empty-row module call sites from pinned native layer assignments.

    Both forward functions fill activation rows indexed by a route mask. Taking
    the complement-only route makes their zero-row applications unreachable.
    """
    tree = ast.parse(model_source.read_text())
    modules = []
    for name, parent in (("VisionExpertAttention", "self_attn"), ("VisionExpertMLP", "mlp")):
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        for stmt in forward.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Subscript) or ast.unparse(target.slice) != "vision_token_mask":
                continue
            call = stmt.value
            if (not isinstance(call, ast.Call) or len(call.args) != 1 or not isinstance(call.func, ast.Attribute)
                    or ast.unparse(call.func.value) != "self" or not isinstance(call.args[0], ast.Subscript)
                    or ast.unparse(call.args[0].slice) != "vision_token_mask"):
                raise ValueError("Unexpected zero-row application site")
            modules.append(f"{parent}.{call.func.attr}")
    if set(modules) != set(EXPERT_MODULES):
        raise ValueError("Generic reachability and audited CogVLM2 image-expert branch disagree")
    return tuple(modules)


def keep_parameter(name, layout, inactive):
    if layout in ("P", "monolith_text_fast"):
        return True
    if name.startswith("model.vision."):
        return False
    if not name.startswith("model.layers."):
        return True
    path = name.split(".", 3)[3]
    removed = (() if layout.startswith("resident") else
               EXPERT_MODULES if layout == "visual_pruned" else inactive)
    return not any(path.startswith(module + ".") for module in removed)


def runtime_vision_adapter():
    if importlib.util.find_spec("xformers") is not None:
        return "pinned_source_xformers_vision"
    # The original project's vision tower has one xformers API call. This
    # adapter retains the unmasked attention formula, layout, scale and dtype.
    xformers = ModuleType("xformers")
    ops = ModuleType("xformers.ops")

    def memory_efficient_attention(q, k, v, *, scale):
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                          dropout_p=0, scale=scale)
        return y.transpose(1, 2).contiguous()

    ops.memory_efficient_attention = memory_efficient_attention
    xformers.ops = ops
    sys.modules.update({"xformers": xformers, "xformers.ops": ops})
    return "common_pytorch_sdpa_for_vision_attention_only"


def import_snapshot(directory):
    vision = runtime_vision_adapter()
    namespace = "_verified_cogvlm2_f592f291"
    package = ModuleType(namespace)
    package.__path__ = [str(directory.resolve())]
    sys.modules[namespace] = package
    model_def = importlib.import_module(f"{namespace}.modeling_cogvlm")
    return model_def, vision


def zero_output(rows, width):
    if rows.ndim != 2 or rows.shape[0] != 0:
        raise ValueError("Unexpected nonempty application of a prefilling-only expert on D")
    return rows.new_empty((0, width))


class NoRows(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, rows):
        return zero_output(rows, self.width)


def set_application(layer, name, replacement):
    group, item = name.split(".")
    setattr(getattr(layer, group), item, replacement)


def get_application(layer, name):
    group, item = name.split(".")
    return getattr(getattr(layer, group), item)


def text_attention(self, hidden_states, token_type_ids, position_ids, attention_mask=None,
                   past_key_value=None, output_attentions=False, use_cache=False):
    """Direct language route with the pinned rotary, GQA and attention code."""
    if output_attentions:
        raise ValueError("Auditing requires fused output without attention weights")
    batch, q_len, _ = hidden_states.shape
    q, k, v = self.language_expert_query_key_value(hidden_states.reshape(-1, self.hidden_size)).view(
        batch, q_len, self.qkv_size
    ).split([n * self.head_dim for n in self.stride], dim=-1)
    q, k, v = (self._transpose_for_scores(tensor) for tensor in (q, k, v))
    q, k = self.rotary_emb(q, k, position_ids=position_ids, max_seqlen=position_ids.max() + 1)
    if past_key_value is not None:
        k, v = (torch.cat((old, new), dim=2) for old, new in zip(past_key_value, (k, v), strict=True))
    present = (k, v) if use_cache else None
    multiplicity = self.num_attention_heads // self.num_multi_query_heads
    keys, values = (tensor.unsqueeze(2).expand(-1, -1, multiplicity, -1, -1).contiguous().view(
        batch, self.num_attention_heads, *tensor.shape[2:]) for tensor in (k, v))
    context = self._audited_attention(q, keys, values, attention_mask,
                                      scaling_attention_score=True, attention_dropout=None)
    context = context.transpose(1, 2).contiguous().reshape(batch * q_len, self.hidden_size)
    output = self.language_expert_dense(context).view(batch, q_len, self.hidden_size)
    return output, None, present


def text_mlp(self, hidden_states, token_type_ids):
    return self.language_mlp(hidden_states.flatten(0, 1)).view_as(hidden_states)


def phase_attention(self, hidden_states, token_type_ids, position_ids, **kwargs):
    route = self._original_multimodal_forward if hidden_states.shape[1] > 1 else self._text_forward
    return route(hidden_states, token_type_ids, position_ids, **kwargs)


def phase_mlp(self, hidden_states, token_type_ids):
    route = self._original_multimodal_forward if hidden_states.shape[1] > 1 else self._text_forward
    return route(hidden_states, token_type_ids)


def install_text_route(model, definition, *, phase_aware=False):
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        attn._audited_attention = definition.attention_fn
        attn._original_multimodal_forward = attn.forward
        mlp._original_multimodal_forward = mlp.forward
        attn._text_forward = MethodType(text_attention, attn)
        mlp._text_forward = MethodType(text_mlp, mlp)
        attn.forward = MethodType(phase_attention, attn) if phase_aware else attn._text_forward
        mlp.forward = MethodType(phase_mlp, mlp) if phase_aware else mlp._text_forward


def decode_step(model, definition, token, cache, position, attention_mask, token_types=None):
    if cache is None or len(cache) != len(model.model.layers):
        raise ValueError("D requires complete prefill KV for every layer")
    if any(tensor.device != token.device for tensor in cache[0]):
        raise ValueError("D requires imported KV on the input device")
    if token.ndim != 2 or token.shape[1] != 1 or position.shape != token.shape:
        raise ValueError("D accepts one newly generated token per batch row and explicit positions")
    if token_types is None:
        token_types = torch.full_like(token, definition.LANGUAGE_TOKEN_TYPE)
    elif token_types.shape != token.shape or torch.any(token_types != definition.LANGUAGE_TOKEN_TYPE):
        raise ValueError("D rejects mixed-modality or non-language token roles")
    result = model.model.llm_forward(input_ids=token, token_type_ids=token_types, attention_mask=attention_mask,
                                     position_ids=position, past_key_values=cache, use_cache=True, return_dict=True)
    return model.lm_head(result.last_hidden_state[:, -1]), result.past_key_values


def build_model(definition, config, directory, index, layout, device, inactive):
    with torch.device("meta"):
        model = definition.CogVLMForCausalLM(definition.CogVLMConfig(**config))
    if layout not in ("P", "monolith_text_fast"):
        # Vision encoder residency is irrelevant on all decode arms.
        del model.model.vision
    replaced = EXPERT_MODULES if layout == "visual_pruned" else inactive
    if layout.endswith("_pruned"):
        for layer in model.model.layers:
            for name in replaced:
                original = get_application(layer, name)
                width = original.out_features if isinstance(original, nn.Linear) else original.down_proj.out_features
                set_application(layer, name, NoRows(width))
    expected = {name for name in index["weight_map"] if keep_parameter(name, layout, inactive)}
    declared = set(model.state_dict())
    if expected != declared:
        raise ValueError(f"Checkpoint/module mismatch for {layout}: {sorted(expected ^ declared)}")
    selected = defaultdict(list)
    for name in sorted(expected):
        selected[index["weight_map"][name]].append(name)
    weights = {}
    for shard, names in selected.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as file:
            for name in names:
                weights[name] = file.get_tensor(name).to(device)
    model.load_state_dict(weights, strict=True, assign=True)
    if layout in ("resident_text_fast", AA_LAYOUT, "monolith_text_fast") or layout.endswith("_pruned"):
        install_text_route(model, definition, phase_aware=layout == "monolith_text_fast")
    recorded = {name: {"bytes": t.numel() * t.element_size(), "dtype": str(t.dtype)}
                for name, t in model.state_dict().items()}
    return model.eval(), recorded


def select_prompts(path, ids, count=None):
    rows = json.loads(path.read_text())["requests"]
    lookup = {r["request_id"]: r for r in rows if r["image_count"] == 1}
    if ids:
        return [lookup[key] for key in ids]
    if count is not None:
        grouped = defaultdict(list)
        for row in lookup.values():
            grouped[row["source"]].append(row)
        names = sorted(grouped)
        selected = []
        for index in range(max(len(group) for group in grouped.values())):
            for name in names:
                if index < len(grouped[name]):
                    selected.append(grouped[name][index])
        if len(selected) < count:
            raise ValueError("Requested nominal count exceeds eligible frozen one-image roster")
        return selected[:count]
    policies = (
        lambda r: "DocVQA" in r["source"] and "MP-DocVQA" not in r["source"],
        lambda r: "MMMU" in r["source"],
        lambda r: "MathVista" in r["source"],
        lambda r: "MMReason" in r["source"],
    )
    chosen = []
    for policy in policies:
        candidates = sorted((r for r in lookup.values() if policy(r)), key=lambda row: row["request_id"])
        if candidates:
            chosen.append(candidates[len(candidates) // 2])
    if not chosen:
        raise ValueError("No eligible one-image requests")
    return chosen


def prefill(model, tokenizer, definition, row):
    query = row.get("question") or row["question_original"]
    if not row["images"]:
        prepared = model.build_conversation_input_ids(tokenizer, query=query, images=None)
    else:
        img, = row["images"]
        path = Path(img["path"])
        if sha(path) != img["SHA256"]:
            raise ValueError("Original image file hash differs from frozen cohort")
        with Image.open(path) as file:
            prepared = model.build_conversation_input_ids(tokenizer, query=query, images=[file.convert("RGB")])
    device = next(model.parameters()).device
    ids, types, mask = (prepared[k][None].to(device) for k in ("input_ids", "token_type_ids", "attention_mask"))
    position = definition.build_position_ids(types, mask)
    if len(types[0]) >= model.config.max_position_embeddings:
        raise ValueError("Prompt exceeds the supported model context")
    processing = prepared["images"]
    images = [[image.to(device, dtype=torch.bfloat16) for image in processing]] if processing else None
    raw = model.model(input_ids=ids, token_type_ids=types, position_ids=position, attention_mask=mask,
                      images=images,
                      use_cache=True, return_dict=True)
    first = model.lm_head(raw.last_hidden_state[:, -1]).argmax(-1)
    return {
        "request_id": row["request_id"], "model_input_tokens": ids.shape[1],
        "input_image_type_rows": int((types == definition.VISION_TOKEN_TYPE).sum()),
        "visual_expert_prefill_rows": int(definition.get_expert_mask(types)[0].sum()),
        "last_position": int(position[0, -1]), "first_id": first, "kv": raw.past_key_values,
    }


def exact_copy(cache, device):
    moved = tuple(tuple(t.to(device, non_blocking=True) for t in layer) for layer in cache)
    torch.cuda.synchronize(device)
    return moved


def physical_copy(cache):
    begin = time.perf_counter_ns()
    moved = exact_copy(cache, "cuda:1")
    wall_ms = (time.perf_counter_ns() - begin) / 1e6
    match = sum(torch.equal(source, target.to(source.device))
                for orig, copy in zip(cache, moved, strict=True) for source, target in zip(orig, copy, strict=True))
    tensors = sum(len(layer) for layer in cache)
    if match != tensors:
        raise ValueError(f"Prompt import altered KV: {match}/{tensors}")
    return moved, {
        "wall_ms": wall_ms, "tensors_equal": match, "tensor_count": tensors,
        "logical_peer_bytes": sum(t.numel() * t.element_size() for layer in cache for t in layer),
    }


def generation(model, definition, source, batch, steps, *, compute_greedy=False, teachers=None):
    """All rows are consumed generated language tokens, including special IDs."""
    home = str(next(model.parameters()).device)
    cache = tuple(tuple(t.expand(batch, -1, -1, -1).contiguous() for t in layer) for layer in source["kv"])
    token = source["first_id"].to(home).view(1, 1).expand(batch, 1).contiguous()
    counterexample_ids = torch.tensor([model.config.pad_token_id, model.config.bos_token_id,
                                       *model.config.eos_token_id], device=home)
    negative_route = definition.get_expert_mask(torch.full_like(counterexample_ids.view(-1, 1),
                                                                definition.LANGUAGE_TOKEN_TYPE))[0]
    if negative_route.any():
        raise ValueError("A generated special ID incorrectly changed expert reachability")
    total = source["model_input_tokens"]
    attend = torch.ones((batch, total + steps), dtype=torch.bool, device=home)
    positions = torch.arange(source["last_position"] + 1, source["last_position"] + steps + 1,
                             dtype=torch.long, device=home)
    logits = []
    outputs = []
    eoses = tuple(model.config.eos_token_id)
    torch.cuda.synchronize(home)
    start = time.perf_counter_ns()
    for index in range(steps):
        if teachers is not None:
            token = teachers[index].to(home).view(1, 1).expand(batch, 1).contiguous()
        position = positions[index:index + 1].view(1, 1).expand(batch, 1).contiguous()
        scores, cache = decode_step(model, definition, token, cache, position,
                                    attend[:, :total + index + 1])
        token = scores.argmax(-1, keepdim=True)
        logits.append(scores)
        outputs.append(token)
        if compute_greedy and int(token[0, 0]) in eoses:
            break
    torch.cuda.synchronize(home)
    wall = (time.perf_counter_ns() - start) / 1e6
    return wall, torch.stack(logits).cpu(), torch.stack(outputs).cpu(), cache


def special_token_trial(model, definition, source):
    """Consume real generated-vocabulary IDs with one language role per query."""
    device = next(model.parameters()).device
    ids = [model.config.pad_token_id, model.config.bos_token_id, *model.config.eos_token_id]
    token = torch.tensor(ids, dtype=torch.long, device=device).view(-1, 1)
    pos = torch.full_like(token, source["last_position"] + 1)
    cache = tuple(tuple(t.expand(len(ids), -1, -1, -1).contiguous() for t in layer)
                  for layer in source["kv"])
    mask = torch.ones((len(ids), source["model_input_tokens"] + 1), dtype=torch.bool, device=device)
    types = torch.full_like(token, definition.LANGUAGE_TOKEN_TYPE)
    types[0, 0] = definition.VISION_TOKEN_TYPE
    try:
        decode_step(model, definition, token, cache, pos, mask, token_types=types)
    except ValueError as error:
        if "non-language" not in str(error):
            raise
    else:
        raise ValueError("Decode unexpectedly admitted a non-language role")
    output, _ = decode_step(model, definition, token, cache, pos, mask)
    return ids, output.cpu()


def pack_prefills(sources, device):
    """Pack useful imported rows for a fixed ragged batch, retaining left-pad masks."""
    length = max(src["model_input_tokens"] for src in sources)
    mask = torch.zeros((len(sources), length), dtype=torch.bool, device=device)
    device = torch.device(device)
    for row, src in enumerate(sources):
        mask[row, length - src["model_input_tokens"]:] = True
        if src["kv"][0][0].device != device:
            raise ValueError("All ragged source cache tensors must already be on the target device")
    packed = []
    for layer in zip(*(src["kv"] for src in sources), strict=True):
        combined = []
        for tensors in zip(*layer, strict=True):
            combined.append(torch.cat([F.pad(t, (0, 0, length - t.shape[-2], 0)) if t.shape[-2] < length else t
                                       for t in tensors], dim=0))
        packed.append(tuple(combined))
    positions = torch.tensor([src["last_position"] + 1 for src in sources], dtype=torch.long,
                             device=device).view(-1, 1)
    token = torch.stack([src["first_id"].to(device)[0] for src in sources]).view(-1, 1)
    return tuple(packed), mask, positions, token


def real_ragged_trial(model, definition, imports, bases, steps, repeats):
    device = next(model.parameters()).device
    prefix, mask, pos, source_token = pack_prefills(imports, device)
    teachers = torch.stack([base[0][:steps] for base in bases], dim=1).to(device)
    mask = F.pad(mask, (0, steps), value=True)
    timings, result = [], None
    for repetition in range(repeats + 1):
        cache = prefix
        logits = []
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        for step in range(steps):
            token = teachers[step].view_as(source_token)
            scores, cache = decode_step(model, definition, token, cache, pos + step,
                                        mask[:, :mask.shape[1] - steps + step + 1])
            logits.append(scores)
        torch.cuda.synchronize(device)
        if repetition:
            timings.append((time.perf_counter_ns() - start) / 1e6)
        else:
            result = torch.stack(logits).cpu()
    return {
        "request_ids": [src["request_id"] for src in imports],
        "model_input_lengths": [src["model_input_tokens"] for src in imports],
        "actual_batch": len(imports), "steps": steps, "wall_ms": timings,
        "median_ms_per_step": statistics.median(timings) / steps,
        "teacher_logits": result,
    }


def compare(reference, actual):
    ref, out = reference.float(), actual.float()
    delta = ref - out
    return {"elementwise_equal": torch.equal(reference, actual),
            "finite": bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()),
            "max_abs": delta.abs().amax().item(),
            "relative_l2": (delta.norm() / ref.norm().clamp_min(1e-10)).item()}


def cache_alloc(source, weight_bytes, budget, reserve):
    rows = source["model_input_tokens"]
    sample = source["kv"][0][0]
    kv_per_row = len(source["kv"]) * 2 * sample.shape[1] * sample.shape[-1] * sample.element_size()
    per_sequence = kv_per_row * (rows + reserve)
    # Explicit common budget. This reserves a real contiguous KV tensor; it
    # omits paged fragmentation and cannot predict scheduler concurrency.
    slots = (budget - weight_bytes) // per_sequence
    if slots < 1:
        raise ValueError("The common memory budget cannot hold this route")
    before = torch.cuda.memory_allocated(1)
    cache = torch.empty((slots, len(source["kv"]), 2, sample.shape[1], rows + reserve, sample.shape[-1]),
                        dtype=sample.dtype, device=sample.device)
    cache.zero_()
    torch.cuda.synchronize(1)
    actual = torch.cuda.memory_allocated(1) - before
    payload = cache.numel() * cache.element_size()
    if actual < payload:
        raise ValueError("Allocator did not record the requested KV storage")
    del cache
    return {"allocated_slots": slots, "allocation_bytes": actual, "logical_KV_per_slot": per_sequence,
            "common_route_budget_bytes": budget, "decode_reserved_output_per_slot": reserve}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--count", type=int)
    parser.add_argument("--ragged-batch", type=int, default=4)
    parser.add_argument("--mode", choices=(*LAYOUTS, AA_LAYOUT), action="append", default=[])
    parser.add_argument("--aa-control", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--text-guard", action="store_true")
    parser.add_argument("--batch", type=int, action="append", default=[])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--natural-steps", type=int, default=48)
    parser.add_argument("--memory-budget-gib", type=float, default=64)
    parser.add_argument("--kv-reserve", type=int, default=128)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 2 or args.natural_steps < 2:
        parser.error("Use at least two repeats, two timed steps and two Natural steps")
    if args.count is not None and (args.id or args.count < 1):
        parser.error("--count selects a positive number of requests without manual IDs")
    if args.ragged_batch < 2 or any(batch <= 0 for batch in args.batch):
        parser.error("Use positive fixed batches and a ragged-batch target of at least two")
    cfg, index = checked_snapshot(args.snapshot)
    inactive = generic_inactive_modules(args.snapshot / "modeling_cogvlm.py")
    included = {mode: sorted(k for k in index["weight_map"] if keep_parameter(k, mode, inactive))
                for mode in (*LAYOUTS, AA_LAYOUT, "P")}
    if included["visual_pruned"] != included["generic_reachability_pruned"]:
        raise ValueError("Generic reachability selected a different weight set")
    summary = {"checkpoint": REPO, "revision": REVISION, "source_hashes": DIGESTS,
               "model_config": cfg, "checkpoint_bytes": index["metadata"]["total_size"],
               "runner_sha256": sha(Path(__file__)),
               "load_keys": included, "unreachable_applications": list(inactive)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.source_only:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        return
    torch.set_num_threads(4)
    if args.deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG on the deterministic experiment command")
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
    if torch.cuda.device_count() != 2 or not torch.cuda.can_device_access_peer(1, 0):
        raise RuntimeError("Expose two leased CUDA devices with peer access")
    torch.cuda.set_device(0)
    definition, vision = import_snapshot(args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    prompts = select_prompts(args.manifest, args.id, count=args.count)
    p, p_weights = build_model(definition, cfg, args.snapshot, index, "P", "cuda:0", inactive)
    # Test the two-token visual-pair condition in addition to single-query
    # false reachability; output-vocabulary IDs never choose this axis.
    types = torch.tensor([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1]])
    mask, language = definition.get_expert_mask(types)
    if mask.tolist() != [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0]] or not torch.equal(language, ~mask):
        raise ValueError("Upstream visual-pair rule differs from source proof")
    samples = [prefill(p, tokenizer, definition, row) for row in prompts]
    bases, imports = [], []
    for input_state in samples:
        teacher = generation(p, definition, input_state, 1, args.steps)[2].flatten()
        teacher = torch.cat((input_state["first_id"].cpu(), teacher[:-1]))
        natural = generation(p, definition, input_state, 1, args.natural_steps, compute_greedy=True)
        device_state = dict(input_state)
        device_state["kv"], receipt = physical_copy(input_state["kv"])
        bases.append((teacher, natural[2]))
        imports.append((device_state, receipt))
    summary.update({
        "evidence": "ISOLATED_GPU_MEASUREMENT",
        "vision_path": vision, "torch": torch.__version__,
        "deterministic_algorithms": args.deterministic,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu": [{"uuid": str(torch.cuda.get_device_properties(i).uuid), "name": torch.cuda.get_device_name(i)}
                for i in (0, 1)],
        "manifest_sha256": sha(args.manifest), "p_weight_bytes": sum(d["bytes"] for d in p_weights.values()),
        "cases": [{k: v for k, v in src.items() if k not in ("first_id", "kv")} | {"P_to_D_KV": receipt}
                  for src, receipt in imports],
        "layouts": {},
    })
    if args.text_guard:
        probe = {"request_id": "text_only_route_guard", "images": [],
                 "question": "Which number is greater, twelve or nine? Answer in one sentence."}
        text_p = prefill(p, tokenizer, definition, probe)
        if text_p["input_image_type_rows"] or text_p["visual_expert_prefill_rows"]:
            raise ValueError("The official publisher processed a text-only prompt as vision")
        text_d = dict(text_p)
        text_d["kv"], copy = physical_copy(text_p["kv"])
        reference_text = generation(p, definition, text_p, 1, min(args.steps, 4))
        summary["text_only_guard"] = {
            "evidence": "REAL_TEXT_ONLY_PUBLISHER_PREFILL_DIAGNOSTIC",
            "prefill": {k: v for k, v in text_p.items() if k not in ("kv", "first_id")},
            "K_V_copy": copy, "teacher_count": min(args.steps, 4),
        }
        text_teachers = torch.cat((text_p["first_id"].cpu(), reference_text[2].flatten()[:-1]))
        del text_p, reference_text
    order = args.mode or list(LAYOUTS)
    if args.aa_control and AA_LAYOUT not in order:
        order.append(AA_LAYOUT)
    refs = {}
    for mode in order:
        begin_bytes = torch.cuda.memory_allocated(1)
        d, resident = build_model(definition, cfg, args.snapshot, index, mode, "cuda:1", inactive)
        torch.cuda.synchronize(1)
        weight_bytes = sum(t["bytes"] for t in resident.values())
        arm = {"parameter_plus_buffer_bytes": weight_bytes,
               "D_allocator_load_growth_bytes": torch.cuda.memory_allocated(1) - begin_bytes,
               "loaded_vision_expert_tensors": sum(not keep_parameter(key, "visual_pruned", inactive)
                                                   for key in resident),
               "cases": []}
        for index_no, (src, _) in enumerate(imports):
            teachers, reference_natural = bases[index_no]
            _, scores, predicts, generated = generation(d, definition, src, 1, args.steps, teachers=teachers)
            _, natural_scores, natural, _ = generation(d, definition, src, 1, args.natural_steps, compute_greedy=True)
            rows = {"request_id": src["request_id"], "teacher_predictions": predicts.flatten().tolist(),
                    "natural_predictions": natural.flatten().tolist(),
                    "equal_to_local_P_natural": torch.equal(reference_natural, natural),
                    "natural_ended_eos": int(natural[-1, 0, 0]) in tuple(d.config.eos_token_id),
                    "batch_timings": {},
                    "teacher_scores": scores, "natural_scores": natural_scores,
                    "generated_KV": tuple(tuple(t.cpu() for t in layer) for layer in generated)}
            for batch in args.batch or [1, 4, 8]:
                generation(d, definition, src, batch, args.steps, teachers=teachers)
                raw = [generation(d, definition, src, batch, args.steps, teachers=teachers)[0]
                       for _ in range(args.repeats)]
                rows["batch_timings"][batch] = {"wall_ms": raw, "steps": args.steps,
                                                "median_ms_per_step": statistics.median(raw) / args.steps}
            arm["cases"].append(rows)
        if mode == order[0] or mode == "resident_text_fast":
            refs[mode] = [{k: row[k] for k in ("teacher_scores", "natural_scores", "generated_KV")}
                          for row in arm["cases"]]
        for i, case in enumerate(arm["cases"]):
            teacher = case.pop("teacher_scores")
            natural_logits = case.pop("natural_scores")
            cache = case.pop("generated_KV")
            for arm_name, suffix in ((order[0], "first_arm"), ("resident_text_fast", "fast_resident")):
                if arm_name not in refs:
                    continue
                base = refs[arm_name][i]
                case[f"teacher_logits_vs_{suffix}"] = compare(base["teacher_scores"], teacher)
                common = min(len(natural_logits), len(base["natural_scores"]))
                case[f"natural_shared_prefix_logits_vs_{suffix}"] = compare(
                    base["natural_scores"][:common], natural_logits[:common])
                case[f"final_teacher_kv_equal_layers_vs_{suffix}"] = sum(
                    all(torch.equal(a, b) for a, b in zip(x, y, strict=True))
                    for x, y in zip(base["generated_KV"], cache, strict=True))
            del cache
        del generated
        if args.text_guard:
            _, text_logits, text_predictions, _ = generation(d, definition, text_d, 1, min(args.steps, 4),
                                                               teachers=text_teachers)
            if mode == order[0]:
                resident_text_logits = text_logits
            arm["text_only_decode_guard"] = {
                "teacher_logits_vs_first_arm": compare(resident_text_logits, text_logits),
                "teacher_predictions": text_predictions.flatten().tolist(),
            }
        special_ids, special_logits = special_token_trial(d, definition, imports[0][0])
        if mode == order[0]:
            special_reference = special_logits
        if mode == "resident_text_fast":
            fast_special_reference = special_logits
        arm["explicit_decode_role_validation"] = {
            "mixed_roles_rejected": True,
            "consumed_language_role_special_ids": special_ids,
            "logits_vs_first_arm": compare(special_reference, special_logits),
        }
        if "resident_text_fast" in refs:
            arm["explicit_decode_role_validation"]["logits_vs_fast_resident"] = compare(
                fast_special_reference, special_logits)
        if len(imports) >= args.ragged_batch:
            ragged = real_ragged_trial(d, definition, [src for src, _ in imports[:args.ragged_batch]],
                                       bases[:args.ragged_batch], min(args.steps, 4), args.repeats)
            logits = ragged.pop("teacher_logits")
            if mode == order[0]:
                ragged_reference = logits
            if mode == "resident_text_fast":
                fast_ragged_reference = logits
            if "resident_text_fast" in refs:
                ragged["logits_vs_fast_resident"] = compare(fast_ragged_reference, logits)
            ragged["matches_first_arm_argmax_count"] = int((ragged_reference.argmax(-1) == logits.argmax(-1)).sum())
            ragged["logits_vs_first_arm"] = compare(ragged_reference, logits)
            arm["actual_ragged_batch"] = ragged
        src = max((source for source, _ in imports), key=lambda r: r["model_input_tokens"])
        arm["KV_capacity_trial"] = cache_alloc(src, weight_bytes, int(args.memory_budget_gib * 2**30),
                                              args.kv_reserve)
        summary["layouts"][mode] = arm
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(mode, "cases", len(arm["cases"]), "D bytes", weight_bytes, flush=True)
        del d, arm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
