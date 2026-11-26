"""Bounded post-MRoPE attention and residual observations for original Qwen2.5-VL."""

from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


VISUAL, QUESTION, TEXT, SPECIAL, GENERATED = range(5)
TOKEN_CLASSES = ("visual", "question", "other_prompt_text", "special", "generated")
LAYER_GROUPS = tuple((n, n + 4) for n in range(0, 28, 4))
MATCHED_ROWS = 96
MAX_QUERY_ROWS = 16


def spaced(values, count):
    if count >= len(values):
        return list(values)
    if count <= 0:
        return []
    return [values[(2 * i + 1) * len(values) // (2 * count)] for i in range(count)]


def matched_rows(categories, limit=MATCHED_ROWS):
    visual = [i for i, mode in enumerate(categories) if mode == VISUAL]
    text = [i for i, mode in enumerate(categories) if mode in (TEXT, QUESTION)]
    selected_text = spaced(text, min(len(visual), len(text), limit))
    selected_visual = []
    # Independent nearest available position for each fixed text row. Keep the
    # pairing so that both keys are dropped iff both were causally accessible.
    for position in selected_text:
        idx = bisect_left(visual, position)
        choices = range(max(0, idx - 1), min(len(visual), idx + 1))
        selected_visual.append(visual.pop(min(choices, key=lambda i: (abs(position - visual[i]), i))))
    return selected_visual, selected_text


@dataclass(frozen=True)
class Arm:
    operation: str
    category: str
    start: int
    stop: int

    @property
    def label(self):
        return f"{self.operation}_{self.category}_L{self.start + 1:02d}-{self.stop:02d}"


def experiment_arms():
    for start, stop in LAYER_GROUPS:
        for operation in ("attention", "mlp", "kv"):
            for category in ("visual", "text"):
                yield Arm(operation, category, start, stop)
    yield Arm("attention_mlp", "visual_dose55", 20, 28)


def causal_rows(mask, local_rows, query_len, key_len, device):
    positions = key_len - query_len + local_rows
    allowed = torch.arange(key_len, device=device)[None, :] <= positions[:, None]
    if mask is not None:
        if mask.ndim != 4 or mask.shape[:2] != (1, 1):
            raise ValueError("Track B uses single-request, head-shared causal SDPA masks")
        mask = mask[0, 0, :, :key_len].index_select(0, local_rows)
        if mask.dtype == torch.bool:
            allowed = allowed & mask
        else:
            # Qwen's full causal additive mask is zero or the dtype's minimum.
            allowed = allowed & (mask > torch.finfo(mask.dtype).min / 2)
    return allowed


def selected_probabilities(q, k, local_rows, allowed, scaling):
    # q [1, Hq, Q, d], k [1, Hkv, K, d]. The GQA replication order
    # places consecutive query heads in the same key/value head group.
    n_groups, key_count, dim = k.shape[1:]
    rows = q[0].index_select(1, local_rows).float().reshape(n_groups, -1, local_rows.numel(), dim)
    scores = torch.einsum("ghqd,gkd->ghqk", rows, k[0].float()) * scaling
    if allowed.shape != (local_rows.numel(), key_count):
        raise RuntimeError("Causal mask geometry changed")
    return scores.masked_fill(~allowed[None, None], -torch.inf).softmax(dim=-1)


class FlowProbe:
    """A single-request probe, explicitly restored after the HF run."""

    def __init__(self, model, hf_module, device, max_new_tokens):
        self.layers = model.model.language_model.layers
        self.hf_module = hf_module
        self.device = device
        self.max_new_tokens = max_new_tokens
        if len(self.layers) != 28 or any(layer.self_attn.sliding_window for layer in self.layers):
            raise ValueError("Expected Full28 dense Qwen2.5-VL")
        self.arm = None
        self.observe = False
        self.handles = []
        self.inputs = {}

    @contextmanager
    def install(self):
        original_registry = self.hf_module.ALL_ATTENTION_FUNCTIONS
        registry = type(original_registry)()
        original_sdpa = original_registry["sdpa"]
        self.original_sdpa = original_sdpa
        registry["sdpa"] = self.dispatch
        self.hf_module.ALL_ATTENTION_FUNCTIONS = registry
        try:
            for index, layer in enumerate(self.layers):
                self.handles.extend((
                    layer.register_forward_pre_hook(self.layer_input(index), with_kwargs=True),
                    layer.self_attn.register_forward_hook(self.operation(index, "attention")),
                    layer.mlp.register_forward_hook(self.operation(index, "mlp")),
                    layer.register_forward_hook(self.layer_output(index)),
                ))
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.hf_module.ALL_ATTENTION_FUNCTIONS = original_registry

    def prepare(self, original_categories, image_spans=()):
        self.prompt_length = len(original_categories)
        self.categories = torch.tensor(
            original_categories + [GENERATED] * self.max_new_tokens, dtype=torch.long, device=self.device,
        )
        image_of = [-1] * self.categories.numel()
        for index, (start, end) in enumerate(image_spans):
            image_of[start:end] = [index] * (end - start)
        self.image_of = torch.tensor(image_of, dtype=torch.long, device=self.device)
        visual, text = matched_rows(original_categories)
        self.pairs_cpu = (visual, text)
        self.pairs = torch.tensor([visual, text], dtype=torch.long, device=self.device)
        self.dose_cpu = spaced(
            [i for i, token_class in enumerate(original_categories) if token_class == VISUAL],
            round(0.55 * original_categories.count(VISUAL)),
        )
        self.target_queries_cpu = [i for i, value in enumerate(original_categories) if value in (TEXT, QUESTION)]
        self.offset = 0

    def configure(self, arm=None, observation_tokens=0):
        self.arm = arm
        self.observe = observation_tokens > 0
        self.inputs = {}
        self.flow = defaultdict(list)
        self.kv_pairs_removed = []
        self.kv_no_mask_relative_L2 = None
        if self.observe:
            classes = self.categories[: self.prompt_length + observation_tokens - 1].tolist()
            selected = []
            for target in range(len(TOKEN_CLASSES)):
                selected.extend(spaced([i for i, label in enumerate(classes) if label == target], 8))
            self.sampled = torch.tensor(sorted(selected), device=self.device, dtype=torch.long)
            self.sampled_types = self.categories[self.sampled]
        if arm:
            if arm.category == "visual_dose55":
                selected = self.dose_cpu
            else:
                selected = self.pairs_cpu[0 if arm.category == "visual" else 1]
            self.intervention_index = torch.tensor(selected, device=self.device, dtype=torch.long)

    def layer_input(self, index):
        def capture(_, args, kwargs):
            if self.observe:
                value = args[0] if args else kwargs["hidden_states"]
                self.inputs[index] = value[0].index_select(0, self.sampled).float()
        return capture

    def operation(self, index, name):
        def hook(_, args, result):
            value = result[0] if name == "attention" else result
            if self.observe:
                selected = value[0].index_select(0, self.sampled).float()
                reference = self.inputs[index]
                self.flow[index].append((name + "_relative_L2",
                    (selected.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)).cpu()))
            arm = self.arm
            applies = arm and arm.start <= index < arm.stop and (arm.operation == name or arm.operation == "attention_mlp")
            # Every ablated residual row is an original prompt token. Decode
            # retains the unmodified operators in both condition arms.
            if applies and self.offset == 0:
                value = torch.index_fill(value, 1, self.intervention_index, 0.)
                return (value, *result[1:]) if name == "attention" else value
            return None
        return hook

    def layer_output(self, index):
        def hook(_, args, result):
            if not self.observe:
                return
            current = result[0].index_select(0, self.sampled).float()
            before = self.inputs.pop(index)
            if index == 0:
                self.initial_hidden = before
            first = self.initial_hidden
            self.flow[index].extend((
                ("hidden_relative_L2", ((current - before).norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12)).cpu()),
                ("hidden_cosine", F.cosine_similarity(current, before).cpu()),
                ("from_input_relative_L2", ((current - first).norm(dim=-1) / first.norm(dim=-1).clamp_min(1e-12)).cpu()),
                ("from_input_cosine", F.cosine_similarity(current, first).cpu()),
            ))
        return hook

    def dispatch(self, module, q, k, v, attention_mask, **kwargs):
        original, weights = self.original_sdpa(module, q, k, v, attention_mask, **kwargs)
        if not isinstance(module, self.hf_module.Qwen2_5_VLAttention):
            return original, weights
        scaling = kwargs["scaling"]
        if self.observe:
            self.observe_attention(module.layer_idx, q, k, attention_mask, scaling)
        arm = self.arm
        if arm and arm.operation == "kv" and arm.start <= module.layer_idx < arm.stop:
            original = self.intervene_kv(original, q, k, v, attention_mask, scaling)
        return original, weights

    def observe_attention(self, index, q, k, mask, scaling):
        key_classes = self.categories[:k.shape[-2]]
        rows = []
        for selected in self.sampled.split(MAX_QUERY_ROWS):
            allowed = causal_rows(mask, selected, q.shape[-2], k.shape[-2], q.device)
            prob = selected_probabilities(q, k, selected, allowed, scaling).mean(dim=(0, 1))
            mass, counts = [], []
            for token_class in range(len(TOKEN_CLASSES)):
                label_mask = key_classes == token_class
                mass.append((prob * label_mask).sum(dim=-1))
                counts.append((allowed & label_mask).sum(dim=-1))
            image_queries = self.image_of[selected, None]
            image_keys = self.image_of[None, : k.shape[-2]]
            same_image = (image_queries == image_keys) & (image_queries >= 0)
            other_image = (image_queries >= 0) & (image_keys >= 0) & ~same_image
            edges = [(prob * mask).sum(dim=-1) for mask in (same_image, other_image)]
            edge_counts = [(allowed & mask).sum(dim=-1) for mask in (same_image, other_image)]
            rows.append(torch.stack([*mass, *counts, *edges, *edge_counts], dim=-1).cpu())
        self.flow[index].append(("attention", torch.cat(rows, dim=0)))

    def intervene_kv(self, original, q, k, v, mask, scaling):
        query_len, key_len = q.shape[-2], k.shape[-2]
        if not self.pairs.numel():
            return original
        if self.offset == 0:
            # The added teacher prefix consists entirely of generated tokens.
            indices = torch.tensor(
                self.target_queries_cpu + list(range(self.prompt_length, query_len)),
                dtype=torch.long, device=q.device,
            )
        else:
            indices = torch.arange(query_len, device=q.device)
        modified = original.clone()
        chosen_keys = self.pairs[0 if self.arm.category == "visual" else 1]
        visual, text = self.pairs
        value_f32 = v[0].float()
        for selected in indices.split(MAX_QUERY_ROWS):
            if selected.numel() == 0:
                continue
            allowed = causal_rows(mask, selected, query_len, key_len, q.device)
            matched = allowed[:, visual] & allowed[:, text]
            can_change = matched.any(dim=-1)
            altered = allowed.clone()
            altered[:, chosen_keys] &= ~matched
            if self.kv_no_mask_relative_L2 is None:
                # One identical-query no-mask control measures rounding
                # differences from recomputing attention outside fused SDPA.
                probs = selected_probabilities(q, k, selected[:1], allowed[:1], scaling)
                native = original[:, selected[:1]].float()
                safe = torch.einsum("ghqk,gkd->qghd", probs, value_f32).reshape_as(native)
                self.kv_no_mask_relative_L2 = ((native - safe).norm() / native.norm().clamp_min(1e-12)).detach()
            probs = selected_probabilities(q, k, selected, altered, scaling)
            filtered = torch.einsum("ghqk,gkd->qghd", probs, value_f32)
            filtered = filtered.reshape(1, selected.numel(), q.shape[1], q.shape[-1]).to(original.dtype)
            modified[:, selected] = torch.where(can_change[None, :, None, None], filtered, original[:, selected])
            if self.offset == 0:
                self.kv_pairs_removed.append(matched.sum().detach())
        return modified

    def attention_rows(self):
        types = self.sampled_types.tolist()
        indices = self.sampled.tolist()
        image_ids = self.image_of[self.sampled].tolist()
        output = []
        class_count = len(TOKEN_CLASSES)
        for index, values in sorted(self.flow.items()):
            recorded = dict(values)
            attention = recorded.pop("attention").tolist()
            metrics = {key: tensor.tolist() for key, tensor in recorded.items()}
            for position, token_class in enumerate(types):
                mass = attention[position][:class_count]
                eligible = [int(x) for x in attention[position][class_count : 2 * class_count]]
                if abs(sum(mass) - 1) > 5e-5:
                    raise ValueError("Attention mass did not normalize over modeled causal keys")
                output.append(dict(
                    layer=index + 1, group=f"L{(index // 4) * 4 + 1:02d}-{(index // 4) * 4 + 4:02d}",
                    query_index=indices[position], query_type=TOKEN_CLASSES[token_class],
                    key_eligible=dict(zip(TOKEN_CLASSES, eligible, strict=True)),
                    key_mass=dict(zip(TOKEN_CLASSES, mass, strict=True)),
                    query_image_index=image_ids[position] if image_ids[position] >= 0 else None,
                    visual_edges={
                        key: {"mass": attention[position][2 * class_count + i],
                              "eligible": int(attention[position][2 * class_count + 2 + i])}
                        for i, key in enumerate(("same_image", "other_image"))
                    } if image_ids[position] >= 0 else {},
                    alignment={key: vec[position] for key, vec in metrics.items()},
                ))
        return output
