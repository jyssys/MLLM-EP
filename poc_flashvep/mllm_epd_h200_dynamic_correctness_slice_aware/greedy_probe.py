"""Mixed-batch geometry and first-Greedy-divergence diagnostics."""

import os
import re
import time

from ..mllm_epd_prefill_heavy_token_split.batch_prefix_probe import request_base
from ..mllm_epd_prefill_heavy_token_split.common import ROOT, digest, write


def validate_plan(plan):
    if len(plan) > 8:
        raise ValueError("At most eight targeted requests per diagnostic")
    for target in plan.values():
        limit = target["steps"]
        prefix = target.get("prefix", [])
        layers = target.get("layer_steps", [])
        if not 1 <= limit <= 64 or len(prefix) >= limit or any(
            type(token) is not int or token < 0 for token in prefix
        ):
            raise ValueError("Invalid teacher token sequence")
        if len(set(layers)) != len(layers) or any(type(step) is not int or not 0 <= step < limit for step in layers):
            raise ValueError("Invalid layer capture steps")


class GreedyProbe:
    def __init__(self, worker, name, ids, targets):
        if os.environ.get("MLLM_EPD_LATE_CORRECTNESS") != "1":
            raise RuntimeError("Greedy diagnostics require an explicit correctness-only worker")
        self.role = {"prefill": "P", "decode": "D"}[os.environ["MLLM_EPD_ROLE"]]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("Invalid diagnostic name")
        self.ids = frozenset(ids)
        if not self.ids or not set(targets) <= self.ids or (self.role == "P" and targets):
            raise ValueError("Invalid diagnostic request scope")
        validate_plan(targets)
        self.runner = worker.model_runner
        self.model = self.runner.model
        self.targets = targets
        self.eager_layers = any(target.get("layer_steps") for target in targets.values())
        if self.eager_layers and not self.runner.model_config.enforce_eager:
            raise ValueError("Layer hooks require eager execution to avoid changing the compiled graph")
        self.folder = ROOT / "raw/greedy_probe" / name
        self.folder.mkdir(parents=True)
        self.patch_slots = []
        self.handles = []
        self.events = []
        self.samples = {}
        self.tensors = {}
        self.selected = []
        self.layer_selected = []
        self.layer_tensors = {}

    def patch(self, obj, name, wrapper):
        original = getattr(obj, name)
        self.patch_slots.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, wrapper(original))

    def save_layers(self, module, layer, names, projection=False):
        def hook(_module, _args, output):
            if not self.layer_selected:
                return
            if projection:
                output = output[0].split([module.q_size, module.kv_size, module.kv_size], dim=-1)
            values = output if isinstance(output, (tuple, list)) else (output,)
            if len(values) != len(names):
                raise RuntimeError(f"Unexpected Qwen2 layer {layer} output count")
            for row in self.layer_selected:
                prefix = f"{row['base']}.{row['step']}.layer{layer:02d}"
                for name, value in zip(names, values, strict=True):
                    self.layer_tensors[f"{prefix}.{name}"] = value[row["last_query_row"]].detach().clone()

        return hook

    def attach_layers(self):
        layers = self.model.language_model.model.layers
        if len(layers) != 28:
            raise RuntimeError("Layer diagnostics require the original 28-layer language model")
        for index, layer in enumerate(layers):
            attention = layer.self_attn
            capture_rotary = self.save_layers(attention, index, ("rotary_Q_input", "rotary_K_input"))

            def before_rotary(module, args, capture=capture_rotary):
                capture(module, (), args[1:3])

            self.handles.append(attention.rotary_emb.register_forward_pre_hook(before_rotary))
            for module, names, projection in (
                (attention.qkv_proj, ("Q_pre_rope", "K_pre_rope", "V"), True),
                (attention.rotary_emb, ("Q", "K"), False),
                (attention.attn, ("attention_kernel_output",), False),
                (attention, ("attention_projection_output",), False),
                (layer, ("MLP_output", "residual"), False),
            ):
                hook = self.save_layers(attention, index, names, projection)
                self.handles.append(module.register_forward_hook(hook))

    def begin(self):
        from vllm.forward_context import get_forward_context

        def wrap_execute(original):
            def execute(scheduled, *args, **kwargs):
                self.scheduled = scheduled
                return original(scheduled, *args, **kwargs)

            return execute

        def wrap_forward(original):
            def forward(*args, **kwargs):
                context = get_forward_context()
                batch = self.runner.input_batch
                rows = []
                offset = 0
                self.selected = []
                self.layer_selected = []
                for index, rid in enumerate(batch.req_ids):
                    base = request_base(rid, self.role, self.ids)
                    count = int(self.scheduled.num_scheduled_tokens[rid])
                    length = int(batch.num_prompt_tokens[index])
                    start = int(batch.num_computed_tokens_cpu[index])
                    step = start + count - length
                    row = dict(
                        base=base, start=start, query_tokens=count, prompt_tokens=length,
                        position=index, last_query_row=offset + count - 1, step=step,
                    )
                    target = self.targets.get(base)
                    if target is not None and 0 <= step < target["steps"]:
                        self.selected.append(row)
                        if step in target.get("layer_steps", []):
                            self.layer_selected.append(row)
                    rows.append(row)
                    offset += count
                if offset != self.scheduled.total_num_scheduled_tokens:
                    raise RuntimeError("Inconsistent scheduled query geometry")
                event = dict(
                    event=len(self.events), worker_entry=time.monotonic(), requests=rows,
                    actual_query_rows=offset, padded_query_rows=int(kwargs["positions"].shape[-1]),
                    graph_mode=str(context.cudagraph_runtime_mode),
                    graph_descriptor=str(context.batch_descriptor),
                )
                self.events.append(event)
                self.layer_tensors = {}
                result = original(*args, **kwargs)
                if self.selected:
                    input_ids = self.runner.input_ids.gpu
                    for row in self.selected:
                        row["actual_last_input_id"] = int(input_ids[row["last_query_row"]].item())
                        row["mrope_position_ids"] = kwargs["positions"][..., row["last_query_row"]].tolist()
                for name, tensor in self.layer_tensors.items():
                    self.tensors[name] = tensor.cpu().contiguous()
                self.layer_selected = []
                self.layer_tensors = {}
                return result

            return forward

        def wrap_logits(original):
            def compute_logits(hidden, *args, **kwargs):
                logits = original(hidden, *args, **kwargs)
                if self.selected and len(self.runner.input_batch.req_ids) != logits.shape[0]:
                    raise RuntimeError("Unexpected mixed-batch logit row allocation")
                for row in self.selected:
                    key = f"{row['base']}.{row['step']}"
                    position = row["position"]
                    output = logits[position].detach().float()
                    values, indices = output.topk(8)
                    self.samples[key] = dict(
                        event=self.events[-1]["event"], token_ids=indices.tolist(),
                        logits=values.tolist(), top1_top2_margin=float((values[0]-values[1]).item()),
                        input_token_id=row["actual_last_input_id"],
                        mrope_position_ids=row["mrope_position_ids"],
                    )
                    self.tensors[key + ".final_hidden"] = hidden[position].detach().cpu().contiguous().clone()
                    self.tensors[key + ".logits"] = output.cpu().contiguous().clone()
                return logits

            return compute_logits

        def wrap_sample(original):
            def sample(logits, spec_decode_metadata):
                sampled = original(logits, spec_decode_metadata)
                if self.selected and (
                    spec_decode_metadata is not None or sampled.sampled_token_ids.shape[1] != 1
                ):
                    raise RuntimeError("Teacher forcing supports one greedy token per request per step")
                for row in self.selected:
                    base, step, position = row["base"], row["step"], row["position"]
                    key = f"{base}.{step}"
                    token_ids = sampled.sampled_token_ids
                    self.samples[key]["greedy_id"] = int(token_ids[position, 0].item())
                    prefix = self.targets[base].get("prefix", [])
                    if step < len(prefix):
                        if prefix[step] >= logits.shape[-1]:
                            raise ValueError("Teacher token is outside the model vocabulary")
                        self.samples[key]["forced_id"] = prefix[step]
                        token_ids[position, 0] = prefix[step]
                return sampled

            return sample

        try:
            if self.eager_layers:
                self.attach_layers()
            self.patch(self.runner, "execute_model", wrap_execute)
            self.patch(self.runner, "_model_forward", wrap_forward)
            if self.targets:
                self.patch(self.model, "compute_logits", wrap_logits)
                self.patch(self.runner, "_sample", wrap_sample)
        except BaseException:
            self.restore()
            raise
        return {"folder": str(self.folder), "role": self.role}

    def restore(self):
        for handle in self.handles:
            handle.remove()
        for obj, method, existed, value in reversed(self.patch_slots):
            if existed:
                setattr(obj, method, value)
            else:
                delattr(obj, method)

    def close(self):
        self.restore()
        receipt = {}
        if self.tensors:
            from safetensors.torch import save_file

            path = self.folder / "tensors.safetensors"
            save_file(self.tensors, str(path))
            receipt = {"path": str(path), "sha256": digest(path)}
        output = {
            "role": self.role, "events": self.events, "samples": self.samples,
            "receipt": receipt, "eager_layer_hooks": self.eager_layers,
            "teacher_forced": any(target.get("prefix") for target in self.targets.values()),
        }
        write(self.folder / "manifest.json", output)
        return {"folder": str(self.folder), "forwards": len(self.events), "captured_steps": len(self.samples)}
