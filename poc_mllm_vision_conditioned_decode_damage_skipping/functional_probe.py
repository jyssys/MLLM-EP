"""Exact bounded drop-one intervention and pre-forcing logit capture.

The intervention changes only one already-selected expert path.  It removes
that path, scales every remaining original weight by ``1/(1-w_drop)``, and
does not select a replacement expert.  The hook is diagnostic and installed
only when explicitly enabled by the task wrapper.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

_INSTALLED = False
_CONTEXT = threading.local()


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_DAMAGE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fingerprints(tensor: torch.Tensor) -> list[str]:
    bits = tensor.detach()[:, :32].contiguous().view(torch.int16).cpu().numpy()
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in bits]


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _marker(payload: dict[str, Any]) -> None:
    root = Path(os.environ["MLLM_DAMAGE_MARKERS"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{payload['probe_id']}.pid{os.getpid()}.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel

    original_model_forward = Qwen3MoeLLMModel.forward
    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        _CONTEXT.probe = entry
        return original_model_forward(self, *args, **kwargs)

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._damage_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_damage_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        entry = getattr(_CONTEXT, "probe", {})
        if (
            not entry.get("intervene", False)
            or int(getattr(_CONTEXT, "layer", -1)) != int(entry.get("target_layer", -2))
        ):
            return original_experts(self, *args, **kwargs)
        names = (
            "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
            "topk_ids", "activation", "global_num_experts", "local_num_experts",
            "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
        )
        values = dict(zip(names, args, strict=False))
        values.update(kwargs)
        x = values["a1q"]
        if x.numel() == 0:
            return original_experts(self, *args, **kwargs)
        hashes = _fingerprints(x)
        target_hash = str(entry["target_hidden_hash"])
        matches = [index for index, value in enumerate(hashes) if value == target_hash]
        if not matches:
            return original_experts(self, *args, **kwargs)
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous target fingerprint {target_hash}: {matches}")
        row = matches[0]
        weights = values["topk_weights"].clone()
        ids = values["topk_ids"]
        drop_weight = float(entry["target_weight"])
        denominator = 1.0 - drop_weight
        if denominator <= 0:
            raise RuntimeError(f"invalid drop weight {drop_weight}")
        weights[row] /= denominator
        target_expert = int(entry["target_expert_id"])
        local_target = (ids[row] == target_expert) & (values["topk_weights"][row] > 0)
        weights[row, local_target] = 0
        modified = original_experts(
            self,
            values["in_dtype"], x, values.get("a1q_scale"), values["w1"], values["w2"],
            weights, ids, values["activation"], values["global_num_experts"],
            values["local_num_experts"], values.get("expert_map"),
            values["apply_router_weight_on_input"], values.get("expert_tokens_meta"),
        )
        _marker({
            "probe_id": entry["probe_id"],
            "pid": os.getpid(),
            "layer": int(entry["target_layer"]),
            "target_expert_id": target_expert,
            "target_weight": drop_weight,
            "fingerprint": target_hash,
            "matched_rows": 1,
            "owned_target": bool(local_target.any()),
            "renormalization": "remaining_original_weights_sum_to_one",
        })
        return modified

    Qwen3MoeLLMModel.forward = patched_model_forward
    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts


from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class DamageProbeLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        forced = extra.get("damage_forced_token_ids")
        if forced is None:
            return
        if not isinstance(forced, list) or not forced:
            raise ValueError("damage_forced_token_ids must be non-empty")
        probe_step = int(extra["damage_probe_step"])
        if probe_step < 0 or probe_step >= len(forced):
            raise ValueError((probe_step, len(forced)))

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        from vllm.distributed import get_tensor_model_parallel_rank

        extra = params.extra_args or {}
        forced = extra.get("damage_forced_token_ids")
        if forced is None:
            return None
        forced_ids = tuple(int(value) for value in forced)
        probe_step = int(extra["damage_probe_step"])
        output = Path(str(extra["damage_logits_path"]))
        probe_id = str(extra["damage_probe_id"])

        def process(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            step = len(output_ids)
            if step == probe_step and get_tensor_model_parallel_rank() == 0 and not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(output.suffix + ".tmp")
                target = forced_ids[step]
                values = logits.detach().float().cpu().numpy()
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        logits=values,
                        target_token_id=np.asarray(target, dtype=np.int64),
                        probe_step=np.asarray(step, dtype=np.int64),
                        probe_id=np.asarray(probe_id),
                    )
                temporary.replace(output)
            if step < len(forced_ids):
                token = forced_ids[step]
                logits.fill_(-float("inf"))
                logits[token] = 0.0
            return logits

        return process
